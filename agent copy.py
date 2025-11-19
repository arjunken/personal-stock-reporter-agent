#!/usr/bin/env python3
"""
agent.py (HTML email version) — OpenRouter edition (reworked)
- Fetches news (Finnhub or Google RSS fallback)
- Calls OpenRouter (if configured) to analyze articles into JSON
- Persists a markdown file, renders styled HTML via Jinja2, and sends via Resend
- Timezone-aware guard for 14:00 America/Vancouver with --force override for local testing
"""

import os
import json
import time
import re
import feedparser
import requests
from pathlib import Path
from datetime import datetime, timedelta, timezone, UTC
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from markdown import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape

# ---------- Memory helpers (paste near imports) ----------
import json
from pathlib import Path
from datetime import datetime, timezone, date, UTC

# Resend SDK
import resend

# CLI flags
import argparse
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--force", action="store_true", help="Force run regardless of timezone guard (local testing)")
_known_args, _unknown = parser.parse_known_args()
FORCE_RUN_FLAG = bool(_known_args.force)
FORCE_RUN_ENV = os.getenv("FORCE_RUN", "") in ("1", "true", "True")

load_dotenv()  # loads .env into environment for local testing

# ----- Config / paths -----
COMPANIES_PATH = "companies.json"
TEMPLATES_DIR = "templates"
TEMPLATE_FILE = "report_template.html"
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL_DEFAULT = "openrouter/polaris-alpha"
MEMORY_PATH = Path("memory.json")
DEBUG_RESPONSES_PATH = Path("openrouter_debug_responses.json")

MEMORY_PATH = Path("memory.json")

def load_memory(path: Path = MEMORY_PATH):
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # If the file is corrupt, back it up and return empty
        try:
            path.rename(path.with_suffix(".json.bak"))
        except Exception:
            pass
        return {}

def save_memory(memory: dict, path: Path = MEMORY_PATH):
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def sentiment_to_score(sentiment_field):
    """
    Convert model sentiment text (e.g., "Positive — reason") to numeric score.
    - Positive -> +1
    - Neutral  ->  0
    - Negative -> -1
    If the model returns a numeric hint, try to parse it.
    Default fallback = 0
    """
    if not sentiment_field:
        return 0.0
    s = str(sentiment_field).lower()
    # quick keyword-based mapping
    if "positive" in s:
        return 1.0
    if "negative" in s:
        return -1.0
    if "neutral" in s:
        return 0.0
    # try to parse a numeric token if model returned one (rare)
    import re
    m = re.search(r"([-+]?\d+(\.\d+)?)", s)
    if m:
        try:
            v = float(m.group(1))
            # clamp to -1..1 if it's obviously a score
            if v > 1: v = 1.0
            if v < -1: v = -1.0
            return v
        except Exception:
            pass
    # default neutral
    return 0.0

def compute_company_sentiment_from_analyses(analyses):
    """
    Given a list like [{'article': {...}, 'analysis': {'summary':..., 'sentiment':...}}, ...],
    return (avg_score, combined_summary).
    """
    scores = []
    summaries = []
    for item in analyses:
        an = item.get("analysis") or {}
        # if analysis is a dict and has a sentiment/summary keys
        sentiment_text = ""
        if isinstance(an, dict):
            sentiment_text = an.get("sentiment") or ""
            summary_text = an.get("summary") or an.get("raw") or ""
        else:
            summary_text = str(an)
        # parse and collect
        score = sentiment_to_score(sentiment_text)
        scores.append(score)
        if summary_text:
            # keep short summaries only
            summaries.append(summary_text.strip())
    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    # make brief combined summary (first 2 summaries joined)
    combined = "  ".join(summaries[:2]) if summaries else ""
    return avg_score, combined

def update_memory_with_results(company_results: dict, memory_path: Path = MEMORY_PATH):
    """
    - company_results: dict mapping company -> list of article analyses (as in your existing code)
    This function:
      1) loads memory.json,
      2) for each company computes avg sentiment & combined summary,
      3) appends an entry {date, sentiment, summary} to memory[company],
      4) saves memory.json,
      5) returns a dict of trends: {company: {"previous": prev_sentiment, "current": curr_sentiment, "delta": delta}}
    """
    mem = load_memory(memory_path)
    trends = {}
    today = datetime.now(UTC).date().isoformat()  # YYYY-MM-DD in UTC
    for company, analyses in company_results.items():
        avg_score, combined_summary = compute_company_sentiment_from_analyses(analyses)
        # get history list
        hist = mem.get(company, [])
        prev_sentiment = None
        if hist and isinstance(hist, list):
            # take last recorded sentiment
            try:
                prev_sentiment = float(hist[-1].get("sentiment", 0.0))
            except Exception:
                prev_sentiment = None
        # append new record
        entry = {"date": today, "sentiment": avg_score, "summary": combined_summary}
        hist.append(entry)
        mem[company] = hist
        # compute delta if previous exists
        if prev_sentiment is None:
            delta = None
        else:
            delta = round(avg_score - prev_sentiment, 4)
        trends[company] = {"previous": prev_sentiment, "current": avg_score, "delta": delta}
    # save back memory
    save_memory(mem, memory_path)
    return trends
# ---------- end memory helpers ----------

# ---- Helpers ----
def get_env(name, default=None):
    v = os.getenv(name)
    if v is None:
        return default
    return v

def load_companies():
    with open(COMPANIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_email_to_list(v):
    if not v:
        return []
    try:
        parsed = json.loads(v)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, str):
            return [parsed]
    except Exception:
        return [x.strip() for x in v.split(",") if x.strip()]
    return []

# ---- Finnhub news (and Google RSS fallback) ----
FINNHUB_SEARCH_URL = "https://finnhub.io/api/v1/search"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"

def looks_like_ticker(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,6}", s.strip()))

def resolve_symbol_via_finnhub(name_or_symbol: str, finnhub_api_key: str):
    candidate = name_or_symbol.strip()
    if looks_like_ticker(candidate):
        return candidate
    if not finnhub_api_key:
        return None
    try:
        params = {"q": candidate, "token": finnhub_api_key}
        r = requests.get(FINNHUB_SEARCH_URL, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        results = data.get("result") or data.get("results") or []
        if not results:
            return None
        top = results[0]
        return top.get("symbol")
    except Exception as e:
        print("Finnhub search error:", e)
        return None

def finnhub_company_news(symbol: str, days_back: int=7, max_items: int=5, finnhub_api_key: str=None):
    if not finnhub_api_key or not symbol:
        return []
    try:
        to_date = datetime.now(UTC).date()
        from_date = to_date - timedelta(days=days_back)
        params = {
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": finnhub_api_key
        }
        r = requests.get(FINNHUB_COMPANY_NEWS_URL, params=params, timeout=12)
        r.raise_for_status()
        items = r.json() or []
        out = []
        for it in items:
            ts = None
            if isinstance(it.get("datetime"), (int, float)):
                try:
                    ts = datetime.fromtimestamp(int(it.get("datetime")), tz=timezone.utc)
                except Exception:
                    ts = None
            elif it.get("datetime"):
                try:
                    ts = datetime.fromisoformat(it.get("datetime"))
                except Exception:
                    ts = None

            title = it.get("headline") or it.get("title") or it.get("summary") or "News"
            link = it.get("url") or it.get("news_url") or ""
            summary = it.get("summary") or it.get("source") or ""
            entry = {
                "title": title,
                "link": link,
                "summary": summary,
                "published": ts.isoformat() if ts else None
            }
            out.append(entry)
            if len(out) >= max_items:
                break
        return out
    except Exception as e:
        print("Finnhub news fetch error:", e)
        return []

def google_news_rss_query(query: str, days_back: int=7, max_items: int=5):
    q = requests.utils.quote(query)
    url = f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    entries = []
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    for e in feed.entries:
        published = None
        if hasattr(e, 'published_parsed') and e.published_parsed:
            try:
                published = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)
            except Exception:
                published = None
        entry = {
            "title": e.get("title"),
            "link": e.get("link"),
            "summary": e.get("summary"),
            "published": published.isoformat() if published else None
        }
        if published and published < cutoff:
            continue
        entries.append(entry)
        if len(entries) >= max_items:
            break
    return entries

def _articles_have_useful_content(articles):
    """
    Return True if articles is a non-empty list and contains at least one
    article with a non-empty title or link.
    """
    if not articles or not isinstance(articles, list):
        return False
    for a in articles:
        if not isinstance(a, dict):
            continue
        title = (a.get("title") or "").strip()
        link = (a.get("link") or "").strip()
        if title or link:
            return True
    return False

def fetch_news_for_company(company_entry: str, finnhub_api_key: str, days_back: int=7, max_items: int=5):
    """
    Try Finnhub first. If Finnhub returns nothing useful (empty list, or list of empty items),
    fall back to Google News RSS.
    Returns a tuple: (articles_list, source_string) where source_string is "finnhub" or "rss".
    """
    # Try Finnhub first if API key is present
    if finnhub_api_key:
        symbol = resolve_symbol_via_finnhub(company_entry, finnhub_api_key)
        if symbol:
            try:
                articles = finnhub_company_news(symbol, days_back=days_back, max_items=max_items, finnhub_api_key=finnhub_api_key)
            except Exception as e:
                print(f"[fetch_news] Finnhub error for {symbol}: {e} → falling back to RSS")
                articles = None

            # If articles is truthy and contains at least one useful item, use it.
            if _articles_have_useful_content(articles):
                print(f"[fetch_news] Using Finnhub for {company_entry} (symbol={symbol}) — {len(articles)} items")
                # return shallow copy to avoid accidental mutation
                return list(articles)[:max_items], "finnhub"
            else:
                print(f"[fetch_news] Finnhub returned no useful articles for {symbol} — falling back to RSS")
        else:
            print(f"[fetch_news] Could not resolve symbol for '{company_entry}' via Finnhub; falling back to RSS.")
    else:
        print("[fetch_news] No FINNHUB_API_KEY provided — using Google News RSS fallback.")

    # Fallback: google RSS search
    try:
        rss_articles = google_news_rss_query(company_entry, days_back=days_back, max_items=max_items)
        if _articles_have_useful_content(rss_articles):
            print(f"[fetch_news] Using Google News RSS for '{company_entry}' — {len(rss_articles)} items")
            return list(rss_articles)[:max_items], "rss"
        else:
            print(f"[fetch_news] Google News RSS returned no useful articles for '{company_entry}'.")
            return [], "none"
    except Exception as e:
        print(f"[fetch_news] Fallback RSS fetch error for '{company_entry}': {e}")
        return [], "none"

# ---- Deduplicate by link/title ----
def dedupe_articles(list_of_entries: List[Dict]):
    seen = set()
    out = []
    for e in list_of_entries:
        key = (e.get('link') or '') + '::' + (e.get('title') or '')
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out

# ---- OpenRouter analysis  ----
def get_openrouter_config():
    key = get_env("OPENROUTER_API_KEY")
    model = get_env("OPENROUTER_MODEL", OPENROUTER_MODEL_DEFAULT)
    return key, model

def _extract_text_from_openrouter_choice(choice: dict) -> str:
    msg = choice.get("message") or {}
    content = msg.get("content")
    if isinstance(content, dict):
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    return content if content is not None else ""

def _local_simple_analysis(title: str, snippet: str) -> Dict[str, str]:
    """
    Deterministic simple fallback: construct a minimal summary/sentiment/watch
    so that the report has content even if the LLM is unavailable.
    """
    text = (title or "") + (" — " + snippet if snippet else "")
    summary = text[:280] + ("…" if len(text) > 280 else "")
    # very naive sentiment: look for positive/negative words
    pos_words = ["beat", "beats", "growth", "surge", "record", "upgrade", "positive", "gain"]
    neg_words = ["miss", "missed", "down", "fall", "cut", "delay", "recall", "negative", "loss"]
    lower = text.lower()
    sentiment = "Neutral — no clear signal"
    for w in pos_words:
        if w in lower:
            sentiment = "Positive — contains positive signal"
            break
    for w in neg_words:
        if w in lower:
            sentiment = "Negative — contains negative signal"
            break
    watch = "Monitor for follow-up coverage and official company statements."
    return {"summary": summary, "sentiment": sentiment, "watch": watch}

def analyze_articles_with_openrouter(articles: List[Dict]) -> List[Dict]:
    """
    Returns a list of {"article": <article>, "analysis": <parsed dict>} entries.
    - Attempts to call OpenRouter for each article (if API key present).
    - Robust parsing: extracts JSON blob from response and verifies keys.
    - Retries once with a clarification prompt if parsing fails.
    - If OpenRouter is unavailable or returns non-parseable text, falls back to local simple analysis.
    - Writes debug output to DEBUG_RESPONSES_PATH for later inspection.
    """
    api_key, model = get_openrouter_config()
    headers = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results = []
    debug_out = []

    json_schema_example = (
        '{\n'
        '  "summary": "Two- to three-sentence factual summary of the article.",\n'
        '  "sentiment": "Positive — short reason OR Neutral — short reason OR Negative — short reason",\n'
        '  "watch": "One-line suggested next step or watchpoint for an investor."\n'
        '}'
    )

    for art in articles:
        title = art.get("title") or ""
        snippet = art.get("summary") or ""
        parsed = None
        raw_text = ""
        attempt = 0
        max_attempts = 2

        # If no API key, skip calling OpenRouter and use local fallback
        if not api_key:
            parsed = _local_simple_analysis(title, snippet)
            debug_out.append({"title": title, "attempt": 0, "source": "local_fallback", "parsed": parsed})
            results.append({"article": art, "analysis": parsed})
            continue

        while attempt < max_attempts and parsed is None:
            attempt += 1
            user_prompt = f"""
You are a concise investor-facing analyst.

Given the article title and snippet, return ONLY valid JSON that matches this schema (no extra commentary):

{json_schema_example}

Article title: {title}
Article snippet: {snippet}

Rules:
1) Return only the JSON object and nothing else.
2) If you cannot determine a field, return an empty string for that field.
3) Be brief and factual.
"""
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are an expert investor-facing summarization assistant."},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.0,
                "max_tokens": 600
            }

            try:
                resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(body), timeout=60)
                resp.raise_for_status()
                resp_json = resp.json()
            except Exception as e:
                raw_text = f"openrouter_http_error: {e}"
                debug_out.append({"title": title, "attempt": attempt, "error": raw_text})
                # on HTTP error, break out and fall back to local
                break

            # Extract content from choices
            choices = resp_json.get("choices") or []
            choice = choices[0] if choices else {}
            raw_text = _extract_text_from_openrouter_choice(choice) or ""
            # try to extract JSON object from raw_text
            parsed_candidate = None
            parse_error = None
            try:
                m = re.search(r"\{[\s\S]*\}", raw_text)
                json_text = m.group(0) if m else raw_text
                parsed_candidate = json.loads(json_text)
            except Exception as e_parse:
                parse_error = str(e_parse)
                parsed_candidate = None

            # Validate parsed_candidate has required keys
            if isinstance(parsed_candidate, dict):
                has_keys = all(k in parsed_candidate for k in ("summary", "sentiment", "watch"))
                if has_keys:
                    parsed = parsed_candidate
                    debug_out.append({"title": title, "attempt": attempt, "raw_text": raw_text, "parsed": parsed_candidate, "parse_error": parse_error})
                    break

            # Not parsed successfully
            debug_out.append({"title": title, "attempt": attempt, "raw_text": raw_text, "parsed": parsed_candidate, "parse_error": parse_error})

            # If we still have attempt left, send a short clarification asking for JSON only
            if attempt < max_attempts:
                clarification_prompt = f"""Previous assistant response (shown above) could not be parsed as JSON. Please return ONLY a JSON object that matches the schema below, with keys "summary","sentiment","watch". If unknown, use empty strings.

{json_schema_example}

Quote your JSON only."""
                body["messages"].append({"role": "user", "content": clarification_prompt})
                # loop to retry

        # End attempts
        if parsed is None:
            # fallback to local simple analysis
            parsed = _local_simple_analysis(title, snippet)
            debug_out.append({"title": title, "attempt": attempt, "source": "fallback_local", "parsed": parsed, "last_raw": raw_text})

        results.append({"article": art, "analysis": parsed})

    # persist debug file
    try:
        DEBUG_RESPONSES_PATH.write_text(json.dumps(debug_out, indent=2, ensure_ascii=False))
    except Exception as e:
        print("Could not write debug responses:", e)

    return results

# ---- Compose markdown report (includes original analysis fields) ----
def compose_markdown_report(company_results: Dict[str, List[Dict]]):
    """
    Compose the markdown report, enhanced with optional memory.json sections:
    - If memory.json contains per-company notes such as thesis, risk_factors,
      or position_notes, they are injected before the articles section.
    - Existing formatting and functionality remain unchanged.
    """
    # Load memory.json if present
    memory_data = {}
    try:
        if MEMORY_PATH.exists():
            memory_data = json.loads(MEMORY_PATH.read_text())
    except Exception as e:
        print(f"[compose_markdown_report] Warning: Could not load memory.json: {e}")

    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    md = [f"# Daily Holdings Report — {now}\n"]

    for company, analyses in company_results.items():
        md.append(f"## {company}\n")

        # Inject portfolio memory if available
        company_memory = memory_data.get(company, {})
        if company_memory:
            md.append("### 📌 Portfolio Memory Notes\n")
            thesis = company_memory.get("thesis")
            risks = company_memory.get("risk_factors")
            pos_notes = company_memory.get("position_notes")

            if thesis:
                md.append(f"- **Thesis:** {thesis}\n")

            if risks:
                # risks may be list or string
                if isinstance(risks, list):
                    risks_fmt = "; ".join(str(x) for x in risks)
                else:
                    risks_fmt = str(risks)
                md.append(f"- **Risk Factors:** {risks_fmt}\n")

            if pos_notes:
                md.append(f"- **Position Notes:** {pos_notes}\n")

            md.append("\n")  # spacer before articles

        # === Original article rendering logic (unchanged) ===
        if not analyses:
            md.append("_No recent articles found._\n")
            continue

        for idx, item in enumerate(analyses, start=1):
            a = item.get('article') or {}
            an = item.get('analysis') or {}
            title = a.get('title') or "No title"
            link = a.get('link') or ""
            summary = an.get('summary') if isinstance(an, dict) else an
            sentiment = an.get('sentiment') if isinstance(an, dict) else ""
            watch = an.get('watch') if isinstance(an, dict) else ""

            # fallbacks if parsed fields missing
            if summary is None:
                summary = an.get('raw') if isinstance(an, dict) else "None"
            if sentiment is None:
                sentiment = ""
            if watch is None:
                watch = ""

            md.append(f"### {idx}. [{title}]({link})\n")
            md.append(f"- **Summary:** {summary}\n")
            md.append(f"- **Sentiment:** {sentiment}\n")
            md.append(f"- **Watch:** {watch}\n")

        md.append("\n")

    return "\n".join(md)

# ---- HTML rendering ----
def render_html_from_markdown(report_md: str, title: str):
    body_html = markdown(report_md, extensions=["extra", "sane_lists", "nl2br"])
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(enabled_extensions=('html',))
    )
    template = env.get_template(TEMPLATE_FILE)
    rendered = template.render(title=title, generated_at=datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z"), content=body_html)
    return rendered

# ---- Send report via Resend (HTML) ----
def send_report_via_resend(resend_api_key, subject, html_body, from_email, to_emails):
    resend.api_key = resend_api_key
    params = {
        "from": from_email,
        "to": to_emails,
        "subject": subject,
        "html": html_body,
    }
    try:
        sent = resend.Emails.send(params)
        return sent
    except Exception as e:
        return {"error": str(e)}

# --- timezone guard helpers ---
from datetime import datetime as _dt
try:
    from zoneinfo import ZoneInfo
except Exception:
    from pytz import timezone as ZoneInfo  # requires pytz if used

def should_run_now_pacific_hour(target_hour=14, tz_name="America/Vancouver"):
    try:
        now_pacific = _dt.now(ZoneInfo(tz_name))
    except Exception:
        now_pacific = _dt.now()
    return now_pacific.hour == target_hour

# ---- Main flow ----
def main():
    # timezone guard (allow override)
    if FORCE_RUN_FLAG or FORCE_RUN_ENV:
        print("Timezone guard overridden (FORCE_RUN). Proceeding with run.")
    else:
        if not should_run_now_pacific_hour():
            print("Not 14:00 America/Vancouver right now — exiting. (This action runs twice UTC to handle DST.)")
            return

    # env / config
    openrouter_key = get_env("OPENROUTER_API_KEY")
    openrouter_model = get_env("OPENROUTER_MODEL", OPENROUTER_MODEL_DEFAULT)
    resend_key = get_env("RESEND_API_KEY")
    email_from = get_env("EMAIL_FROM")
    email_to_raw = get_env("EMAIL_TO")
    max_articles = int(get_env("MAX_ARTICLES_PER_COMPANY", "5"))
    days_back = int(get_env("DAYS_LOOKBACK", "7"))

    email_to = parse_email_to_list(email_to_raw)

    # NOTE: we will allow missing OPENROUTER_API_KEY and use the local fallback,
    # but prefer you set it for higher-quality summaries.
    if not openrouter_key:
        print("Warning: OPENROUTER_API_KEY not set — using local fallback analysis (less accurate).")

    companies = load_companies()
    company_results = {}
    for comp in tqdm(companies, desc="Companies"):
        finnhub_key = get_env("FINNHUB_API_KEY")
        # fetch returns (articles, source)
        try:
            entries, source = fetch_news_for_company(comp, finnhub_key, days_back=days_back, max_items=max_articles*2)
        except Exception as e:
            print(f"[main] fetch_news_for_company raised for {comp}: {e}")
            entries, source = [], "error"

        # Defensive: ensure entries is a list
        if not isinstance(entries, list):
            print(f"[main] Warning: entries for {comp} is not a list (type={type(entries)}). Forcing empty list.")
            entries = []

        # log sample titles for debugging
        try:
            sample_titles = [ (e.get("title") or e.get("summary") or "")[:120] for e in entries[:3] if isinstance(e, dict) ]
            print(f"[main] {comp} — source={source} — fetched {len(entries)} articles; samples: {sample_titles}")
        except Exception as e:
            print(f"[main] {comp} — source={source} — fetched {len(entries)} articles (could not list samples): {e}")

        entries = dedupe_articles(entries)[:max_articles]
        if not entries:
            company_results[comp] = []
            continue

        # analyze returns list of {"article":..., "analysis":...}
        analyses = analyze_articles_with_openrouter(entries)
        company_results[comp] = analyses

    # --- Memory update & trend computation ---
    trends = update_memory_with_results(company_results)
    # attach trend info into company_results so report builder can include it
    for comp, t in trends.items():
        # augment company_results: we'll add a special key `_trend` at company level
        # company_results[comp] is a list of article analyses
        if comp in company_results:
            # store trend dict in a wrapper dict so existing code still iterates articles
            company_results[comp + "##_trend"] = t

    # save local copy (optional)
    try:
        num_companies = len(company_results)
        print(f"[main] Processed {num_companies} companies.")
        sample = list(company_results.keys())[:6]
        print(f"[main] Sample companies: {sample}")
    except Exception:
        print("[main] Could not inspect company_results for debugging.")

    # Try to compose the markdown report; if that fails produce a safe fallback
    try:
        report_md = compose_markdown_report(company_results)
    except Exception as e:
        print("ERROR composing markdown report:", str(e))
        try:
            Path("debug_company_results.json").write_text(json.dumps(company_results, indent=2, ensure_ascii=False))
            print("Wrote debug_company_results.json for inspection.")
        except Exception as e2:
            print("Also failed to write debug_company_results.json:", e2)
        report_md = "# Report generation failed\n\nThere was an error composing the report. See debug_company_results.json for details."

    # Optionally save markdown (original behaviour; may fail on permission issues)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"daily_report_{ts}.md"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Saved report to {fname}")
    except Exception as e:
        print("Skipping saving markdown report (could not write file):", str(e))

    # Build HTML from (possibly fallback) markdown
    try:
        email_html = render_html_from_markdown(report_md, title=f"Daily Holdings Report — {datetime.now().strftime('%Y-%m-%d')}")
    except Exception as e:
        print("ERROR rendering HTML from markdown:", str(e))
        email_html = f"<html><body><h1>Daily Holdings Report</h1><pre>{str(report_md)[:1000]}</pre><p>Rendering error: {e}</p></body></html>"

    # send via Resend if configured
    if resend_key and email_from and email_to:
        subj = f"Daily Holdings Report — {datetime.now().strftime('%Y-%m-%d')}"
        try:
            resp = send_report_via_resend(resend_key, subj, email_html, email_from, email_to)
            print("Resend response:", resp)
        except Exception as e:
            print("Error sending via Resend:", str(e))
    else:
        print("Resend not configured or EMAIL_FROM/EMAIL_TO missing; skipping send. You can preview HTML in local file 'preview_report.html'.")
        try:
            with open("preview_report.html", "w", encoding="utf-8") as f:
                f.write(email_html)
            print("Saved preview_report.html for inspection.")
        except Exception as e:
            print("Could not save preview_report.html:", e)

if __name__ == "__main__":
    main()
