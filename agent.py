#!/usr/bin/env python3
"""
agent.py (HTML email version) — OpenRouter edition
- Fetches news (Finnhub or Google RSS fallback)
- Calls OpenRouter (openrouter/polaris-alpha) to analyze articles into JSON
- Persists an optional markdown file, renders styled HTML via Jinja2, and sends via Resend
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

def fetch_news_for_company(company_entry: str, finnhub_api_key: str, days_back: int=7, max_items: int=5):
    """
    High level: resolve symbol if needed, then call Finnhub.
    If Finnhub returns an error OR returns an empty list, fall back to Google News RSS.
    This prevents empty reports when Finnhub returns no recent items for the symbol.
    """
    # Try Finnhub first if API key is present
    if finnhub_api_key:
        symbol = resolve_symbol_via_finnhub(company_entry, finnhub_api_key)
        if symbol:
            try:
                articles = finnhub_company_news(symbol, days_back=days_back, max_items=max_items, finnhub_api_key=finnhub_api_key)
            except Exception as e:
                print(f"Finnhub fetch raised an exception for {symbol}: {e}. Falling back to RSS.")
                articles = None

            # Treat empty list as a signal to fallback to RSS
            if articles:
                print(f"Fetched {len(articles)} articles from Finnhub for {symbol}.")
                return articles
            else:
                print(f"Finnhub returned no articles for symbol {symbol} (or error). Falling back to Google News RSS for '{company_entry}'.")
        else:
            print(f"Could not resolve symbol for '{company_entry}' via Finnhub; falling back to RSS.")
    else:
        print("No FINNHUB_API_KEY provided — using Google News RSS fallback.")

    # Fallback: google RSS search
    try:
        rss_articles = google_news_rss_query(company_entry, days_back=days_back, max_items=max_items)
        print(f"Fetched {len(rss_articles)} articles from Google News RSS for '{company_entry}'.")
        return rss_articles
    except Exception as e:
        print("Fallback RSS fetch error:", e)
        return []


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

# ---- Prompt builder (ask for JSON) ----
def prepare_prompt_for_article(article):
    title = article.get('title') or ''
    snippet = article.get('summary') or ''
    prompt = f"""
You are a concise investor-facing analyst. Given the article title and snippet, produce JSON with these keys:
- summary: 2-3 sentence factual executive summary.
- sentiment: one of "Positive", "Neutral", "Negative", with one short reason.
- watch: one short suggested next step or watchpoint for an investor.

Article title: {title}
Article snippet: {snippet}

Return ONLY valid JSON similar to:
{{ "summary": "…", "sentiment": "Positive — reason", "watch": "..." }}
If you cannot determine a field, return an empty string for that field.
"""
    return prompt

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

def analyze_articles_with_openrouter(articles: List[Dict]) -> List[Dict]:
    api_key, model = get_openrouter_config()
    headers = {
        "Authorization": f"Bearer {api_key}" if api_key else "",
        "Content-Type": "application/json",
    }

    results = []
    debug_out = []

    for art in articles:
        prompt = prepare_prompt_for_article(art)
        messages = [
            {"role": "system", "content": "You are an expert investor-facing summarization assistant."},
            {"role": "user", "content": prompt}
        ]
        body = {
            "model": model,
            "messages": messages,
            "extra_body": {"reasoning": {"enabled": True}},
            # "max_tokens": 400,
            # "temperature": 0.0
        }

        try:
            resp = requests.post(OPENROUTER_API_URL, headers=headers, data=json.dumps(body), timeout=60)
            resp.raise_for_status()
            resp_json = resp.json()
        except Exception as e:
            parsed = {"error": f"openrouter_http_error: {str(e)}"}
            results.append({"article": art, "analysis": parsed})
            debug_out.append({"article_title": art.get("title"), "error": str(e)})
            time.sleep(0.2)
            continue

        choices = resp_json.get("choices") or []
        if not choices:
            parsed = {"error": "no_choices_returned", "raw": resp_json}
            results.append({"article": art, "analysis": parsed})
            debug_out.append({"article_title": art.get("title"), "raw_response": resp_json})
            time.sleep(0.2)
            continue

        choice = choices[0]
        text = _extract_text_from_openrouter_choice(choice)
        debug_out.append({"article_title": art.get("title"), "raw_text": text, "raw_choice": choice})

        # try to pull reasoning_details if present
        reasoning = None
        try:
            msg = choice.get("message", {})
            reasoning = msg.get("reasoning_details")
        except Exception:
            reasoning = None

        # parse JSON from text
        parsed = None
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            json_text = m.group(0) if m else text
            parsed = json.loads(json_text)
        except Exception as e_parse:
            parsed = {"raw": text, "parse_error": str(e_parse)}
            if reasoning:
                parsed["_reasoning_details"] = reasoning

        results.append({"article": art, "analysis": parsed})
        time.sleep(0.2)

    # write debug_out to artifact file for CI debugging (optional)
    try:
        DEBUG_RESPONSES_PATH.write_text(json.dumps(debug_out, indent=2, ensure_ascii=False))
        print(f"Wrote debug responses to {DEBUG_RESPONSES_PATH}")
    except Exception:
        pass

    return results

# ---- Compose markdown report (includes original analysis fields) ----
def compose_markdown_report(company_results: Dict[str, List[Dict]]):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    md = [f"# Daily Holdings Report — {now}\n"]
    for company, analyses in company_results.items():
        md.append(f"## {company}\n")
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

    if not openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is required.")

    companies = load_companies()
    company_results = {}

    for comp in tqdm(companies, desc="Companies"):
        finnhub_key = get_env("FINNHUB_API_KEY")
        entries = fetch_news_for_company(comp, finnhub_key, days_back=days_back, max_items=max_articles*2)
        entries = dedupe_articles(entries)[:max_articles]
        if not entries:
            company_results[comp] = []
            continue

        # use OpenRouter analysis
        analyses = analyze_articles_with_openrouter(entries)
        company_results[comp] = analyses

    report_md = compose_markdown_report(company_results)

    # save local copy (optional)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"daily_report_{ts}.md"
    try:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"Saved report to {fname}")
    except Exception:
        print("Skipping saving markdown report (could not write file).")

    # Build HTML
    email_html = render_html_from_markdown(report_md, title=f"Daily Holdings Report — {datetime.now().strftime('%Y-%m-%d')}")

    # send via Resend if configured
    if resend_key and email_from and email_to:
        subj = f"Daily Holdings Report — {datetime.now().strftime('%Y-%m-%d')}"
        resp = send_report_via_resend(resend_key, subj, email_html, email_from, email_to)
        print("Resend response:", resp)
    else:
        print("Resend not configured or EMAIL_FROM/EMAIL_TO missing; skipping send. You can preview HTML in local file 'preview_report.html'.")
        with open("preview_report.html", "w", encoding="utf-8") as f:
            f.write(email_html)
        print("Saved preview_report.html for inspection.")

if __name__ == "__main__":
    main()
