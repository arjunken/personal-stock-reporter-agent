#!/usr/bin/env python3
"""
agent.py (HTML email version)
- Converts generated markdown report to styled HTML using Jinja2 template
- Sends via Resend (html field)
"""

import os
import json
import time
import feedparser
import requests
from datetime import datetime, timedelta, timezone
from typing import List, Dict
from tqdm import tqdm
from dotenv import load_dotenv
from markdown import markdown
from jinja2 import Environment, FileSystemLoader, select_autoescape
from datetime import UTC

# Groq SDK
from groq import Groq

# Resend SDK
import resend
# ---- small CLI / env override for timezone guard ----
import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--force", action="store_true", help="Force run regardless of timezone guard (local testing)")
_known_args, _unknown = parser.parse_known_args()
FORCE_RUN_FLAG = bool(_known_args.force)
FORCE_RUN_ENV = os.getenv("FORCE_RUN", "") in ("1", "true", "True")


load_dotenv()  # loads .env into environment for local testing

# ---- Config / env helpers ----
COMPANIES_PATH = "companies.json"
TEMPLATES_DIR = "templates"
TEMPLATE_FILE = "report_template.html"

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

# --- New / replacement code for news fetching using Finnhub (plus fallback) ---
import re
from datetime import datetime, timedelta, timezone

FINNHUB_SEARCH_URL = "https://finnhub.io/api/v1/search"
FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"

def looks_like_ticker(s: str) -> bool:
    # simple heuristic: uppercase letters or digits, 1-5 length (adjust if you have longer tickers)
    return bool(re.fullmatch(r"[A-Z0-9\.\-]{1,6}", s.strip()))

def resolve_symbol_via_finnhub(name_or_symbol: str, finnhub_api_key: str):
    """
    If input already looks like a ticker, return it.
    Otherwise call Finnhub /search to find the best match.
    Returns: symbol string or None if not found.
    """
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
        # Finnhub returns 'result' list with items containing 'symbol' and 'description'
        results = data.get("result") or data.get("results") or []
        if not results:
            return None
        # pick the top match
        top = results[0]
        symbol = top.get("symbol")
        return symbol
    except Exception as e:
        print("Finnhub search error:", e)
        return None

def finnhub_company_news(symbol: str, days_back: int=7, max_items: int=5, finnhub_api_key: str=None):
    """
    Calls Finnhub company-news endpoint and returns a list of entries:
    { "title", "link", "summary", "published" }
    """
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
            # Finnhub returns timestamps as unix epoch (seconds) in field 'datetime' often.
            # We'll try multiple possible keys defensively.
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

def fetch_news_for_company(company_entry: str, finnhub_api_key: str, days_back: int=7, max_items: int=5):
    """
    High level: resolve symbol if needed, then call Finnhub. If anything fails, fallback to Google News RSS.
    """
    # Try Finnhub first if API key is present
    if finnhub_api_key:
        symbol = resolve_symbol_via_finnhub(company_entry, finnhub_api_key)
        if symbol:
            articles = finnhub_company_news(symbol, days_back=days_back, max_items=max_items, finnhub_api_key=finnhub_api_key)
            if articles:
                return articles
            else:
                print(f"Finnhub returned no articles for symbol {symbol}, falling back to RSS for '{company_entry}'.")
        else:
            print(f"Could not resolve symbol for '{company_entry}' via Finnhub; falling back to RSS.")
    else:
        print("No FINNHUB_API_KEY provided — using Google News RSS fallback.")

    # Fallback: google RSS search (existing function, keep for older behaviour)
    try:
        return google_news_rss_query(company_entry, days_back=days_back, max_items=max_items)
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

# ---- Prepare prompt for Groq (same as before) ----
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

Return ONLY valid JSON.
"""
    return prompt

# ---- Call Groq model ----
def analyze_articles_with_groq(groq_client: Groq, model: str, articles: List[Dict]):
    results = []
    for art in articles:
        prompt = prepare_prompt_for_article(art)
        try:
            resp = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert investor-facing summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                model = model,
                max_tokens = 400,
                temperature = 0.0,
            )
            text = ""
            try:
                text = resp.choices[0].message.content
            except Exception:
                text = str(resp)

            parsed = None
            try:
                import re
                m = re.search(r"\{.*\}", text, re.DOTALL)
                json_text = m.group(0) if m else text
                parsed = json.loads(json_text)
            except Exception:
                parsed = {"raw": text}

        except Exception as e:
            parsed = {"error": str(e)}
        results.append({"article": art, "analysis": parsed})
        time.sleep(0.2)
    return results

# ---- Compose markdown report (same as before) ----
def compose_markdown_report(company_results: Dict[str, List[Dict]]):
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
    md = [f"# Weekly Holdings Report — {now}\n"]
    for company, analyses in company_results.items():
        md.append(f"## {company}\n")
        if not analyses:
            md.append("_No recent articles found._\n")
            continue
        for idx, item in enumerate(analyses, start=1):
            a = item['article']
            an = item['analysis']
            title = a.get('title')
            link = a.get('link')
            summary = an.get('summary') if isinstance(an, dict) else an
            sentiment = an.get('sentiment') if isinstance(an, dict) else ""
            watch = an.get('watch') if isinstance(an, dict) else ""
            md.append(f"### {idx}. [{title}]({link})\n")
            md.append(f"- **Summary:** {summary}\n")
            md.append(f"- **Sentiment:** {sentiment}\n")
            md.append(f"- **Watch:** {watch}\n")
        md.append("\n")
    return "\n".join(md)

# ---- HTML rendering ----
def render_html_from_markdown(report_md: str, title: str):
    # Convert markdown -> HTML (basic)
    body_html = markdown(report_md, extensions=["extra", "sane_lists", "nl2br"])
    # Load Jinja2 template
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

# --- add/import at top of your agent.py ---
from datetime import datetime
try:
    # zoneinfo exists in Python 3.9+
    from zoneinfo import ZoneInfo
except Exception:
    # fallback for very old Python versions (not recommended)
    from pytz import timezone as ZoneInfo  # requires pytz in requirements

# --- add this near the start of main() ---
def should_run_now_pacific_hour(target_hour=14, tz_name="America/Vancouver"):
    """
    Return True only when current time in tz_name has hour == target_hour.
    We expect the workflow to be scheduled twice (two UTC crons). Only one
    of those runs will find the local hour == target_hour; the other will exit.
    """
    try:
        now_pacific = datetime.now(ZoneInfo(tz_name))
    except Exception:
        # if ZoneInfo fallback to pytz style
        now_pacific = datetime.now()  # best-effort, but zoneinfo is preferred
    # compare hour (24h) and allow minute==0 (exact run)
    # If you want a small tolerance window (e.g. minutes 0-5) change condition accordingly.
    return now_pacific.hour == target_hour


# ---- Main flow ----
def main():

    # Quick timezone guard: run only if it's 14:00 America/Vancouver
    # allow override for local testing
    if FORCE_RUN_FLAG or FORCE_RUN_ENV:
        print("Timezone guard overridden (FORCE_RUN). Proceeding with run.")
    else:
        # Quick timezone guard: run only if it's 14:00 America/Vancouver
        if not should_run_now_pacific_hour():
            print("Not 14:00 America/Vancouver right now — exiting. (This action runs twice UTC to handle DST.)")
            return

    groq_key = get_env("GROQ_API_KEY")
    resend_key = get_env("RESEND_API_KEY")
    email_from = get_env("EMAIL_FROM")
    email_to_raw = get_env("EMAIL_TO")
    max_articles = int(get_env("MAX_ARTICLES_PER_COMPANY", "5"))
    days_back = int(get_env("DAYS_LOOKBACK", "7"))
    model = get_env("GROQ_MODEL", "openai/gpt-oss-120b")

    email_to = parse_email_to_list(email_to_raw)

    if not groq_key:
        raise RuntimeError("GROQ_API_KEY environment variable is required.")

    companies = load_companies()

    # init Groq client
    groq_client = Groq(api_key=groq_key)

    company_results = {}
    for comp in tqdm(companies, desc="Companies"):
        finnhub_key = get_env("FINNHUB_API_KEY")
        entries = fetch_news_for_company(comp, finnhub_key, days_back=days_back, max_items=max_articles*2)

        entries = dedupe_articles(entries)[:max_articles]
        if not entries:
            company_results[comp] = []
            continue
        analyses = analyze_articles_with_groq(groq_client, model, entries)
        company_results[comp] = analyses

    report_md = compose_markdown_report(company_results)

    # save local copy
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    fname = f"weekly_report_{ts}.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved report to {fname}")

    # Build HTML
    email_html = render_html_from_markdown(report_md, title=f"Weekly Holdings Report — {datetime.now().strftime('%Y-%m-%d')}")

    # send via Resend if configured
    if resend_key and email_from and email_to:
        subj = f"Weekly Holdings Report — {datetime.now().strftime('%Y-%m-%d')}"
        resp = send_report_via_resend(resend_key, subj, email_html, email_from, email_to)
        print("Resend response:", resp)
    else:
        print("Resend not configured or EMAIL_FROM/EMAIL_TO missing; skipping send. You can preview HTML in local file 'preview_report.html'.")
        # Save preview
        with open("preview_report.html", "w", encoding="utf-8") as f:
            f.write(email_html)
        print("Saved preview_report.html for inspection.")

if __name__ == "__main__":
    main()
