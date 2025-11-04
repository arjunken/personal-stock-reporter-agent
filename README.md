# AI Stock Reporter Agent

Automated AI agent that generates **daily investment news reports** for your portfolio companies, analyzes sentiment using an LLM, and sends styled HTML reports via email.

---

## 🚀 Project Overview

This agent combines:

1. **LLM-based analysis** (Groq `openai/gpt-oss-120b`) for summarizing financial news and producing investor-friendly insights.  
2. **Real-time news fetching** using **Finnhub.io** (preferred) or Google News RSS as a fallback.  
3. **Automated email delivery** via **Resend API**, with styled HTML reports.  
4. **Timezone-aware scheduling** to run daily at **2:00 PM Pacific Time**, including DST handling.  

Your weekly or daily investment research report is fully automated, giving concise summaries, sentiment, and watchpoints for each company in your portfolio.

---

## 📦 Features

- Fetches the latest news for all companies listed in `companies.json`.  
- Uses an LLM to summarize news articles into JSON with:
  - `summary` – concise 2–3 sentence summary  
  - `sentiment` – Positive / Neutral / Negative  
  - `watch` – suggested next step or watchpoint  
- Generates HTML email reports with clean, responsive styling.  
- Optional Markdown file output (can be disabled).  
- Timezone-aware scheduling for daily automated reports at 2 PM PST/PDT.  
- Fully GitHub Actions compatible for cloud automation.  

---

## 🏗 Architecture / Workflow

```
┌───────────────┐
│ companies.json│
└──────┬────────┘
       │
       ▼
┌───────────────┐
│ News Fetching │ (Finnhub API / Google RSS)
└──────┬────────┘
       ▼
┌───────────────┐
│ LLM Analysis  │ (Groq GPT model)
│ summary + sentiment + watch
└──────┬────────┘
       ▼
┌───────────────┐
│ HTML Renderer │ (Jinja2 template)
└──────┬────────┘
       ▼
┌───────────────┐
│ Email Delivery│ (Resend API)
└───────────────┘
```

- Scheduled daily at 2 PM Pacific via **GitHub Actions cron**.
- Includes DST-proof logic to avoid duplicate sends.

---

## ⚙️ Requirements

- Python 3.11+  
- Pip packages (see `requirements.txt`):
  ```
  feedparser
  requests
  python-dotenv
  groq
  resend
  tqdm
  markdown
  jinja2
  ```

- API keys (set via `.env` or GitHub Secrets):
  - `GROQ_API_KEY` – your Groq API key  
  - `RESEND_API_KEY` – your Resend email API key  
  - `FINNHUB_API_KEY` – your Finnhub API key for latest news  
  - `EMAIL_FROM` – sender email  
  - `EMAIL_TO` – recipient(s) JSON array: `["you@example.com"]`  
  - Optional: `MAX_ARTICLES_PER_COMPANY`, `DAYS_LOOKBACK`, `GROQ_MODEL`

---

## 📝 Setup Instructions

1. **Clone repository**:
```bash
git clone <repo_url>
cd <repo_name>
```

2. **Create a virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create `.env` file** in the project root:
```ini
GROQ_API_KEY=your_groq_api_key
RESEND_API_KEY=your_resend_api_key
FINNHUB_API_KEY=your_finnhub_api_key
EMAIL_FROM="Reports <reports@yourdomain.com>"
EMAIL_TO='["you@example.com"]'
MAX_ARTICLES_PER_COMPANY=5
DAYS_LOOKBACK=7
GROQ_MODEL=openai/gpt-oss-120b
SAVE_REPORT=false  # disable markdown saving if desired
```

---

## 🏃 Running Locally

- **Dry-run / preview without sending email**:
```bash
python agent.py --no-send
# Or override SAVE_REPORT and force run:
SAVE_REPORT=false python agent.py --force
```

- **Send real email**:
```bash
python agent.py --force
```

> `--force` bypasses timezone guard for local testing. GitHub Actions runs do not require `--force`.

---

## 📅 GitHub Actions Deployment

1. Copy `.env` keys into **GitHub Secrets**:
   - `GROQ_API_KEY`
   - `RESEND_API_KEY`
   - `FINNHUB_API_KEY`
   - `EMAIL_FROM`
   - `EMAIL_TO`
   - `MAX_ARTICLES_PER_COMPANY`
   - `DAYS_LOOKBACK`
   - `GROQ_MODEL`

2. Cron schedule (`.github/workflows/schedule.yml`):
```yaml
on:
  schedule:
    - cron: '0 21 * * *' # 2 PM PDT
    - cron: '0 22 * * *' # 2 PM PST
```

- The agent checks local Pacific time before sending to ensure only one run triggers per day.

---

## 💻 Files Overview

- `agent.py` – main agent script  
- `companies.json` – portfolio companies to track  
- `templates/report_template.html` – HTML email template  
- `requirements.txt` – Python dependencies  
- `.github/workflows/schedule.yml` – GitHub Actions workflow  
- `.env` – local environment variables (not committed to GitHub)  

---

## ⚡ Notes & Tips

- Markdown files can be disabled via `SAVE_REPORT=false` in `.env` or `--no-save`.  
- HTML emails are **responsive and email-client friendly**, using inline-safe CSS.  
- Finnhub free tier has API rate limits; for larger portfolios, consider upgrading.  
- You can extend the agent to:
  - Supabase or Notion storage  
  - Slack/Telegram notifications  
  - Breaking news alerts  

---

## 🧠 Future Enhancements

- Track **only new articles** week-over-week  
- Add **multi-portfolio support**  
- Integrate a **dashboard** for easy portfolio overview  
- Add **AI-driven trend detection**  

---

## 📄 License

MIT License – free to use and modify.

