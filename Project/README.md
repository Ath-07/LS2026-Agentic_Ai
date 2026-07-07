# IITB Deadline Assistant — Telegram Bot

An AI-powered Telegram bot that helps students manage deadlines by accepting unstructured task messages and generating a prioritized schedule using a LangGraph pipeline.

Built as part of **Learner Space 2026 — Course: Agentic AI**.

---

## Features

- **Log tasks naturally** — Send messages like `"CS302 lab due Thursday, fix SoS git bug by Friday"` and the bot stores them.
- **Generate a schedule** — On `/generate`, runs a hierarchical agent pipeline (planner → priority → reminder) that parses, scores, and formats your pending tasks.
- **Mark tasks done** — `/done <id>` removes a task from the pending backlog.
- **Persistent backlog** — Tasks survive bot restarts via JSON file storage.
- **Graceful degradation** — Works with or without a configured LLM; falls back to rule-based logic.

---

## Architecture

```
raw text → planner → priority → reminder → Telegram / console
```

- **planner** — Splits raw text into individual tasks, classifies each into a track (IITB Coursework, SoS Project, LeetCode) and extracts deadlines.
- **priority** — Scores tasks by track weight and deadline urgency, then sorts descending.
- **reminder** — Formats the sorted list as a Markdown message and delivers it.

---

## Tech Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.14 |
| Bot Framework | python-telegram-bot |
| Agent Pipeline | LangGraph |
| LLM (optional) | Google Gemini via langchain-google-genai |
| Persistence | JSON file (thread-safe) |

---

## Setup

1. **Clone and install**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Fill in `GEMINI_API_KEY`, `TELEGRAM_BOT_TOKEN`, and `TELEGRAM_CHAT_ID`.

3. **Run the bot**
   ```bash
   python src/main.py
   ```

4. **Run tests**
   ```bash
   python tests/test_agents.py
   python tests/test_bot.py
   ```

---

## Project Structure

```
src/
├── main.py                # Entry point — Telegram bot and handler registration
├── config.py              # Environment & LLM configuration
├── agents/
│   ├── planner.py         # Task extraction and track classification
│   ├── priority.py        # Priority scoring and sorting
│   └── reminder.py        # Formatting and delivery
├── graph/
│   ├── state.py           # LangGraph state schema
│   └── workflow.py        # Pipeline wiring and compilation
└── utils/
    └── storage.py         # Thread-safe JSON persistence
tests/
├── test_agents.py         # Full pipeline integration test
└── test_bot.py            # Telegram connectivity test
```

---

## Usage

| Command / Input | Description |
|---|---|
| `/start` | Welcome message |
| `raw text message` | Logs a new task (no command needed) |
| `/generate` | Runs the pipeline and posts a prioritized schedule |
| `/done <id>` | Marks a task as complete |
