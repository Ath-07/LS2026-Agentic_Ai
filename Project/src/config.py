"""
Central configuration module.

Loads environment variables via python-dotenv and exposes a lazily
initialized Gemini LLM client (via langchain_google_genai) for any
agent that needs LLM-assisted reasoning.
"""

import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _validate_env() -> None:
    """Warn (without crashing) if required environment variables are missing."""
    missing = [
        name
        for name, val in [
            ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
            ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
            ("GEMINI_API_KEY", GEMINI_API_KEY),
        ]
        if not val
    ]
    if missing:
        print(
            f"[CONFIG WARNING] Missing environment variables: {', '.join(missing)}. "
            "Set these in a .env file at the project root for full functionality "
            "(Telegram delivery and/or LLM calls will be degraded or skipped)."
        )


_validate_env()


def get_llm(temperature: float = 0.2):
    """
    Construct and return a Gemini chat model client.

    Raises:
        EnvironmentError: if GEMINI_API_KEY is not configured.
        ImportError: if langchain_google_genai is not installed.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. Add it to your .env file before requesting the LLM."
        )
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain_google_genai is not installed. Run "
            "`pip install langchain-google-genai` to enable LLM features."
        ) from exc

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=temperature,
    )


# Best-effort eager initialization. If this fails (missing key or package),
# `llm` stays None and downstream agents fall back to rule-based logic.
llm = None
try:
    if GEMINI_API_KEY:
        llm = get_llm()
except Exception as exc:  # noqa: BLE001 - deliberately broad for a config-time guard
    print(f"[CONFIG WARNING] Failed to initialize LLM client: {exc}")