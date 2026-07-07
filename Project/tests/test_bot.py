"""
Standalone connectivity test for the Telegram Bot integration.

Run directly from the project root:

    python tests/test_bot.py

This does NOT go through the LangGraph pipeline — it is a minimal,
isolated sanity check that the configured bot token and chat id can
actually deliver a message via the Telegram Bot API.
"""

import os
import sys

import requests
from dotenv import load_dotenv

# Ensure the project root is on sys.path so this script works when run
# directly (e.g. `python tests/test_bot.py`) as well as via a test runner.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TEST_MESSAGE = "🤖 *Test Alert*: Connection to Study Assistant Bot successful!"
TELEGRAM_API_URL_TEMPLATE = "https://api.telegram.org/bot{token}/sendMessage"


def test_telegram_connection() -> bool:
    """
    Send a hardcoded test message to the configured Telegram chat.

    Returns:
        True if the message was delivered successfully, False otherwise.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "[TEST_BOT ERROR] TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing. "
            "Please set both in your .env file at the project root."
        )
        return False

    url = TELEGRAM_API_URL_TEMPLATE.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": TEST_MESSAGE,
        "parse_mode": "Markdown",
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
    except requests.exceptions.ConnectionError as exc:
        print(f"[TEST_BOT ERROR] Could not connect to Telegram API: {exc}")
        return False
    except requests.exceptions.Timeout as exc:
        print(f"[TEST_BOT ERROR] Request to Telegram API timed out: {exc}")
        return False
    except requests.exceptions.RequestException as exc:
        print(f"[TEST_BOT ERROR] Unexpected request failure: {exc}")
        return False

    try:
        response_payload = response.json()
    except ValueError:
        response_payload = {"raw_text": response.text}

    if response.status_code == 200 and response_payload.get("ok"):
        print("=" * 60)
        print("✅ SUCCESS: Telegram Bot connection verified.")
        print(f"   Message delivered to chat_id: {TELEGRAM_CHAT_ID}")
        print(f"   Telegram message_id: {response_payload.get('result', {}).get('message_id')}")
        print("=" * 60)
        return True

    print("=" * 60)
    print("❌ FAILED: Telegram Bot connection test failed.")
    print(f"   HTTP status code: {response.status_code}")
    print(f"   Response payload: {response_payload}")
    print("=" * 60)
    return False


if __name__ == "__main__":
    success = test_telegram_connection()
    sys.exit(0 if success else 1)