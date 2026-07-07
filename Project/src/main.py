"""
Asynchronous, long-running Telegram bot service for the IITB Study/Deadline
Assistant.

Users drop raw tasks into the chat as plain messages, which get logged
to a local JSON backlog with a sequential ID. Sending /generate compiles
all pending tasks (without clearing them) into a prioritized schedule
and pushes it out via the LangGraph pipeline. Tasks are only removed
from the pending pool explicitly, via /done <id>.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.config import TELEGRAM_BOT_TOKEN
from src.graph.workflow import app as agent_graph
from src.utils.storage import (
    initialize_db,
    add_raw_task,
    get_pending_tasks_string,
    mark_task_complete,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command with a welcome and usage explanation."""
    welcome_message = (
        "👋 *Welcome to the IITB Study/Deadline Assistant!*\n\n"
        "Here's how it works:\n"
        "1️⃣ Just send me your tasks as plain messages — coursework, "
        "SoS milestones, LeetCode grinding, anything. I'll log each one "
        "with a unique ID.\n"
        "2️⃣ Send /generate anytime to compile everything currently "
        "pending into a prioritized, formatted schedule. Tasks stay in "
        "your backlog after generating, so you can regenerate freely.\n"
        "3️⃣ Once a task is actually done, send /done <id> to clear it "
        "off your pending list.\n\n"
        "Example:\n"
        "  _CS302 lab due Thursday_\n"
        "  _SoS Docker milestone on Friday_\n"
        "  _do 3 DP medium questions today_\n\n"
        "Then /generate to compile, and /done 2 once you've finished task #2 🚀"
    )
    try:
        await update.message.reply_text(welcome_message, parse_mode="Markdown")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to send /start welcome message: {exc}")


async def handle_incoming_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log any incoming plain-text message as a pending raw task."""
    try:
        message_text = update.message.text if update.message else None

        if not message_text or not message_text.strip():
            await update.message.reply_text(
                "⚠️ That message looked empty — try sending some task text."
            )
            return

        add_raw_task(message_text)
        await update.message.reply_text(
            f"✅ Logged: \"{message_text.strip()}\"\n"
            "Send more tasks anytime, /generate to compile your schedule, "
            "or /done <id> once a task is finished."
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to log incoming task: {exc}")
        try:
            await update.message.reply_text(
                "❌ Something went wrong logging that task. Please try again."
            )
        except Exception:  # noqa: BLE001
            pass


async def generate_schedule(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Compile all pending tasks through the LangGraph pipeline and report the result."""
    try:
        raw_inputs_combined = get_pending_tasks_string()

        if not raw_inputs_combined or not raw_inputs_combined.strip():
            await update.message.reply_text(
                "📭 No pending tasks logged yet. Send me some tasks first, "
                "then run /generate."
            )
            return

        await update.message.reply_text(
            "⚙️ Compiling your schedule — planning, prioritizing, and "
            "pushing to your assistant channel..."
        )

        final_state = agent_graph.invoke({"raw_input": raw_inputs_combined})
        telegram_status = final_state.get("telegram_status", "FAILED")

        if telegram_status == "SUCCESS":
            task_count = len(final_state.get("prioritized_list", []))
            await update.message.reply_text(
                f"✅ Schedule generated and delivered! "
                f"({task_count} task(s) processed.)\n"
                "Your tasks remain in the pending backlog — use /done <id> "
                "to clear off anything you've completed."
            )
        else:
            await update.message.reply_text(
                "⚠️ The schedule was compiled, but delivery to Telegram "
                "reported a failure. Your pending tasks have been kept "
                "so you can retry /generate — check the bot logs and your "
                "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID configuration."
            )

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to generate schedule: {exc}")
        try:
            await update.message.reply_text(
                "❌ An unexpected error occurred while generating your "
                "schedule. Your pending tasks have not been affected — "
                "please try /generate again shortly."
            )
        except Exception:  # noqa: BLE001
            pass


async def complete_task(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /done <id> by marking the matching task as completed."""
    try:
        if not context.args:
            await update.message.reply_text(
                "⚠️ Please provide a task ID, e.g. `/done 3`",
                parse_mode="Markdown",
            )
            return

        raw_id = context.args[0]

        try:
            task_id = int(raw_id)
        except ValueError:
            await update.message.reply_text(
                f"⚠️ '{raw_id}' isn't a valid task ID. Please use a number, "
                "e.g. `/done 3`",
                parse_mode="Markdown",
            )
            return

        success = mark_task_complete(task_id)

        if success:
            await update.message.reply_text(f"✅ Task #{task_id} marked as completed.")
        else:
            await update.message.reply_text(
                f"❌ Couldn't complete task #{task_id}. It may not exist or "
                "may already be marked completed."
            )

    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to complete task: {exc}")
        try:
            await update.message.reply_text(
                "❌ An unexpected error occurred while marking that task complete."
            )
        except Exception:  # noqa: BLE001
            pass


def main() -> None:
    """Build the Telegram Application, register handlers, and start polling."""
    if not TELEGRAM_BOT_TOKEN:
        raise EnvironmentError(
            "TELEGRAM_BOT_TOKEN is not set. Add it to your .env file before "
            "starting the bot service."
        )

    initialize_db()

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("generate", generate_schedule))
    application.add_handler(CommandHandler("done", complete_task))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_incoming_task)
    )

    logger.info("IITB Study/Deadline Assistant bot is starting polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()