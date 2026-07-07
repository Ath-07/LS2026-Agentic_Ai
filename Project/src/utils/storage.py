"""
Thread-safe JSON-backed persistence layer for raw incoming task text.

Tasks are logged here as they arrive via Telegram messages, each with a
sequential integer ID. Tasks persist across /generate runs and are only
removed from the "pending" pool when the user explicitly marks them
complete via /done <id>.
"""

import json
import os
import threading
from typing import List, Dict, Any

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))), "tasks_db.json")

# Guards all read-modify-write cycles against the JSON file so concurrent
# Telegram updates (handled on the same event loop, but potentially
# overlapping I/O) don't corrupt the backlog.
_LOCK = threading.Lock()


def initialize_db() -> None:
    """Create the backing JSON file with an empty array if it doesn't exist."""
    with _LOCK:
        if not os.path.exists(_DB_PATH):
            try:
                with open(_DB_PATH, "w", encoding="utf-8") as f:
                    json.dump([], f)
                print(f"[STORAGE] Initialized new task database at {_DB_PATH}")
            except OSError as exc:
                print(f"[STORAGE ERROR] Failed to initialize database file: {exc}")
                raise


def _read_all_tasks() -> List[Dict[str, Any]]:
    """
    Read and return the full task list from disk.

    Returns an empty list (and self-heals the file) if it is missing,
    empty, or contains corrupted JSON.
    """
    if not os.path.exists(_DB_PATH):
        initialize_db()
        return []

    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                return []
            data = json.loads(content)
            if not isinstance(data, list):
                print("[STORAGE WARNING] tasks_db.json did not contain a list. Resetting.")
                return []
            return data
    except json.JSONDecodeError as exc:
        print(f"[STORAGE ERROR] Corrupted JSON in tasks_db.json, resetting file: {exc}")
        return []
    except OSError as exc:
        print(f"[STORAGE ERROR] Failed to read tasks_db.json: {exc}")
        return []


def _write_all_tasks(tasks: List[Dict[str, Any]]) -> None:
    """Overwrite the backing JSON file with the given task list."""
    try:
        with open(_DB_PATH, "w", encoding="utf-8") as f:
            json.dump(tasks, f, indent=2)
    except OSError as exc:
        print(f"[STORAGE ERROR] Failed to write tasks_db.json: {exc}")
        raise


def _next_id(tasks: List[Dict[str, Any]]) -> int:
    """Compute the next sequential task ID based on existing entries."""
    existing_ids = [
        task.get("id") for task in tasks
        if isinstance(task.get("id"), int)
    ]
    if not existing_ids:
        return 1
    return max(existing_ids) + 1


def add_raw_task(task_text: str) -> None:
    """
    Append a new pending task entry to the JSON backlog with a unique,
    sequentially-incrementing integer ID.

    Args:
        task_text: The raw text of the task, as sent by the user.
    """
    if not task_text or not task_text.strip():
        print("[STORAGE WARNING] Ignored empty task_text in add_raw_task.")
        return

    with _LOCK:
        try:
            tasks = _read_all_tasks()
            new_id = _next_id(tasks)
            tasks.append({
                "id": new_id,
                "text": task_text.strip(),
                "status": "pending",
            })
            _write_all_tasks(tasks)
            print(f"[STORAGE] Logged new pending task #{new_id}: '{task_text.strip()}'")
        except Exception as exc:  # noqa: BLE001
            print(f"[STORAGE ERROR] Failed to add raw task: {exc}")
            raise


def get_pending_tasks_string() -> str:
    """
    Retrieve all pending task texts, each prefixed with its ID, joined
    by newlines.

    Returns:
        A newline-separated string in the form "[ID] Task text", or an
        empty string if there are no pending tasks.
    """
    with _LOCK:
        try:
            tasks = _read_all_tasks()
            pending_lines = [
                f"[{task.get('id')}] {task.get('text', '').strip()}"
                for task in tasks
                if task.get("status") == "pending" and task.get("text", "").strip()
            ]
            return "\n".join(pending_lines)
        except Exception as exc:  # noqa: BLE001
            print(f"[STORAGE ERROR] Failed to read pending tasks: {exc}")
            return ""


def mark_task_complete(task_id: int) -> bool:
    """
    Mark the task matching `task_id` as 'completed'.

    Args:
        task_id: The integer ID of the task to complete.

    Returns:
        True if a matching pending task was found and updated, False otherwise.
    """
    with _LOCK:
        try:
            tasks = _read_all_tasks()
            found = False

            for task in tasks:
                if task.get("id") == task_id:
                    if task.get("status") == "completed":
                        print(f"[STORAGE] Task #{task_id} was already marked completed.")
                        return False
                    task["status"] = "completed"
                    found = True
                    break

            if found:
                _write_all_tasks(tasks)
                print(f"[STORAGE] Marked task #{task_id} as completed.")
                return True

            print(f"[STORAGE WARNING] No task found with id #{task_id}.")
            return False

        except Exception as exc:  # noqa: BLE001
            print(f"[STORAGE ERROR] Failed to mark task #{task_id} complete: {exc}")
            return False