"""Small local feedback store for assessment quality signals."""

import json
from datetime import datetime, timezone
from pathlib import Path

FEEDBACK_PATH = Path("assessment_feedback.json")


def save_feedback(payload):
    entries = []
    if FEEDBACK_PATH.exists():
        try:
            entries = json.loads(FEEDBACK_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries = []
    if not isinstance(entries, list):
        entries = []
    entry = {
        "rating": int(payload.get("rating", 0)),
        "comment": str(payload.get("comment", "")).strip()[:1000],
        "speaker": str(payload.get("speaker", "Speaker")).strip()[:120],
        "filename": str(payload.get("filename", "")).strip()[:255],
        "score": payload.get("score"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    entries.append(entry)
    FEEDBACK_PATH.write_text(json.dumps(entries[-500:], indent=2), encoding="utf-8")
    return entry
