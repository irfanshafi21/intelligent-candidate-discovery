"""
Local settings persistence — stores non-sensitive UI preferences (default
theme, notification toggles) in a small JSON file next to the app, so they
survive app restarts on the same machine without needing a database.

Deliberately does NOT store API keys or secrets here — those stay in
st.secrets/.streamlit/secrets.toml (or a session-only override), never
written to this file.
"""

import json
import os

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".local_settings.json")

DEFAULTS = {
    "default_theme": "light",   # "light" | "dark"
    "show_success_toasts": True,
}


def load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        return dict(DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            data = json.load(f)
        return {**DEFAULTS, **data}
    except Exception:
        return dict(DEFAULTS)


def save_settings(updates: dict) -> bool:
    try:
        current = load_settings()
        current.update(updates)
        with open(SETTINGS_PATH, "w") as f:
            json.dump(current, f, indent=2)
        return True
    except Exception:
        return False
