"""
Yelp RAG Streamlit Application Entrypoint
Delegates to yelp_streamlit_ui/app.py
"""

import sys
import os
from pathlib import Path

# Redirect execution to yelp_streamlit_ui/app.py
UI_APP_PATH = Path(__file__).resolve().parent / "yelp_streamlit_ui" / "app.py"

with open(UI_APP_PATH, "r", encoding="utf-8") as f:
    code = compile(f.read(), str(UI_APP_PATH), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(UI_APP_PATH)})
