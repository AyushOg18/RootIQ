"""Root-level Streamlit entry point for cloud deployment (Streamlit Community Cloud).

Directly loads and executes app/app.py without relying on package imports.
"""

import os
import sys
from pathlib import Path
import runpy

# Ensure project root and app directory are on sys.path
ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Run app/app.py in the current namespace / as __main__
app_file = APP_DIR / "app.py"
runpy.run_path(str(app_file), run_name="__main__")
