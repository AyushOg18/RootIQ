"""Root-level Streamlit entry point for cloud deployment (Streamlit Community Cloud).

Streamlit Community Cloud looks for streamlit_app.py at the repo root by default.
This file sets up sys.path and imports app.app — which auto-calls main() via its
own _in_streamlit guard. Do NOT call main() here to avoid a double-call.
"""

import sys
from pathlib import Path

# Ensure project root and app/ dir are on sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

# Importing app.app triggers its _in_streamlit guard which calls main() automatically
import app.app  # noqa: F401, E402
