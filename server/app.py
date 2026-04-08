"""
OpenEnv-compliant server entry point.
This module imports the FastAPI app from the root server.py file.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from root server.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import app

__all__ = ["app"]
