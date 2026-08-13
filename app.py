"""Root app entry — FastAPI auto-detected by Vercel."""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
os.environ["RENDER"] = "true"
from web.app import app
