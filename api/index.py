"""Vercel serverless entry point."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ["RENDER"] = "true"
from web.app import app
