#!/usr/bin/env python3
"""Production entry point for Render / cloud deployment."""
import os, sys, logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prod")

# Render provides PORT env var
port = int(os.environ.get("PORT", 8000))
host = "0.0.0.0"

logger.info(f"Starting production server on {host}:{port}")

import uvicorn
uvicorn.run("web.app:app", host=host, port=port, log_level="info")
