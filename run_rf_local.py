"""Launch with small dataset + Random Forest model on localhost."""
import os
import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["RENDER"] = "true"

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn
from web.app import app

print("Starting Amazon Sales Predictor with Random Forest + small dataset")
print("Dashboard: http://127.0.0.1:8000/")

uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
