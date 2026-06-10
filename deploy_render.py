#!/usr/bin/env python3
"""Deploy to Render using API key."""
import requests, json, sys, time

KEY = "rnd_LVgQFwhKQVvOIU3N0gtYzjbKc10o"
HEADERS = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
BASE = "https://api.render.com/v1"

# Step 1: Get owner ID
print("[1/4] Getting owner...")
r = requests.get(f"{BASE}/owners", headers=HEADERS, timeout=10)
r.raise_for_status()
owner = r.json()[0]["owner"]
owner_id = owner["id"]
print(f"  Owner: {owner['name']} ({owner_id})")

# Step 2: Check for existing services
print("[2/4] Checking existing services...")
r = requests.get(f"{BASE}/services", headers=HEADERS, timeout=10)
services = r.json()
for s in services:
    if s.get("name") == "amazon-sales-predictor":
        print(f"  Found existing service: {s['id']} - deleting...")
        requests.delete(f"{BASE}/services/{s['id']}", headers=HEADERS, timeout=10)
        time.sleep(2)
        print("  Deleted.")
print("  No conflicts.")

# Step 3: Create web service
print("[3/4] Creating web service...")
payload = {
    "type": "web_service",
    "name": "amazon-sales-predictor",
    "ownerId": owner_id,
    "repo": "https://github.com/lavanurusumathi-web/amazon-sales-prediction-system",
    "branch": "main",
    "autoDeploy": "yes",
    "serviceDetails": {
        "runtime": "python",
        "plan": "free",
        "region": "oregon",
        "envSpecificDetails": {
            "buildCommand": "pip install -r requirements-prod.txt && python pretrain_light.py",
            "startCommand": "python run_prod.py"
        }
    }
}
r = requests.post(f"{BASE}/services", headers=HEADERS, json=payload, timeout=30)
print(f"  Status: {r.status_code}")
resp = r.json()
service_id = resp.get("id") or resp.get("serviceId")
if not service_id:
    print(f"  ERROR: {json.dumps(resp, indent=2)}")
    sys.exit(1)
print(f"  Service created: {service_id}")

# Step 4: Poll for deploy status
print("[4/4] Waiting for first deploy...")
for attempt in range(30):
    time.sleep(10)
    r = requests.get(f"{BASE}/services/{service_id}", headers=HEADERS, timeout=10)
    svc = r.json()
    details = svc.get("serviceDetails", {})
    url = details.get("url", "")
    state = details.get("deployStatus", details.get("suspended", "unknown"))
    print(f"  [{attempt+1}] Status: {state} | URL: {url or 'pending...'}")
    if url and "live" in str(state).lower():
        print(f"\n  LIVE at https://{url}")
        break
else:
    print(f"\n  Deploy still in progress. Check at https://dashboard.render.com")
