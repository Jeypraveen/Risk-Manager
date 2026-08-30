import requests
import json
import uuid

BASE_URL = "http://localhost:8000"

def score(data, files=None):
    r = requests.post(f"{BASE_URL}/api/score", data=data, files=files)
    return r.json()

print("== A. Legitimate Return (High account age, low velocity, normal order value) ==")
req_a = {
    "return_id": f"TEST-LEGIT-{uuid.uuid4().hex[:6]}",
    "account_age_days": 1000,
    "return_velocity_30d": 0,
    "order_value": 500,
    "user_return_rate": 0.0,
    "product_category_electronics": 0,
    "product_category_fashion": 1,
    "is_cod": 0
}
with open("data/images/catalog/iphone.jpg", "rb") as c, open("data/images/returns/iphone_return.jpg", "rb") as r:
    res_a = score(req_a, files={"catalog_image": c, "return_image": r})
print(json.dumps(res_a, indent=2))

print("\n== B. Suspicious Return (New account, high velocity, high value, COD) ==")
req_b = {
    "return_id": f"TEST-FRAUD-{uuid.uuid4().hex[:6]}",
    "account_age_days": 2,
    "return_velocity_30d": 15,
    "order_value": 50000,
    "user_return_rate": 0.9,
    "product_category_electronics": 1,
    "product_category_fashion": 0,
    "is_cod": 1
}
with open("data/images/catalog/smartphone.png", "rb") as c, open("data/images/returns/substitution_fraud.png", "rb") as r:
    res_b = score(req_b, files={"catalog_image": c, "return_image": r})
print(json.dumps(res_b, indent=2))

print("\n== C. No Photo Return ==")
req_c = {
    "return_id": f"TEST-NOPHOTO-{uuid.uuid4().hex[:6]}",
    "account_age_days": 150,
    "return_velocity_30d": 2,
    "order_value": 1000,
    "user_return_rate": 0.1,
    "product_category_electronics": 0,
    "product_category_fashion": 1,
    "is_cod": 0
}
res_c = score(req_c)
print(json.dumps(res_c, indent=2))

print("\n== D. 3 Consecutive Re-photo Requests ==")
rid = f"TEST-REPHOTO-{uuid.uuid4().hex[:6]}"
for i in range(1, 4):
    print(f"\n--- Request {i} ---")
    req_d = {
        "return_id": rid,
        "account_age_days": 150,
        "return_velocity_30d": 2,
        "order_value": 1000,
        "user_return_rate": 0.1,
        "product_category_electronics": 0,
        "product_category_fashion": 1,
        "is_cod": 0
    }
    res_d = score(req_d)
    print(json.dumps(res_d, indent=2))

print("\n== E. Circuit Breaker - Timeout ==")
req_e = req_c.copy()
req_e["return_id"] = f"TEST-TIMEOUT-{uuid.uuid4().hex[:6]}"
requests.post(f"{BASE_URL}/api/trigger-failure", data={"failure_type": "timeout"})
with open("data/images/catalog/iphone.jpg", "rb") as c, open("data/images/returns/iphone_return.jpg", "rb") as r:
    res_timeout = score(req_e, files={"catalog_image": c, "return_image": r})
print(json.dumps(res_timeout, indent=2))

print("\n== F. Circuit Breaker - Checkpoint Mismatch ==")
req_f = req_c.copy()
req_f["return_id"] = f"TEST-MISMATCH-{uuid.uuid4().hex[:6]}"
requests.post(f"{BASE_URL}/api/reset-failure")
requests.post(f"{BASE_URL}/api/trigger-failure", data={"failure_type": "checkpoint_mismatch"})
with open("data/images/catalog/iphone.jpg", "rb") as c, open("data/images/returns/iphone_return.jpg", "rb") as r:
    res_mismatch = score(req_f, files={"catalog_image": c, "return_image": r})
print(json.dumps(res_mismatch, indent=2))

requests.post(f"{BASE_URL}/api/reset-failure")

print("\n== DB Audit Check ==")
import sqlite3
conn = sqlite3.connect("audit.db")
c = conn.cursor()
c.execute("SELECT return_id, decision, forced_by, failure_event FROM decisions ORDER BY timestamp DESC LIMIT 2")
for row in c.fetchall():
    print(row)

