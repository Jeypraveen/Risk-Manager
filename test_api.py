import requests
import json
import uuid

url = "http://localhost:8000/api/score"
data = {
    "order_value": 1500,
    "account_age_days": 1500,
    "prior_returns_count": 0,
    "prior_return_approval_rate": 1.0,
    "return_velocity_7d": 0,
    "is_cod": 0,
    "delivery_to_return_hours": 72,
    "item_category": "electronics",
    "address_order_distance_km": 2.0,
}

cases = [
    ("no photo", None, None),
    ("genuine photo", "data/images/catalog/iphone.jpg", "data/images/returns/iphone_return.jpg"),
    ("SUBSTITUTION", "data/images/catalog/smartphone.png", "data/images/returns/substitution_fraud.png"),
    ("SUBSTITUTION 2", "data/images/catalog/shoes_red.jpg", "data/images/returns/return_fraud_box_1787926845277.jpg"),
    ("EMPTY BOX", "data/images/catalog/smartphone.png", "data/images/returns/empty_box_fraud.png"),
]

for name, cat_path, ret_path in cases:
    files = {}
    if cat_path:
        files["catalog_image"] = open(cat_path, "rb")
    if ret_path:
        files["return_image"] = open(ret_path, "rb")
        
    try:
        # Use a fresh return_id for each to avoid state pollution (e.g. store credit threshold penalties)
        req_data = data.copy()
        req_data["return_id"] = f"throwaway-{uuid.uuid4().hex[:6]}"
        resp = requests.post(url, data=req_data, files=files)
        res = resp.json()
        print(f"{name:20s} trust={res['scores']['trust_score']:.3f} -> {res['decision']['outcome']}")
    finally:
        for f in files.values():
            f.close()
