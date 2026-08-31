import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.vision.similarity import compute_similarity

genuine_pairs = [
    ("smartphone.png", "smartphone_legitimate.png"),
    ("headphones.png", "headphones_legitimate.png"),
    ("iphone.jpg", "iphone_return.jpg"),
    ("bag.png", "bag_nudge_blurry.jpg"),
    ("cetaphil.jpg", "cetaphil_return.jpg"),
    ("shoes.png", "shoes_legitimate.jpg"),
    ("watch.png", "watch_legitimate.jpg"),
]

mismatch_pairs = [
    ("smartphone.png", "empty_box_fraud.png"),
    ("smartphone.png", "substitution_fraud.png"),
    ("watch.png", "shoes_legitimate.jpg"),
    ("iphone.jpg", "cetaphil_return.jpg"),
    ("headphones.png", "return_fraud_box_1787926845277.jpg"),
]

print(f"{'Catalog Image':<18} | {'Return Image':<32} | {'Type':<12} | Score")
print("-" * 75)

for cat, ret in genuine_pairs:
    score = compute_similarity(f"data/images/catalog/{cat}", f"data/images/returns/{ret}")
    if score is not None:
        print(f"{cat:<18} | {ret:<32} | {'GENUINE':<12} | {score:.4f}")

for cat, ret in mismatch_pairs:
    score = compute_similarity(f"data/images/catalog/{cat}", f"data/images/returns/{ret}")
    if score is not None:
        print(f"{cat:<18} | {ret:<32} | {'MISMATCH':<12} | {score:.4f}")
