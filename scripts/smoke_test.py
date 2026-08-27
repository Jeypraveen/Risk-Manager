"""Smoke test — runs without a live server, tests all core modules directly."""
import sys
import os
import tempfile
sys.path.insert(0, '.')

from pathlib import Path

PASS = 0
FAIL = 0

def check(label, condition, detail=""):
    global PASS, FAIL
    if condition:
        print(f"  [OK]  {label}")
        PASS += 1
    else:
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))
        FAIL += 1

print("=" * 55)
print("  Smoke Test — Return-Risk Scorer")
print("=" * 55)

# ── Imports ──
print("\n--- Imports ---")
try:
    from src.config import THRESHOLD_AUTO_APPROVE, THRESHOLD_MANUAL_REVIEW, MAX_REPHOTO_REQUESTS
    from src.tabular.predict import TabularScorer
    from src.fusion.decision import make_decision
    from src.fusion.meta_learner import MetaLearner
    from src.recovery.circuit_breaker import CircuitBreaker, FailureType
    from src.audit.trail import AuditTrail
    from src.vision.pipeline import run_vision_pipeline
    check("All module imports", True)
except Exception as e:
    check("All module imports", False, str(e))

# ── Tabular scorer ──
print("\n--- Tabular scorer ---")
try:
    scorer = TabularScorer()
    check("Model loads", True)
    check("Category medians saved in checkpoint", scorer.category_medians is not None,
          "medians=None means model was saved before the fix; retrain needed")

    legit = scorer.predict({
        'order_value': 1500, 'account_age_days': 1000, 'prior_returns_count': 0,
        'prior_return_approval_rate': 1.0, 'return_velocity_7d': 0, 'is_cod': 0,
        'delivery_to_return_hours': 72, 'item_category': 'electronics',
        'address_order_distance_km': 2.0,
    })
    fraud = scorer.predict({
        'order_value': 45000, 'account_age_days': 2, 'prior_returns_count': 5,
        'prior_return_approval_rate': 0.5, 'return_velocity_7d': 3, 'is_cod': 1,
        'delivery_to_return_hours': 10, 'item_category': 'electronics',
        'address_order_distance_km': 150.0,
    })
    check("Legit score > Fraud score", legit > fraud,
          f"legit={legit:.3f}, fraud={fraud:.3f}")
    check("Legit score > 0.5", legit > 0.5, f"legit={legit:.3f}")
    check("Fraud score < 0.5", fraud < 0.5, f"fraud={fraud:.3f}")
except FileNotFoundError as e:
    check("Model loads", False, "Run: python -m src.tabular.train")

# ── Meta-learner ──
print("\n--- Meta-learner fusion ---")
try:
    ml = MetaLearner()
    check("Meta-learner loads", True)
    check("Meta-learner is trained", ml.is_trained,
          "Run: python -m scripts.train_meta_learner")
    if ml.is_trained:
        no_vis = ml.predict(0.7, modality_confidence=0.0)
        good_vis = ml.predict(0.7, semantic_similarity=0.9, modality_confidence=1.0, empty_box_flag=0)
        bad_vis = ml.predict(0.7, semantic_similarity=0.2, modality_confidence=1.0, empty_box_flag=1)
        check("Good vision raises trust vs no-vision", good_vis > no_vis,
              f"good={good_vis:.3f}, no-vis={no_vis:.3f}")
        check("Bad vision lowers trust vs no-vision", bad_vis < no_vis,
              f"bad={bad_vis:.3f}, no-vis={no_vis:.3f}")
except Exception as e:
    check("Meta-learner loads", False, str(e))

# ── Decision logic ──
print("\n--- Decision logic ---")
try:
    r0 = make_decision(0.5, modality_confidence=0.0, rephoto_count=0)
    r1 = make_decision(0.5, modality_confidence=0.0, rephoto_count=1)
    r2 = make_decision(0.5, modality_confidence=0.0, rephoto_count=2)
    check("rephoto_count=0 → NUDGE/request_photo", r0.decision.value == 'nudge')
    check("rephoto_count=1 → NUDGE/request_photo", r1.decision.value == 'nudge')
    check("rephoto_count=2 → MANUAL_REVIEW (cap hit)", r2.decision.value == 'manual_review',
          f"got {r2.decision.value}")
    check("Cap forced_by=rephoto_cap", r2.forced_by == 'rephoto_cap', f"got {r2.forced_by}")

    vf = make_decision(0.9, vision_failed=True)
    check("Vision failure → MANUAL_REVIEW (not auto-approve)", vf.decision.value == 'manual_review')
    check("Vision failure forced_by=circuit_breaker", vf.forced_by == 'circuit_breaker')

    auto = make_decision(0.9)
    manual = make_decision(0.1)
    check("High score → AUTO_APPROVE", auto.decision.value == 'auto_approve')
    check("Low score → MANUAL_REVIEW", manual.decision.value == 'manual_review')
except Exception as e:
    check("Decision logic", False, str(e))

# ── Circuit breaker ──
print("\n--- Circuit breaker ---")
try:
    import numpy as np
    cb = CircuitBreaker()
    no_img = cb.check_image_availability(None)
    check("None image → IMAGE_UNAVAILABLE", no_img is not None and no_img.failure_type == FailureType.IMAGE_UNAVAILABLE)
    real_img = cb.check_image_availability("some/path.jpg")
    check("Non-None path → no failure", real_img is None)

    cb2 = CircuitBreaker()
    cb2.enable_checkpoint_mismatch_simulation()
    mismatch = cb2.check_embedding_dimensions(None)
    check("Mismatch sim → VISION_MODEL_FAILURE", mismatch is not None and mismatch.failure_type == FailureType.VISION_MODEL_FAILURE)

    good_emb = np.zeros(384)
    no_fail = cb.check_embedding_dimensions(good_emb)
    check("Correct dim 384 → no failure", no_fail is None)

    bad_emb = np.zeros(768)
    dim_fail = cb.check_embedding_dimensions(bad_emb)
    check("Wrong dim 768 → VISION_MODEL_FAILURE", dim_fail is not None)

    result, err = cb.wrap_vision_call(lambda: 1/0)
    check("wrap_vision_call catches exceptions", result is None and err is not None)
except Exception as e:
    check("Circuit breaker", False, str(e))

# ── Audit trail ──
print("\n--- Audit trail ---")
try:
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        tmp_db = f.name
    audit = AuditTrail(db_path=Path(tmp_db))
    aid = audit.log_decision(
        return_id='TEST-001', tabular_score=0.8, semantic_similarity=0.9,
        empty_box_flag=0, modality_confidence=1.0, trust_score=0.85,
        decision='auto_approve', model_version='v0.1.0',
    )
    decisions = audit.get_decisions_for_return('TEST-001')
    stats = audit.get_stats()
    os.unlink(tmp_db)
    check("Writes a row", len(decisions) == 1)
    check("Reads back correct decision", decisions[0]['decision'] == 'auto_approve')
    check("Stats reports total_decisions=1", stats['total_decisions'] == 1)
except Exception as e:
    check("Audit trail", False, str(e))

# ── Image files ──
print("\n--- Image assets ---")
catalog_dir = Path("data/images/catalog")
returns_dir = Path("data/images/returns")
catalog_images = list(catalog_dir.glob("*.png")) + list(catalog_dir.glob("*.jpg"))
returns_images = list(returns_dir.glob("*.png")) + list(returns_dir.glob("*.jpg"))
check("Catalog images present (>=5)", len(catalog_images) >= 5, f"found {len(catalog_images)}")
check("Return images present (>=3)", len(returns_images) >= 3, f"found {len(returns_images)}")
check("image_mapping.csv exists", Path("data/images/image_mapping.csv").exists())

# ── Summary ──
print()
print("=" * 55)
print(f"  PASSED: {PASS}   FAILED: {FAIL}")
print("=" * 55)
sys.exit(0 if FAIL == 0 else 1)
