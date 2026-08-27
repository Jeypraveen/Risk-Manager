"""Core check script avoiding PowerShell f-string quoting issues."""
import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.fusion.decision import make_decision  # type: ignore # noqa: E402
from src.recovery.circuit_breaker import CircuitBreaker, FailureType  # type: ignore # noqa: E402
from src.audit.trail import AuditTrail  # type: ignore # noqa: E402
from src.fusion.meta_learner import MetaLearner  # type: ignore # noqa: E402
import numpy as np  # type: ignore # noqa: E402

PASS = []
FAIL = []

def ok(label, detail=""):
    PASS.append(label)
    print(f"  [OK]  {label}")

def fail(label, detail=""):
    FAIL.append(label)
    print(f"  [FAIL] {label}: {detail}")

print("Decision logic:")
r0 = make_decision(0.5, modality_confidence=0.0, rephoto_count=0)
r1 = make_decision(0.5, modality_confidence=0.0, rephoto_count=1)
r2 = make_decision(0.5, modality_confidence=0.0, rephoto_count=2)
(ok if r0.decision.value == 'nudge' else fail)("rephoto=0 -> NUDGE", r0.decision.value)
(ok if r1.decision.value == 'nudge' else fail)("rephoto=1 -> NUDGE", r1.decision.value)
(ok if r2.decision.value == 'manual_review' else fail)("rephoto=2 -> MANUAL_REVIEW (cap)", r2.decision.value)
(ok if r2.forced_by == 'rephoto_cap' else fail)("cap forced_by=rephoto_cap", str(r2.forced_by))
vf = make_decision(0.9, vision_failed=True)
(ok if vf.decision.value == 'manual_review' else fail)("vision_failed -> MANUAL_REVIEW", vf.decision.value)
(ok if vf.forced_by == 'circuit_breaker' else fail)("vision_failed forced_by=circuit_breaker", str(vf.forced_by))
auto_r = make_decision(0.9)
manual_r = make_decision(0.1)
(ok if auto_r.decision.value == 'auto_approve' else fail)("high score -> AUTO_APPROVE", auto_r.decision.value)
(ok if manual_r.decision.value == 'manual_review' else fail)("low score -> MANUAL_REVIEW", manual_r.decision.value)

print("Circuit breaker:")
cb = CircuitBreaker()
no_img = cb.check_image_availability(None)
(ok if no_img and no_img.failure_type == FailureType.IMAGE_UNAVAILABLE else fail)("None image -> IMAGE_UNAVAILABLE")
real_img = cb.check_image_availability("some/path.jpg")
(ok if real_img is None else fail)("real path -> no failure")
bad_emb = np.zeros(768)
dim_fail = cb.check_embedding_dimensions(bad_emb)
(ok if dim_fail and dim_fail.failure_type == FailureType.VISION_MODEL_FAILURE else fail)("dim 768 -> VISION_MODEL_FAILURE")
good_emb = np.zeros(384)
no_fail = cb.check_embedding_dimensions(good_emb)
(ok if no_fail is None else fail)("dim 384 -> no failure")
result, err = cb.wrap_vision_call(lambda: int("invalid"))
(ok if result is None and err is not None else fail)("wrap catches exception")

print("Meta-learner:")
ml = MetaLearner()
(ok if ml.is_trained else fail)("meta-learner is trained")
if ml.is_trained:
    nv = ml.predict(0.7, modality_confidence=0.0)
    gv = ml.predict(0.7, semantic_similarity=0.9, modality_confidence=1.0, empty_box_flag=0)
    bv = ml.predict(0.7, semantic_similarity=0.2, modality_confidence=1.0, empty_box_flag=1)
    (ok if gv > nv else fail)("good vision > no vision", f"gv={gv:.3f} nv={nv:.3f}")
    (ok if bv < nv else fail)("bad vision < no vision", f"bv={bv:.3f} nv={nv:.3f}")

print("Audit trail:")
# Use NamedTemporaryFile to avoid deprecated mktemp
tmp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
tmp_path = Path(tmp_db.name)
tmp_db.close()

try:
    audit = AuditTrail(db_path=tmp_path)
    audit.log_decision(
        return_id='TEST-001', tabular_score=0.8, semantic_similarity=0.9,
        empty_box_flag=0, modality_confidence=1.0, trust_score=0.85,
        decision='auto_approve', model_version='v0.1.0',
    )
    decisions = audit.get_decisions_for_return('TEST-001')
    stats = audit.get_stats()
    (ok if len(decisions) == 1 else fail)("writes a row", str(len(decisions)))
    dec_val = decisions[0]['decision']
    (ok if dec_val == 'auto_approve' else fail)("reads correct decision", dec_val)
    (ok if stats['total_decisions'] == 1 else fail)("stats total=1", str(stats['total_decisions']))

finally:
    try:
        tmp_path.unlink()
    except Exception:
        pass

print()
print(f"PASSED={len(PASS)}  FAILED={len(FAIL)}")
sys.exit(0 if not FAIL else 1)
