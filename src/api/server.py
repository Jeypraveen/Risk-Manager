"""
FastAPI server for the Return-Risk Scorer.

Endpoints:
  POST /api/score             — Score a return request
  POST /api/trigger-failure   — Trigger a simulated failure mode
  POST /api/reset-failure     — Reset all failure simulations
  GET  /api/audit/stats       — Aggregate audit stats  (MUST be before /{return_id})
  GET  /api/audit/recent      — Most recent decisions  (MUST be before /{return_id})
  GET  /api/audit/failures    — All failure events     (MUST be before /{return_id})
  GET  /api/audit/{return_id} — Audit trail for a specific return
  GET  /api/health            — Health check

Route ordering note: FastAPI resolves parameterized paths greedily.
Static paths (/stats, /recent, /failures) MUST be registered BEFORE
the parameterized path (/{return_id}) or they will be silently shadowed.
"""

import json
import logging
import os
import sqlite3
import sys
import tempfile
import uuid
import asyncio
import shutil
try:
    import torch
    torch.set_num_threads(1)
except ImportError:
    pass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import (
    C_FP,
    C_FN,
    THRESHOLD_AUTO_APPROVE,
    THRESHOLD_MANUAL_REVIEW,
    AUDIT_DB_PATH,
    AUDIT_TABLE_NAME,
    MODELS_DIR,
    MAX_REPHOTO_REQUESTS,
    ITEM_CATEGORIES,
)
from src.tabular.predict import TabularScorer
from src.fusion.decision import make_decision, Decision
from src.fusion.meta_learner import MetaLearner
from src.recovery.circuit_breaker import CircuitBreaker, FailureType
from src.audit.trail import AuditTrail
from src.vision.pipeline import run_vision_pipeline


# ── Setup ──
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Risk Manager",
    description="AI-powered return fraud detection.",
    version="1.0.0",
)

# CORS — allow all origins for demo (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for demo UI
demo_dir = Path(__file__).resolve().parent.parent.parent / "demo"
if demo_dir.exists():
    app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

# ── Global state ──
scorer: Optional[TabularScorer] = None
meta_learner: Optional[MetaLearner] = None
circuit_breaker = CircuitBreaker()
audit_trail = AuditTrail()


def get_scorer() -> TabularScorer:
    """Lazy-load the tabular scorer. Raises HTTPException with clear message on failure."""
    global scorer
    if scorer is None:
        try:
            scorer = TabularScorer()
        except FileNotFoundError as e:
            logger.error(f"Tabular model not found: {e}")
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model not ready. Run `python -m src.tabular.train` first "
                    "to train and save the tabular scorer."
                ),
            )
    return scorer


def get_meta_learner() -> Optional[MetaLearner]:
    """Lazy-load the meta-learner. Returns None silently if not trained yet."""
    global meta_learner
    if meta_learner is None:
        ml = MetaLearner()
        meta_learner = ml  # May or may not be trained — predict() handles both
    return meta_learner


def _get_rephoto_count_from_db(return_id: str) -> int:
    """
    Query the audit DB for the number of REQUEST_PHOTO nudges already issued
    for this return_id. This replaces client-supplied rephoto_count — the
    server owns the state, not the caller.
    """
    try:
        with sqlite3.connect(str(AUDIT_DB_PATH)) as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}
                WHERE return_id = ?
                  AND nudge_type = 'request_photo'
                """,
                (return_id,),
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        logger.warning(f"Could not query rephoto_count from DB: {e}. Falling back to max cap.")
        return MAX_REPHOTO_REQUESTS

def _get_store_credit_count_from_db(return_id: str) -> int:
    """
    Query the audit DB for the number of OFFER_STORE_CREDIT nudges already issued
    for this return_id. This counts per-return escalation (how many times this
    specific return was offered store credit), not per-account.
    """
    try:
        with sqlite3.connect(str(AUDIT_DB_PATH)) as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) FROM {AUDIT_TABLE_NAME}
                WHERE return_id = ?
                  AND nudge_type = 'offer_store_credit'
                """,
                (return_id,),
            ).fetchone()
            return int(row[0]) if row else 0
    except Exception as e:
        logger.warning(f"Could not query store_credit_count from DB: {e}. Falling back to 0.")
        return 0


# ── Endpoints ──

@app.get("/", include_in_schema=False)
async def root_redirect():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/demo/index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(content=b"", media_type="image/x-icon", status_code=204)


@app.get("/api/health")
async def health():
    """Health check."""
    tabular_ready = (MODELS_DIR / "tabular_scorer.joblib").exists()
    meta_ready = (MODELS_DIR / "meta_learner.joblib").exists()
    return {
        "status": "healthy",
        "model_loaded": tabular_ready,
        "meta_learner_trained": meta_ready,
        "circuit_breaker": circuit_breaker.get_simulation_status(),
    }


# ── ⚠️ IMPORTANT: Static audit sub-paths MUST come before /{return_id} ──

@app.get("/api/audit/stats")
async def get_audit_stats():
    """Get aggregate audit statistics."""
    return audit_trail.get_stats()


@app.get("/api/audit/recent")
async def get_recent_audit(limit: int = 20):
    """Get the most recent decisions."""
    return {"decisions": audit_trail.get_recent_decisions(limit)}


@app.get("/api/audit/failures")
async def get_failures():
    """Get all failure events."""
    return {"failures": audit_trail.get_failure_events()}


@app.get("/api/audit/{return_id}")
async def get_audit(return_id: str):
    """Get all decisions for a specific return."""
    decisions = audit_trail.get_decisions_for_return(return_id)
    return {"return_id": return_id, "decisions": decisions}


# ── Scoring endpoint ──

@app.post("/api/score")
async def score_return(
    return_id: str = Form(default=None),
    order_value: float = Form(...),
    account_age_days: int = Form(...),
    prior_returns_count: int = Form(default=0),
    prior_return_approval_rate: float = Form(default=0.8),
    return_velocity_7d: int = Form(default=0),
    is_cod: int = Form(default=0),
    delivery_to_return_hours: float = Form(default=120),
    item_category: str = Form(default="electronics"),
    address_order_distance_km: float = Form(default=5.0),
    catalog_image: Optional[UploadFile] = File(default=None),
    return_image: Optional[UploadFile] = File(default=None),
):
    """
    Score a return request through the full pipeline.

    Returns the tabular score, vision signals (if images provided),
    fused trust score, decision, and audit trail ID.

    Note on rephoto_count: The server derives this from the audit DB
    (count of prior REQUEST_PHOTO nudges for this return_id) rather than
    trusting the caller's value. This prevents clients from gaming the cap.
    """
    if not return_id:
        return_id = f"RET-{uuid.uuid4().hex[:8].upper()}"

    # ── Input Validation ──
    if order_value < 0:
        raise HTTPException(status_code=400, detail="order_value cannot be negative")
    if account_age_days < 0:
        raise HTTPException(status_code=400, detail="account_age_days cannot be negative")
    if account_age_days > 36500:
        raise HTTPException(status_code=400, detail="account_age_days exceeds maximum (36500 = 100 years)")
    if prior_returns_count < 0:
        raise HTTPException(status_code=400, detail="prior_returns_count cannot be negative")
    if return_velocity_7d < 0:
        raise HTTPException(status_code=400, detail="return_velocity_7d cannot be negative")
    if delivery_to_return_hours < 0:
        raise HTTPException(status_code=400, detail="delivery_to_return_hours cannot be negative")
    if prior_return_approval_rate < 0.0 or prior_return_approval_rate > 1.0:
        raise HTTPException(status_code=400, detail="prior_return_approval_rate must be between 0.0 and 1.0")
    if address_order_distance_km < 0:
        raise HTTPException(status_code=400, detail="address_order_distance_km cannot be negative")
    if is_cod not in (0, 1):
        raise HTTPException(status_code=400, detail="is_cod must be 0 or 1")
    if order_value > 500_000:
        raise HTTPException(status_code=400, detail="order_value exceeds maximum (500000)")
    if delivery_to_return_hours > 8760:
        raise HTTPException(status_code=400, detail="delivery_to_return_hours exceeds maximum (8760 = 1 year)")

    if item_category not in ITEM_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"item_category must be one of: {', '.join(ITEM_CATEGORIES)}")

    ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

    async def _validate_image(img: Optional[UploadFile], name: str):
        if not img or not img.filename:
            return
        if img.content_type not in ALLOWED_IMAGE_TYPES:
            raise HTTPException(status_code=400, detail=f"{name} must be a valid image (JPEG/PNG/WEBP)")
        
        # Read slightly more than max size to detect oversize efficiently
        content = await img.read(MAX_IMAGE_SIZE + 1)
        if len(content) > MAX_IMAGE_SIZE:
            raise HTTPException(status_code=400, detail=f"{name} file size exceeds 10MB limit")
        await img.seek(0)

    await _validate_image(catalog_image, "catalog_image")
    await _validate_image(return_image, "return_image")

    # ── Step 1: Tabular scoring ──
    # Failure here raises HTTPException; nothing is logged (model not ready is
    # a server-side issue, not an auditable decision event).
    request_data = {
        "order_value": order_value,
        "account_age_days": account_age_days,
        "prior_returns_count": prior_returns_count,
        "prior_return_approval_rate": prior_return_approval_rate,
        "return_velocity_7d": return_velocity_7d,
        "is_cod": is_cod,
        "delivery_to_return_hours": delivery_to_return_hours,
        "item_category": item_category,
        "address_order_distance_km": address_order_distance_km,
    }

    try:
        s = get_scorer()
        def _run_tabular():
            return s.predict(request_data), s.get_feature_contributions(request_data)
        tabular_score, feature_contributions = await asyncio.to_thread(_run_tabular)
    except HTTPException:
        raise  # Re-raise model-not-ready errors as-is
    except Exception as e:
        logger.error(f"Tabular scoring failed for {return_id}: {e}")
        # Log a failure audit row so the event is traceable
        try:
            await asyncio.to_thread(
                audit_trail.log_decision,
                return_id=return_id,
                tabular_score=None,
                semantic_similarity=None,
                empty_box_flag=None,
                modality_confidence=0.0,
                trust_score=0.0,
                decision="manual_review",
                failure_event="tabular_model_failure",
                failure_details=str(e),
                model_version="v0.1.0",
            )
        except Exception as log_e:
            logger.error(f"Audit logging itself failed: {log_e}")
        # SEC-FIX: Do not leak the raw exception to the client
        raise HTTPException(status_code=500, detail="An internal server error occurred while scoring the return.")

    # ── Step 2: Derive server-side counts ──
    rephoto_count = await asyncio.to_thread(_get_rephoto_count_from_db, return_id)
    prior_store_credit_count = await asyncio.to_thread(_get_store_credit_count_from_db, return_id)

    # ── Step 3: Vision pipeline ──
    catalog_path: Optional[str] = None
    return_path: Optional[str] = None
    temp_files: list[str] = []

    try:
        # Save uploaded images to temp files for vision pipeline
        if catalog_image and catalog_image.filename:
            suffix = Path(catalog_image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                await asyncio.to_thread(shutil.copyfileobj, catalog_image.file, tmp)
                catalog_path = tmp.name
                temp_files.append(catalog_path)

        if return_image and return_image.filename:
            suffix = Path(return_image.filename).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                await asyncio.to_thread(shutil.copyfileobj, return_image.file, tmp)
                return_path = tmp.name
                temp_files.append(return_path)

        # Run real vision pipeline — circuit breaker catches all exceptions
        vision_result = await asyncio.to_thread(
            run_vision_pipeline,
            catalog_image_path=catalog_path,
            return_image_path=return_path,
            circuit_breaker=circuit_breaker,
        )

        semantic_similarity = vision_result.semantic_similarity
        empty_box_flag = 1 if vision_result.empty_box_flag else 0 if vision_result.empty_box_flag is not None else None
        modality_confidence = vision_result.modality_confidence
        vision_failure_type = vision_result.failure_type
        vision_failure_message = vision_result.failure_message

        # Structured failure event for audit (if vision failed)
        vision_failure = None
        if vision_failure_type is not None:
            from src.recovery.circuit_breaker import FailureEvent
            vision_failure = FailureEvent(
                failure_type=FailureType(vision_failure_type) if vision_failure_type in [ft.value for ft in FailureType] else FailureType.VISION_MODEL_FAILURE,
                message=vision_failure_message or "Vision pipeline failure",
                details=vision_result.failure_details,
            )

    except Exception as e:
        logger.error(f"Unexpected error in vision pipeline for {return_id}: {e}")
        # Safe fallback: treat as no vision available
        semantic_similarity = None
        empty_box_flag = None
        modality_confidence = 0.0
        vision_failure = None
        vision_failure_type = None
    finally:
        # Always clean up temp files
        for path in temp_files:
            try:
                os.unlink(path)
            except OSError:
                pass

    # ── Step 4: Fusion ──
    ml = get_meta_learner()
    if ml is not None and ml.is_trained:
        # Use trained meta-learner for fusion
        trust_score = await asyncio.to_thread(
            ml.predict,
            tabular_score=tabular_score,
            semantic_similarity=semantic_similarity,
            empty_box_flag=empty_box_flag,
            modality_confidence=modality_confidence,
        )
    else:
        # Fallback: weighted combination when meta-learner isn't trained yet
        trust_score = tabular_score
        if modality_confidence > 0 and semantic_similarity is not None:
            eb = empty_box_flag if empty_box_flag is not None else 0
            vision_signal = semantic_similarity * 0.7 + (1 - eb) * 0.3
            trust_score = tabular_score * 0.6 + vision_signal * modality_confidence * 0.4

    # ── Step 5: Decision ──
    is_vision_failure = (
        vision_failure is not None
        and vision_failure.failure_type == FailureType.VISION_MODEL_FAILURE
    )

    # Determine the actual failure type for threshold logic
    actual_failure_type = None
    if vision_failure is not None:
        actual_failure_type = vision_failure.failure_type

    decision_result = make_decision(
        trust_score=trust_score,
        modality_confidence=modality_confidence,
        rephoto_count=rephoto_count,
        prior_store_credit_count=prior_store_credit_count,
        vision_failed=is_vision_failure,
        failure_type=actual_failure_type,
    )

    # ── Step 6: Audit logging ──
    config_snapshot = {
        "threshold_auto_approve": THRESHOLD_AUTO_APPROVE,
        "threshold_manual_review": THRESHOLD_MANUAL_REVIEW,
        "c_fp": C_FP,
        "c_fn": C_FN,
    }

    try:
        audit_id = await asyncio.to_thread(
            audit_trail.log_decision,
            return_id=return_id,
            tabular_score=tabular_score,
            semantic_similarity=semantic_similarity,
            empty_box_flag=empty_box_flag,
            modality_confidence=modality_confidence,
            trust_score=trust_score,
            decision=decision_result.decision.value,
            nudge_type=decision_result.nudge_type.value if decision_result.nudge_type else None,
            nudge_message=decision_result.nudge_message,
            rephoto_count=rephoto_count,
            prior_store_credit_count=prior_store_credit_count,
            failure_event=vision_failure.failure_type.value if vision_failure else None,
            failure_details=vision_failure.details if vision_failure else None,
            effective_approve_threshold=decision_result.effective_approve_threshold,
            effective_review_threshold=decision_result.effective_review_threshold,
            decision_reason=decision_result.reason,
            forced_by=decision_result.forced_by,
            model_version="v0.1.0",
            config_snapshot=config_snapshot,
        )
    except Exception as e:
        logger.error(f"Audit logging failed for {return_id}: {e}")
        audit_id = None  # Decision still returned; audit failure is not fatal

    return {
        "return_id": return_id,
        "audit_id": audit_id,
        "scores": {
            "tabular_score": round(tabular_score, 4),
            "semantic_similarity": round(semantic_similarity, 4) if semantic_similarity is not None else None,
            "empty_box_flag": empty_box_flag,
            "modality_confidence": round(modality_confidence, 4),
            "trust_score": round(trust_score, 4),
        },
        "decision": {
            "outcome": decision_result.decision.value,
            "nudge_type": decision_result.nudge_type.value if decision_result.nudge_type else None,
            "nudge_message": decision_result.nudge_message,
            "reason": decision_result.reason,
            "forced_by": decision_result.forced_by,
        },
        "thresholds": {
            "effective_approve": round(decision_result.effective_approve_threshold, 4),
            "effective_review": round(decision_result.effective_review_threshold, 4),
        },
        "failure": {
            "occurred": True,
            "type": vision_failure.failure_type.value,
            "message": vision_failure.message,
        } if vision_failure and vision_failure.failure_type.value != "image_unavailable" else None,
        "feature_contributions": {
            k: round(v, 4) for k, v in sorted(
                feature_contributions.items(), key=lambda x: x[1], reverse=True
            )[:5]  # Top 5 features
        },
        "rephoto_count_server": rephoto_count,
    }


@app.post("/api/trigger-failure")
async def trigger_failure(failure_type: str = Form(...)):
    """Trigger a simulated failure mode for demo."""
    if failure_type == "timeout":
        circuit_breaker.enable_timeout_simulation()
        return {"status": "active", "failure_type": "image_upload_timeout"}
    elif failure_type == "checkpoint_mismatch":
        circuit_breaker.enable_checkpoint_mismatch_simulation()
        return {"status": "active", "failure_type": "model_checkpoint_mismatch"}
    else:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unknown failure type: {failure_type}. Use 'timeout' or 'checkpoint_mismatch'."},
        )


@app.post("/api/reset-failure")
async def reset_failure():
    """Reset all failure simulations."""
    circuit_breaker.reset_simulations()
    return {"status": "normal_mode", "simulations": circuit_breaker.get_simulation_status()}


# ── Run ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
