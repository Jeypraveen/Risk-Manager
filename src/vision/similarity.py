"""
DINOv2 embedding similarity for item substitution detection.

Compares a return photo against the original catalog photo using
DINOv2 ViT-S/14 (smallest variant, ~86MB) embeddings and cosine
similarity.

High similarity → likely the same item → legitimate return
Low similarity → different item → possible substitution fraud
"""

import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid importing torch at module level
_model = None
_processor = None
_device = None
_model_lock = threading.Lock()


def _load_model():
    global _model, _processor, _device

    if _model is not None:
        return

    with _model_lock:
        if _model is not None:
            return

        import torch
        from transformers import AutoImageProcessor, AutoModel

        from src.config import DINOV2_MODEL_NAME

        logger.info(f"Loading DINOv2 model: {DINOV2_MODEL_NAME}")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_NAME)
        _model = AutoModel.from_pretrained(DINOV2_MODEL_NAME).to(_device)
        _model.eval()
        logger.info(f"DINOv2 loaded on {_device}")


def extract_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract DINOv2 CLS token embedding from an image.

    Args:
        image_path: Path to the image file

    Returns:
        np.ndarray of shape (embedding_dim,) or None if failed
    """
    import torch

    try:
        _load_model()
        im = Image.open(image_path)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            image = bg
        else:
            image = im.convert("RGB")

        assert _processor is not None
        inputs = _processor(images=image, return_tensors="pt").to(_device)

        with torch.no_grad():
            assert _model is not None
            outputs = _model(**inputs)

        # CLS token is the first token in last_hidden_state
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    except Exception as e:
        logger.error(f"Failed to extract embedding from {image_path}: {e}")
        return None


# Empirically measured operating band of raw DINOv2 CLS cosine on this
# project's staged image set, after mapping [-1,1] -> [0,1]:
#   genuine matches   ~0.59 - 0.79
#   mismatched items  ~0.47 - 0.53
# Both sit above 0.45, so an unstretched score leaves mismatches at ~0.50 —
# indistinguishable from the neutral no-vision default. Stretching this band
# across [0,1] pushes mismatches well below 0.5 so a wrong-item photo becomes
# active negative evidence rather than a no-op.
SIM_BAND_LOW = 0.45
SIM_BAND_HIGH = 0.80


def _rescale(cosine: float) -> float:
    """Map raw cosine [-1,1] into [0,1], stretched over the measured band."""
    unit = (cosine + 1.0) / 2.0
    stretched = (unit - SIM_BAND_LOW) / (SIM_BAND_HIGH - SIM_BAND_LOW)
    return max(0.0, min(1.0, stretched))


def compute_similarity(
    catalog_image_path: str,
    return_image_path: str,
    return_embeddings: bool = False,
) -> Optional[float] | tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute cosine similarity between catalog and return item photos.

    Args:
        catalog_image_path: Path to the original product catalog image
        return_image_path: Path to the customer's return photo
        return_embeddings: If True, also return the raw embeddings

    Returns:
        float in [0, 1] — rescaled similarity (higher = more similar)
        None if either embedding failed
        If return_embeddings=True, returns (similarity, catalog_emb, return_emb)
    """
    catalog_emb = extract_embedding(catalog_image_path)
    return_emb = extract_embedding(return_image_path)

    if catalog_emb is None or return_emb is None:
        return (None, catalog_emb, return_emb) if return_embeddings else None

    # Cosine similarity
    dot = np.dot(catalog_emb, return_emb)
    norm = np.linalg.norm(catalog_emb) * np.linalg.norm(return_emb)
    if norm == 0:
        return (None, catalog_emb, return_emb) if return_embeddings else None

    similarity = _rescale(float(dot / norm))

    if return_embeddings:
        return similarity, catalog_emb, return_emb
    return similarity


def compute_similarity_from_embeddings(
    catalog_embedding: np.ndarray,
    return_embedding: np.ndarray,
) -> float:
    """Compute cosine similarity from pre-computed embeddings."""
    dot = np.dot(catalog_embedding, return_embedding)
    norm = np.linalg.norm(catalog_embedding) * np.linalg.norm(return_embedding)
    if norm == 0:
        return 0.0
    return _rescale(float(dot / norm))