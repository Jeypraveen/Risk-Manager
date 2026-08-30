"""
DINOv2 embedding similarity for item substitution detection.

Compares a return photo against the original catalog photo using
DINOv2 ViT-S/14 (smallest variant, ~86MB) embeddings and cosine
similarity.

High similarity → likely the same item → legitimate return
Low similarity → different item → possible substitution fraud
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded globals to avoid importing torch at module level
_model = None
_processor = None
_device = None


def _load_model():
    """Lazy-load DINOv2 model on first use."""
    global _model, _processor, _device

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

    _load_model()

    try:
        im = Image.open(image_path)
        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
            im = im.convert('RGBA')
            bg = Image.new('RGB', im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[3])
            image = bg
        else:
            image = im.convert("RGB")
            
        inputs = _processor(images=image, return_tensors="pt").to(_device)

        with torch.no_grad():
            outputs = _model(**inputs)

        # CLS token is the first token in last_hidden_state
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    except Exception as e:
        logger.error(f"Failed to extract embedding from {image_path}: {e}")
        return None


def compute_similarity(
    catalog_image_path: str,
    return_image_path: str,
) -> Optional[float]:
    """
    Compute cosine similarity between catalog and return item photos.

    Args:
        catalog_image_path: Path to the original product catalog image
        return_image_path: Path to the customer's return photo

    Returns:
        float in [0, 1] — cosine similarity (higher = more similar)
        None if either embedding failed
    """
    catalog_emb = extract_embedding(catalog_image_path)
    return_emb = extract_embedding(return_image_path)

    if catalog_emb is None or return_emb is None:
        return None

    # Cosine similarity
    dot = np.dot(catalog_emb, return_emb)
    norm = np.linalg.norm(catalog_emb) * np.linalg.norm(return_emb)
    if norm == 0:
        return None

    similarity = float(dot / norm)
    # Clamp to [0, 1] — cosine can technically be negative
    return max(0.0, min(1.0, similarity))


def compute_similarity_from_embeddings(
    catalog_embedding: np.ndarray,
    return_embedding: np.ndarray,
) -> float:
    """Compute cosine similarity from pre-computed embeddings."""
    dot = np.dot(catalog_embedding, return_embedding)
    norm = np.linalg.norm(catalog_embedding) * np.linalg.norm(return_embedding)
    if norm == 0:
        return 0.0
    return float(max(0.0, min(1.0, dot / norm)))
