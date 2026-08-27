# Architecture: Return-Risk Scorer

This document explains the system architecture for the Razorpay AI Buildathon 2026.

## System Overview

The Return-Risk Scorer evaluates e-commerce return requests using a **Late-Fusion Meta-Learner** architecture. It combines tabular risk scoring (metadata like account age, return velocity, order value) with a vision pipeline (customer-provided return photos) to produce a unified **Trust Score**.

The system then uses a three-way decision router to decide whether to:
1.  `AUTO_APPROVE`: Instant refund (high trust).
2.  `NUDGE`: Request a photo or offer store credit (medium trust).
3.  `MANUAL_REVIEW`: Route to a human investigator (low trust).

## Core Components

### 1. Tabular Scorer (LightGBM)
*   **Input:** Metadata (Order value, account age, prior returns, COD status, etc.)
*   **Process:** Evaluates historical and behavioral risk using a gradient-boosted tree.
*   **Output:** Base tabular risk score.

### 2. Vision Pipeline (DINOv2 + SAM2/Heuristic)
*   **Input:** Customer return photo + Catalog reference photo.
*   **DINOv2:** Extracts structural embeddings (ViT-S/14) and computes cosine similarity to detect **substitution fraud**.
*   **SAM2 / Edge-Density Heuristic:** Analyzes the image to detect **empty-box fraud** by evaluating mask coverage or edge density.
*   **Circuit Breaker:** Wraps the vision pipeline. If the model fails or times out, it gracefully degrades to a `modality_confidence` of `0.0` rather than crashing the system.

### 3. Late-Fusion Meta-Learner (Logistic Regression)
*   **Input:** Tabular score, vision signals (similarity, empty box), and `modality_confidence`.
*   **Process:** Fuses the inputs. Late fusion is specifically chosen so that if an image is missing or the vision pipeline fails, the system can still make a decision based on tabular data.
*   **Output:** The final **Return Trust Score** [0-1].

### 4. Decision Router & Audit Trail
*   **Router:** Takes the trust score and routes it based on configurable thresholds. Enforces rules like maximum re-photo requests (caps at 2 before forcing manual review) and penalizes accounts that frequently accept store credit.
*   **Audit Trail:** An SQLite-backed logger records every decision, including the exact inputs, intermediate scores, circuit breaker failures, and the exact config thresholds at the time of the decision.

## Why Late Fusion?
In e-commerce, images are often missing, blurry, or fail to upload. An early-fusion model (e.g., a massive multi-modal transformer) would struggle or fail completely if a modality is dropped. By keeping the tabular and vision pipelines separate until the final step, we ensure **graceful degradation** — the system always provides a score, even if it's just based on tabular data.
