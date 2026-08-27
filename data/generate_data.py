"""
Synthetic Data Generator for the Return-Risk Scorer.

Generates realistic (but synthetic) return request data with correlated
fraud signals. Every distributional assumption is documented.

Usage:
    python data/generate_data.py --seed 42 --samples 10000

Data Card (Distributional Assumptions):
────────────────────────────────────────
This data is SYNTHETIC. The distributions below are based on:
  - Published Indian e-commerce return statistics (15-35% return rates by category)
  - Industry fraud estimates (~8% fraudulent/abusive returns)
  - India-specific COD patterns (~60-65% of e-commerce transactions are COD)
  - Observed patterns from seller forums and marketplace policies

These assumptions produce plausible data for METHODOLOGY VALIDATION only.
Absolute model performance numbers (e.g., "92% precision") measured on this
data would NOT transfer to real merchant data without recalibration.
See evaluation/results/validity_boundaries.md for full discussion.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path so we can import config before other third-party libs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # type: ignore # noqa: E402
import pandas as pd  # type: ignore # noqa: E402

from src.config import (  # type: ignore # noqa: E402
    RANDOM_SEED,
    TOTAL_SAMPLES,
    FRAUD_RATE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    PHOTO_AVAILABILITY_RATE,
    FRAUD_SUBTYPE_DIST,
    ITEM_CATEGORIES,
    RETURN_POLICY_HOURS,
    SYNTHETIC_DIR,
)


def generate_legitimate_returns(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate legitimate (non-fraud) return requests.

    Legitimate returns tend to have:
      - Moderate order values (log-normal centered around ₹1,500)
      - Older accounts (exponential with higher mean)
      - Lower return velocity
      - Reasonable delivery-to-return times (peak around 3-7 days)
    """
    data = {
        # Order value: log-normal, ₹200–₹50,000, median ~₹1,500
        "order_value": np.clip(
            rng.lognormal(mean=7.3, sigma=0.8, size=n), 200, 50000
        ),
        # Account age: exponential, mean ~400 days for legitimate users
        "account_age_days": np.clip(
            rng.exponential(scale=400, size=n), 1, 3650
        ).astype(int),
        # Prior returns: Poisson, low λ for legitimate users
        "prior_returns_count": rng.poisson(lam=2.0, size=n),
        # Prior return approval rate: Beta, skewed high (most returns approved)
        "prior_return_approval_rate": np.clip(
            rng.beta(a=8, b=2, size=n), 0, 1
        ),
        # Return velocity (last 7 days): mostly 0-1 for legitimate users
        "return_velocity_7d": rng.poisson(lam=0.8, size=n),
        # COD: ~55% for legitimate returns (slightly lower than overall COD rate)
        "is_cod": rng.binomial(1, 0.55, size=n),
        # Delivery to return: gamma distribution, peak at 3-7 days (72-168 hours)
        "delivery_to_return_hours": np.clip(
            rng.gamma(shape=3.0, scale=50.0, size=n), 6, RETURN_POLICY_HOURS
        ),
        # Item category: uniform across categories
        "item_category": rng.choice(ITEM_CATEGORIES, size=n),
        # Address-order distance: mostly small (same city), occasional large
        "address_order_distance_km": np.clip(
            rng.exponential(scale=15.0, size=n), 0, 500
        ),
        # Label
        "is_fraud": np.zeros(n, dtype=int),
        "fraud_subtype": ["none"] * n,
    }
    return pd.DataFrame(data)


def generate_fraud_returns(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate fraudulent return requests.

    Fraud returns tend to have slightly shifted distributions, but with
    HEAVY overlap with legitimate returns. The signal is weak per-feature
    and only becomes detectable through multi-feature combinations.

    Design principle: no single feature or even pair of features should
    make fraud trivially separable. LightGBM should achieve PR-AUC in
    the 0.3-0.7 range — realistic for a tabular fraud model without
    behavioral/device features.
    """
    # Assign fraud subtypes
    subtypes = rng.choice(
        list(FRAUD_SUBTYPE_DIST.keys()),
        size=n,
        p=list(FRAUD_SUBTYPE_DIST.values()),
    )

    # Order value: slightly higher mean but same distribution shape
    # Overlap is heavy — plenty of legit high-value returns exist
    order_value = np.clip(
        rng.lognormal(mean=7.5, sigma=0.85, size=n), 200, 50000
    )

    # Account age: slightly newer on average, but 40% use aged accounts
    account_age = np.clip(
        rng.exponential(scale=250, size=n), 1, 3650
    ).astype(int)
    # 40% of fraud uses aged/compromised accounts (up from 20%)
    aged_mask = rng.random(size=n) < 0.40
    account_age[aged_mask] = np.clip(
        rng.exponential(scale=450, size=aged_mask.sum()), 30, 3650
    ).astype(int)

    # Prior returns: slightly higher, but many first-timers
    prior_returns = rng.poisson(lam=3.5, size=n)
    # 45% are first-time fraudsters with clean history (up from 30%)
    first_timer_mask = rng.random(size=n) < 0.45
    prior_returns[first_timer_mask] = rng.poisson(
        lam=1.5, size=first_timer_mask.sum()
    )

    # Prior approval rate: mostly similar to legit (most returns get approved)
    approval_rate = np.clip(rng.beta(a=7, b=2.5, size=n), 0, 1)

    # Return velocity: slightly higher but overlapping
    return_velocity = rng.poisson(lam=1.5, size=n)

    # COD: slightly higher for fraud (~65% vs ~55% for legit)
    is_cod = rng.binomial(1, 0.65, size=n)

    # Delivery to return: shifted but not bimodal
    # Mix of slightly faster and slightly later, but mostly in normal range
    delivery_to_return = np.clip(
        rng.gamma(shape=2.0, scale=70.0, size=n), 6, RETURN_POLICY_HOURS
    )
    # Only 20% are extreme (very fast or very late)
    extreme_mask = rng.random(size=n) < 0.20
    n_extreme = extreme_mask.sum()
    fast_or_late = rng.random(size=n_extreme) < 0.5
    delivery_to_return[extreme_mask] = np.where(
        fast_or_late,
        np.clip(rng.exponential(scale=15.0, size=n_extreme), 2, 48),
        np.clip(rng.normal(loc=620, scale=60, size=n_extreme), 500, RETURN_POLICY_HOURS),
    )

    # Item category: slight concentration on electronics, but not extreme
    fraud_category_weights = {
        "electronics": 0.25, "fashion": 0.22, "home": 0.18,
        "beauty": 0.13, "books": 0.10, "sports": 0.12,
    }
    item_category = rng.choice(
        list(fraud_category_weights.keys()),
        size=n,
        p=list(fraud_category_weights.values()),
    )

    # Address distance: slightly larger on average, heavy overlap
    address_distance = np.clip(
        rng.exponential(scale=30.0, size=n), 0, 500
    )

    data = {
        "order_value": order_value,
        "account_age_days": account_age,
        "prior_returns_count": prior_returns,
        "prior_return_approval_rate": approval_rate,
        "return_velocity_7d": return_velocity,
        "is_cod": is_cod,
        "delivery_to_return_hours": delivery_to_return,
        "item_category": item_category,
        "address_order_distance_km": address_distance,
        "is_fraud": np.ones(n, dtype=int),
        "fraud_subtype": subtypes,
    }
    return pd.DataFrame(data)


def add_photo_availability(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Add photo availability flag. ~70% of returns have photos.
    Fraud returns are slightly MORE likely to NOT have photos for empty-box
    (they avoid evidence), but substitution fraudsters DO submit photos
    (they think the substitute will pass).
    """
    df = df.copy()
    has_photo = np.ones(len(df), dtype=int)

    for idx, row in df.iterrows():
        if row["is_fraud"] == 1 and row["fraud_subtype"] == "empty_box":
            # Empty-box fraudsters often skip photo (~60% no photo)
            has_photo[idx] = 1 if rng.random() < 0.40 else 0
        elif row["is_fraud"] == 1 and row["fraud_subtype"] == "substitution":
            # Substitution fraudsters often submit photos (they think it'll pass)
            has_photo[idx] = 1 if rng.random() < 0.80 else 0
        else:
            # Legitimate + wardrobing: ~70% have photos
            has_photo[idx] = 1 if rng.random() < PHOTO_AVAILABILITY_RATE else 0

    df["has_return_photo"] = has_photo
    return df


def add_return_ids(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add unique return request IDs."""
    df = df.copy()
    df["return_id"] = [f"RET-{i:06d}" for i in range(len(df))]
    # Shuffle to remove ordering artifacts
    df = df.sample(frac=1.0, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    return df


def split_data(
    df: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Stratified train/val/test split.
    Test set is sacred — generated once, never touched during development.
    """
    from sklearn.model_selection import train_test_split

    # First split: train+val vs test
    train_val, test = train_test_split(
        df,
        test_size=TEST_RATIO,
        stratify=df["is_fraud"],
        random_state=int(rng.integers(0, 2**31)),
    )

    # Second split: train vs val
    val_relative = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        stratify=train_val["is_fraud"],
        random_state=int(rng.integers(0, 2**31)),
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def validate_data(df: pd.DataFrame, name: str) -> None:
    """Sanity checks on the generated data."""
    fraud_rate = df["is_fraud"].mean()
    print(f"\n{'='*50}")
    print(f"  {name} — {len(df)} samples, fraud rate: {fraud_rate:.3f}")
    print(f"{'='*50}")

    # Check no single feature has too high AUC
    from sklearn.metrics import roc_auc_score

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in ("is_fraud", "has_return_photo"):
            continue
        try:
            auc = roc_auc_score(df["is_fraud"], df[col])
            # AUC can be < 0.5 if feature is inversely correlated
            auc = max(auc, 1 - auc)
            flag = " ⚠️  TOO SEPARABLE" if auc > 0.85 else ""
            print(f"  {col:35s} univariate AUC: {auc:.3f}{flag}")
        except ValueError:
            print(f"  {col:35s} univariate AUC: N/A (constant)")

    # Fraud subtype distribution
    if "fraud_subtype" in df.columns:
        fraud_only = df[df["is_fraud"] == 1]
        if len(fraud_only) > 0:
            print(f"\n  Fraud subtypes:")
            for subtype, count in fraud_only["fraud_subtype"].value_counts().items():
                print(f"    {subtype}: {count} ({count/len(fraud_only):.1%})")

    # Photo availability
    print(f"\n  Photo availability: {df['has_return_photo'].mean():.1%}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic return fraud data")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--samples", type=int, default=TOTAL_SAMPLES, help="Total samples")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    n_fraud = int(args.samples * FRAUD_RATE)
    n_legit = args.samples - n_fraud

    print(f"Generating {args.samples} samples (seed={args.seed})")
    print(f"  Legitimate: {n_legit}, Fraud: {n_fraud} ({FRAUD_RATE:.0%})")

    # Generate
    legit_df = generate_legitimate_returns(n_legit, rng)
    fraud_df = generate_fraud_returns(n_fraud, rng)

    # Combine
    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = add_photo_availability(df, rng)
    df = add_return_ids(df, rng)

    # Split
    train_df, val_df, test_df = split_data(df, rng)

    # Validate
    validate_data(train_df, "TRAIN")
    validate_data(val_df, "VALIDATION")
    validate_data(test_df, "TEST")

    # Save
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SYNTHETIC_DIR / "train.csv", index=False)
    val_df.to_csv(SYNTHETIC_DIR / "val.csv", index=False)
    test_df.to_csv(SYNTHETIC_DIR / "test.csv", index=False)

    # Also save the full dataset before split (for reference)
    df.to_csv(SYNTHETIC_DIR / "full_dataset.csv", index=False)

    print(f"\n[OK] Data saved to {SYNTHETIC_DIR}")
    print(f"  train.csv: {len(train_df)} rows")
    print(f"  val.csv:   {len(val_df)} rows")
    print(f"  test.csv:  {len(test_df)} rows")


if __name__ == "__main__":
    main()
