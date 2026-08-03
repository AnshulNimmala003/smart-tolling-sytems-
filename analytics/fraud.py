"""Fraud detection over toll transactions: rules + Isolation Forest.

Rules catch what we can define (impossible travel, blacklist, fee mismatch,
unpaid). The Isolation Forest catches statistical outliers no rule anticipated.
`evaluate()` scores the detector against the simulator's injected ground truth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from simulate import BLACKLIST, MAX_PLAUSIBLE_KMH, PLAZAS, RATES

ANOMALY_CONTAMINATION = 0.02


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["plate", "timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Impossible travel: implied speed between consecutive sightings of a plate.
    df["prev_ts"] = df.groupby("plate")["timestamp"].shift()
    df["prev_plaza"] = df.groupby("plate")["plaza"].shift()
    dist = (df["plaza"].map(PLAZAS) - df["prev_plaza"].map(PLAZAS)).abs()
    hours = (df["timestamp"] - df["prev_ts"]).dt.total_seconds() / 3600
    speed = dist / hours.replace(0, np.nan)
    df["flag_impossible_travel"] = (dist > 0) & (speed > MAX_PLAUSIBLE_KMH)

    df["flag_blacklisted"] = df["plate"].isin(BLACKLIST)
    df["flag_fee_mismatch"] = df["amount_charged"] != df["vehicle_class"].map(RATES)
    df["flag_unpaid"] = ~df["paid"]
    return df.drop(columns=["prev_ts", "prev_plaza"])


def apply_anomaly_model(df: pd.DataFrame) -> pd.DataFrame:
    """Isolation Forest on engineered behavioral features."""
    feats = pd.DataFrame(
        {
            "hour": df["timestamp"].dt.hour,
            "amount": df["amount_charged"],
            "plate_daily_crossings": df.groupby(
                [df["plate"], df["timestamp"].dt.date]
            )["plate"].transform("count"),
            "unpaid": (~df["paid"]).astype(int),
        }
    )
    model = IsolationForest(
        n_estimators=200, contamination=ANOMALY_CONTAMINATION, random_state=0
    )
    df = df.copy()
    df["anomaly_score"] = -model.fit_predict(feats.fillna(0))  # 1 = anomalous
    df["flag_anomaly"] = df["anomaly_score"] > 0
    return df


def detect(df: pd.DataFrame) -> pd.DataFrame:
    df = apply_anomaly_model(apply_rules(df))
    rule_cols = [c for c in df.columns if c.startswith("flag_") and c != "flag_anomaly"]
    df["flagged"] = df[rule_cols].any(axis=1)
    df["flag_reasons"] = df[rule_cols + ["flag_anomaly"]].apply(
        lambda r: ", ".join(c.removeprefix("flag_") for c in r.index if r[c]) or "—",
        axis=1,
    )
    # Estimated leakage: unpaid full fare + undercharged difference.
    df["leakage"] = np.where(
        ~df["paid"],
        df["vehicle_class"].map(RATES),
        df["vehicle_class"].map(RATES) - df["amount_charged"],
    ).clip(min=0)
    return df.sort_values("timestamp").reset_index(drop=True)


def evaluate(df: pd.DataFrame) -> dict:
    """Precision/recall of `flagged` vs the simulator's injected ground truth."""
    truth = df["injected_fraud"] != "none"
    pred = df["flagged"]
    tp = int((truth & pred).sum())
    precision = tp / max(1, int(pred.sum()))
    recall = tp / max(1, int(truth.sum()))
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "flagged": int(pred.sum()),
        "true_fraud": int(truth.sum()),
    }


if __name__ == "__main__":
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"
    df = detect(pd.read_csv(src, parse_dates=["timestamp"]))
    print(df["flag_reasons"].value_counts().head(8).to_string())
    print("\nDetector vs injected ground truth:", evaluate(df))
    print(f"Estimated leakage: Rs.{int(df['leakage'].sum())}")
