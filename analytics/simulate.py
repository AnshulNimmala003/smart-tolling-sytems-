"""Toll transaction simulator with labeled fraud injection.

Generates a month of realistic transactions across three plazas, then injects
four fraud patterns and RECORDS the ground truth (`injected_fraud` column) so
the detector in `fraud.py` can be scored with precision/recall — not vibes.

Usage:
    python analytics/simulate.py            # writes data/transactions.csv
"""

from __future__ import annotations

import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_PLATES = 1800
N_DAYS = 30
START = datetime(2026, 7, 1)

OUT = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"

# Plaza layout along one highway corridor (km from origin).
PLAZAS = {"PLAZA-A": 0, "PLAZA-B": 55, "PLAZA-C": 130}
MAX_PLAUSIBLE_KMH = 110  # faster than this between sightings = impossible travel

VEHICLE_MIX = {"car": 0.62, "motorcycle": 0.16, "truck": 0.15, "bus": 0.07}
RATES = {"motorcycle": 30, "car": 65, "bus": 225, "truck": 225}

BLACKLIST = ["MH12DE1433", "STOLEN1"]


def _plate(rng: random.Random) -> str:
    state = rng.choice(["MH", "TS", "KA", "AP", "TN", "DL"])
    return (
        f"{state}{rng.randint(1, 39):02d}"
        f"{''.join(rng.choices(string.ascii_uppercase, k=2))}"
        f"{rng.randint(1, 9999):04d}"
    )


def simulate() -> pd.DataFrame:
    rng = random.Random(SEED)
    np_rng = np.random.default_rng(SEED)

    plates = [_plate(rng) for _ in range(N_PLATES)]
    plate_class = {
        p: rng.choices(list(VEHICLE_MIX), weights=list(VEHICLE_MIX.values()))[0]
        for p in plates
    }

    rows = []
    for day in range(N_DAYS):
        # Diurnal traffic curve: morning + evening peaks.
        n_today = int(np_rng.normal(420, 40))
        hours = np.clip(
            np.concatenate(
                [np_rng.normal(9, 2.2, n_today // 2), np_rng.normal(18, 2.6, n_today - n_today // 2)]
            ),
            0, 23.99,
        )
        for h in hours:
            plate = rng.choice(plates)
            cls = plate_class[plate]
            ts = START + timedelta(days=day, hours=float(h), minutes=rng.random() * 60 % 60)
            rows.append(
                {
                    "timestamp": ts,
                    "plaza": rng.choice(list(PLAZAS)),
                    "plate": plate,
                    "vehicle_class": cls,
                    "billed_class": cls,
                    "amount_charged": RATES[cls],
                    "paid": True,
                    "injected_fraud": "none",
                }
            )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    # ---- fraud injection -------------------------------------------------
    idx = np_rng.permutation(len(df))
    cursor = 0

    def take(n):
        nonlocal cursor
        sel = idx[cursor : cursor + n]
        cursor += n
        return sel

    # 1) Cloned plates: duplicate a sighting at a far plaza minutes later.
    clones = []
    for i in take(60):
        src = df.iloc[i]
        far = [p for p in PLAZAS if abs(PLAZAS[p] - PLAZAS[src["plaza"]]) >= 100]
        if not far:
            continue
        clone = src.copy()
        clone["plaza"] = rng.choice(far)
        clone["timestamp"] = src["timestamp"] + timedelta(minutes=rng.randint(10, 35))
        clone["injected_fraud"] = "cloned_plate"
        clones.append(clone)
    df = pd.concat([df, pd.DataFrame(clones)], ignore_index=True)

    # 2) Class fraud: trucks billed as cars (revenue leakage).
    trucks = df[(df["vehicle_class"] == "truck") & (df["injected_fraud"] == "none")]
    for i in np_rng.choice(trucks.index, size=min(180, len(trucks)), replace=False):
        df.loc[i, ["billed_class", "amount_charged", "injected_fraud"]] = [
            "car", RATES["car"], "class_fraud",
        ]

    # 3) Blacklisted vehicles slipping through.
    for i in take(25):
        df.loc[df.index[i], ["plate", "injected_fraud"]] = [rng.choice(BLACKLIST), "blacklisted"]

    # 4) Unpaid passages (barrier tailgating / failed capture).
    for i in take(140):
        j = df.index[i]
        if df.loc[j, "injected_fraud"] == "none":
            df.loc[j, ["paid", "injected_fraud"]] = [False, "unpaid"]

    return df.sort_values("timestamp").reset_index(drop=True)


if __name__ == "__main__":
    df = simulate()
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} transactions -> {OUT}")
    print(df["injected_fraud"].value_counts().to_string())
