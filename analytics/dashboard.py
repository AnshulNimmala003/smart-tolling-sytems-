"""Toll operations & fraud dashboard.

Run:
    streamlit run analytics/dashboard.py
"""

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fraud import detect, evaluate  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "transactions.csv"

# Validated categorical palette (dataviz reference, light mode, fixed order).
CLASS_COLORS = {
    "car": "#2a78d6",
    "motorcycle": "#eb6834",
    "truck": "#1baf7a",
    "bus": "#eda100",
}
SINGLE = "#2a78d6"  # sequential/base hue
LEAK = "#eb6834"
FRAUD_BAR = "#e34948"

st.set_page_config(page_title="Smart Toll — Fraud Analytics", page_icon="🛣️", layout="wide")


@st.cache_data
def load() -> pd.DataFrame:
    if not DATA.exists():
        st.error("No data — run `python analytics/simulate.py` first.")
        st.stop()
    return detect(pd.read_csv(DATA, parse_dates=["timestamp"]))


df = load()

st.title("🛣️ Smart Toll — Operations & Fraud Analytics")
st.caption(
    "Simulated month of ANPR toll transactions across three plazas. "
    "Rule engine + Isolation Forest flag fraud; detector is scored against "
    "the simulator's injected ground truth."
)

# ---- filters (one row above the charts) ---------------------------------
fcol1, fcol2 = st.columns([2, 1])
with fcol1:
    dmin, dmax = df["timestamp"].dt.date.min(), df["timestamp"].dt.date.max()
    drange = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
with fcol2:
    plazas = st.multiselect("Plazas", sorted(df["plaza"].unique()), default=sorted(df["plaza"].unique()))

if len(drange) == 2:
    df = df[(df["timestamp"].dt.date >= drange[0]) & (df["timestamp"].dt.date <= drange[1])]
df = df[df["plaza"].isin(plazas)]

# ---- KPI tiles -----------------------------------------------------------
scores = evaluate(df) if df["injected_fraud"].ne("none").any() else None
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Transactions", f"{len(df):,}")
k2.metric("Revenue", f"₹{int(df['amount_charged'][df['paid']].sum()):,}")
k3.metric("Flagged", f"{int(df['flagged'].sum()):,}")
k4.metric("Est. leakage", f"₹{int(df['leakage'].sum()):,}")
k5.metric(
    "Detector P / R",
    f"{scores['precision']:.0%} / {scores['recall']:.0%}" if scores else "—",
    help="Precision / recall of the fraud flags vs the simulator's injected ground truth",
)

st.divider()

# ---- charts --------------------------------------------------------------
daily = (
    df[df["paid"]]
    .groupby(df["timestamp"].dt.date)["amount_charged"].sum()
    .reset_index().rename(columns={"timestamp": "date"})
)
leak_daily = (
    df.groupby(df["timestamp"].dt.date)["leakage"].sum()
    .reset_index().rename(columns={"timestamp": "date"})
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("Daily revenue")
    st.altair_chart(
        alt.Chart(daily)
        .mark_line(strokeWidth=2, color=SINGLE)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("amount_charged:Q", title="₹ collected"),
            tooltip=["date:T", alt.Tooltip("amount_charged:Q", title="₹", format=",")],
        )
        .properties(height=260),
        use_container_width=True,
    )
with c2:
    st.subheader("Daily estimated leakage")
    st.altair_chart(
        alt.Chart(leak_daily)
        .mark_line(strokeWidth=2, color=LEAK)
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("leakage:Q", title="₹ leaked"),
            tooltip=["date:T", alt.Tooltip("leakage:Q", title="₹", format=",")],
        )
        .properties(height=260),
        use_container_width=True,
    )

c3, c4 = st.columns(2)
with c3:
    st.subheader("Traffic by vehicle class")
    by_class = df["vehicle_class"].value_counts().reset_index()
    st.altair_chart(
        alt.Chart(by_class)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("vehicle_class:N", title=None, sort="-y"),
            y=alt.Y("count:Q", title="crossings"),
            color=alt.Color(
                "vehicle_class:N",
                scale=alt.Scale(
                    domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())
                ),
                legend=None,  # x-axis labels name each bar directly
            ),
            tooltip=["vehicle_class:N", alt.Tooltip("count:Q", title="crossings", format=",")],
        )
        .properties(height=260),
        use_container_width=True,
    )
with c4:
    st.subheader("Fraud flags by type")
    reasons = (
        df.loc[df["flag_reasons"] != "—", "flag_reasons"]
        .str.split(", ").explode().value_counts().reset_index()
    )
    st.altair_chart(
        alt.Chart(reasons)
        .mark_bar(color=FRAUD_BAR, cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title="flags"),
            y=alt.Y("flag_reasons:N", title=None, sort="-x"),
            tooltip=["flag_reasons:N", alt.Tooltip("count:Q", title="flags", format=",")],
        )
        .properties(height=260),
        use_container_width=True,
    )

# ---- flagged transactions table -----------------------------------------
st.subheader("Flagged transactions")
flagged = df[df["flagged"] | df["flag_anomaly"]].sort_values("timestamp", ascending=False)
st.dataframe(
    flagged[
        ["timestamp", "plaza", "plate", "vehicle_class", "billed_class",
         "amount_charged", "paid", "flag_reasons", "leakage"]
    ],
    use_container_width=True,
    height=340,
)
st.caption(
    f"{len(flagged):,} of {len(df):,} transactions flagged. "
    "Rules: impossible travel (cloned plates), blacklist, fee mismatch, unpaid. "
    "Isolation Forest adds statistical outliers no rule anticipated."
)
