"""Streamlit dashboard: Factory Reallocation & Shipping Optimization for Nassau Candy.

    streamlit run app.py
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import FACTORIES, HIGH_RISK_MARGIN_DROP_PCT
from src.pipeline import (TARGET, clean, cluster_routes, congested_combos, load_data, train_models)
from src.simulate import Simulator

st.set_page_config(page_title="Nassau Candy – Factory Optimizer", page_icon="🍬", layout="wide")


@st.cache_resource(show_spinner="Loading data and training models…")
def bootstrap():
    df = clean(load_data())
    metrics, best, model, rf = train_models(df)
    return df, metrics, best, Simulator(df, model, rf), cluster_routes(df), congested_combos(df)


df, metrics, best_model, sim, routes, combos = bootstrap()


@st.cache_data(show_spinner="Simulating scenarios…")
def run_reco(speed_w, regions, modes):
    r = sim.recommend(speed_w, list(regions) or None, list(modes) or None)
    return r, sim.all_scenarios.copy(), sim.kpis(r, df["Product Name"].nunique())


# ---------------- Sidebar ----------------
st.sidebar.title("🍬 Controls")
products = sorted(df["Product Name"].unique())
product = st.sidebar.selectbox("Product", products)
regions = st.sidebar.multiselect("Region (empty = all)", sorted(df["Region"].unique()))
modes = st.sidebar.multiselect("Ship mode (empty = all)", sorted(df["Ship Mode"].unique()))
speed_w = st.sidebar.slider("Optimization priority", 0, 100, 50, help="0 = maximise profit, 100 = maximise speed") / 100
st.sidebar.caption(f"Speed weight {speed_w:.0%} · Profit weight {1 - speed_w:.0%}")
top_n = st.sidebar.number_input("Top-N per product", 1, 4, 3)

recs, scen, kpis = run_reco(speed_w, tuple(regions), tuple(modes))

st.title("Factory Reallocation & Shipping Optimization")
st.caption("Predict lead time under alternative factory assignments and balance speed vs profit.")

c = st.columns(4)
c[0].metric("Lead Time Reduction", f"{kpis['Lead Time Reduction (%)']:.2f}%", help="Order-weighted, best option per product")
c[1].metric("Profit Impact Stability", f"{kpis['Profit Impact Stability (%)']:.0f}%", help=f"Share of top picks with profit impact above -{HIGH_RISK_MARGIN_DROP_PCT}%")
c[2].metric("Scenario Confidence", f"{kpis['Scenario Confidence Score']:.0f}/100")
c[3].metric("Recommendation Coverage", f"{kpis['Recommendation Coverage (%)']:.0f}%")

t1, t2, t3, t4, t5 = st.tabs(["🏭 Factory Simulator", "🔀 What-If Analysis", "🏆 Recommendations",
                              "⚠️ Risk & Impact", "📊 Model & Clusters"])

# ---- 1. Simulator
with t1:
    s = scen[scen["Product"] == product]
    if s.empty:
        st.warning("No orders for this product under the selected filters.")
    else:
        cur = s["Current Factory"].iloc[0]
        st.subheader(f"{product} — currently made at **{cur}**")
        fig = px.bar(s, x="Candidate Factory", y="Pred Lead (days)", color="is_current",
                     color_discrete_map={True: "#636EFA", False: "#00CC96"}, text_auto=".1f",
                     labels={"is_current": "Current factory"})
        lo, hi = s["Pred Lead (days)"].min(), s["Pred Lead (days)"].max()
        fig.update_yaxes(range=[lo - (hi - lo) * 2 - 1, hi + (hi - lo) * 2 + 1])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(s[["Candidate Factory", "is_current", "Avg Distance (km)", "Pred Lead (days)",
                        "Lead Time Change (days)", "Freight Saving", "Profit Impact (%)", "Confidence Score"]]
                     .round(2), hide_index=True, width="stretch")
        st.caption("Note: y-axis is zoomed — absolute lead times are long and differences between factories are small.")

# ---- 2. What-if
with t2:
    s = scen[scen["Product"] == product]
    if not s.empty:
        opts = [f for f in s["Candidate Factory"] if f != s["Current Factory"].iloc[0]]
        alt = st.selectbox("Compare current assignment with", opts)
        a, b = s[s.is_current].iloc[0], s[s["Candidate Factory"] == alt].iloc[0]
        cmp = pd.DataFrame({"Metric": ["Pred lead time (days)", "Avg distance (km)", "Est. freight cost ($)"],
                            "Current": [a["Pred Lead (days)"], a["Avg Distance (km)"], a["Est. Freight Cost"]],
                            alt: [b["Pred Lead (days)"], b["Avg Distance (km)"], b["Est. Freight Cost"]]})
        st.dataframe(cmp.round(2), hide_index=True, width="stretch")
        m1, m2, m3 = st.columns(3)
        m1.metric("Lead time change", f"{b['Lead Time Change (days)']:+.2f} days", f"{b['Lead Time Reduction (%)']:.2f}% faster", delta_color="normal")
        m2.metric("Profit impact", f"${b['Profit Impact ($)']:+,.0f}", f"{b['Profit Impact (%)']:+.1f}%")
        m3.metric("Confidence", f"{b['Confidence Score']:.0f}/100")
        st.plotly_chart(px.bar(s, x="Candidate Factory", y="Lead Time Reduction (%)", color="Profit Impact (%)",
                               color_continuous_scale="RdYlGn", title="Lead-time improvement vs current (colour = profit impact %)"),
                        width="stretch")

# ---- 3. Recommendations
with t3:
    if recs.empty:
        st.info("No beneficial reassignments under the current filters.")
    else:
        view = recs.groupby("Product").head(int(top_n)).reset_index(drop=True)
        view.insert(0, "Rank", range(1, len(view) + 1))
        cols = ["Rank", "Product", "Current Factory", "Candidate Factory", "Lead Time Change (days)",
                "Lead Time Reduction (%)", "Freight Saving", "Profit Impact (%)", "Confidence Score", "Priority Score", "High Risk"]
        st.dataframe(view[cols].round(3), hide_index=True, width="stretch")
        st.download_button("Download recommendations (CSV)", view[cols].to_csv(index=False), "recommendations.csv")
        top = view.groupby("Product").head(1)
        st.plotly_chart(px.bar(top, x="Product", y="Priority Score", color="Candidate Factory",
                               title="Best reassignment per product"), width="stretch")

# ---- 4. Risk
with t4:
    risky = scen[(~scen["is_current"]) & scen["High Risk"]]
    st.subheader("High-risk reassignment warnings")
    if risky.empty:
        st.success("No high-risk scenarios under current filters.")
    else:
        st.warning(f"{len(risky)} scenario(s) flagged: profit impact < -{HIGH_RISK_MARGIN_DROP_PCT}% or confidence < 40.")
        st.dataframe(risky[["Product", "Current Factory", "Candidate Factory", "Profit Impact (%)", "Confidence Score",
                            "Pred Uncertainty (days)"]].round(2), hide_index=True, width="stretch")
    st.plotly_chart(px.scatter(scen[~scen["is_current"]], x="Confidence Score", y="Profit Impact (%)",
                               color="Candidate Factory", hover_data=["Product"],
                               title="Risk map: profit impact vs confidence"), width="stretch")
    st.caption("Freight cost is an assumption-based proxy (see src/config.py) — the dataset has no freight cost field.")

# ---- 5. Model & clusters
with t5:
    st.subheader("Model comparison (hold-out)")
    st.dataframe(metrics.round(3), hide_index=True, width="stretch")
    st.caption(f"Selected: **{best_model}** (accuracy + interpretability rule).")
    st.subheader("Route clusters (factory → state)")
    fig = px.scatter(routes, x="distance_km", y="avg_lead", color="route_cluster", size="orders",
                     hover_data=["Factory", "State/Province"])
    st.plotly_chart(fig, width="stretch")
    st.subheader("Congested product–region combinations")
    st.dataframe(combos[combos.congested].round(1), hide_index=True, width="stretch")
    st.subheader("Factory network")
    fd = pd.DataFrame([(k, *v) for k, v in FACTORIES.items()], columns=["Factory", "lat", "lon"])
    st.plotly_chart(px.scatter_geo(fd, lat="lat", lon="lon", text="Factory", scope="north america"), width="stretch")
