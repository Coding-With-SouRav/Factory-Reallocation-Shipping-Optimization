import numpy as np
import pandas as pd

from .config import *
from .pipeline import CAT, NUM, TARGET, factory_distance, freight_cost

ALL_FACTORIES = list(FACTORIES)


class Simulator:
    def __init__(self, df: pd.DataFrame, model, rf_model, max_unc_rows=300):
        self.df, self.model, self.rf = df, model, rf_model
        self.max_unc_rows = max_unc_rows
        self.support = df.groupby(["Factory", "Region"]).size().to_dict()
        self.max_support = max(self.support.values())

    # ---- helpers -------------------------------------------------------
    def _subset(self, product, regions=None, ship_modes=None):
        d = self.df[self.df["Product Name"] == product]
        if regions:
            d = d[d["Region"].isin(regions)]
        if ship_modes:
            d = d[d["Ship Mode"].isin(ship_modes)]
        return d

    def _with_factory(self, d, factory):
        x = d.copy()
        x["Factory"] = factory
        x["distance_km"] = factory_distance(x["Factory"], x["State/Province"])
        return x

    def _uncertainty(self, x):
        """Mean std (days) across Random-Forest trees -> prediction uncertainty."""
        xs = x if len(x) <= self.max_unc_rows else x.sample(self.max_unc_rows, random_state=RANDOM_STATE)
        Xt = self.rf.named_steps["prep"].transform(xs[CAT + NUM])
        preds = np.stack([t.predict(Xt) for t in self.rf.named_steps["model"].estimators_])
        return float(preds.std(axis=0).mean())

    # ---- core: one product across all factories --------------------------
    def simulate_product(self, product, regions=None, ship_modes=None) -> pd.DataFrame:
        d = self._subset(product, regions, ship_modes)
        if d.empty:
            return pd.DataFrame()
        current = d["Factory"].iloc[0]
        base_x = self._with_factory(d, current)
        base_lead = self.model.predict(base_x[CAT + NUM])
        base_freight = base_x["freight_cost"].sum()
        base_gp = d["Gross Profit"].sum()
        base_profit = base_gp - base_freight
        base_unc = self._uncertainty(base_x)
        reg_w = d["Region"].value_counts()
        rows = []
        for f in ALL_FACTORIES:
            x = self._with_factory(d, f)
            lead = self.model.predict(x[CAT + NUM])
            fr = freight_cost(x["distance_km"], x["Units"], x["Ship Mode"]).sum()
            profit = d["Gross Profit"].sum() - fr
            unc = self._uncertainty(x)
            sup = sum(self.support.get((f, r), 0) * w for r, w in reg_w.items()) / reg_w.sum()
            support_score = min(1.0, np.log1p(sup) / np.log1p(0.25 * self.max_support))
            unc_score = float(np.clip(1 - unc / 150.0, 0, 1))
            conf = 100 * (0.5 * support_score + 0.5 * unc_score)
            rows.append({
                "Product": product, "Current Factory": current, "Candidate Factory": f,
                "is_current": f == current, "Orders": len(d),
                "Avg Distance (km)": x["distance_km"].mean(),
                "Pred Lead (days)": lead.mean(), "Baseline Lead (days)": base_lead.mean(),
                "Lead Time Change (days)": lead.mean() - base_lead.mean(),
                "Lead Time Reduction (%)": 100 * (base_lead.mean() - lead.mean()) / base_lead.mean(),
                "Est. Freight Cost": fr, "Freight Saving": base_freight - fr,
                "Profit Impact ($)": profit - base_profit,
                "Profit Impact (%)": 100 * (profit - base_profit) / base_gp,
                "Confidence Score": conf, "Pred Uncertainty (days)": unc,
                "Support Orders": sup,
            })
        return pd.DataFrame(rows)

    # ---- optimisation across products -------------------------------------
    def recommend(self, speed_weight=0.5, regions=None, ship_modes=None, products=None, top_n=None):
        """Rank (product, candidate-factory) reassignments.

        score = w*speed_norm + (1-w)*profit_norm - 0.2*(1-confidence/100)
        where speed/profit are min-max normalised across all candidates in the run.
        """
        products = products or sorted(self.df["Product Name"].unique())
        sims = [self.simulate_product(p, regions, ship_modes) for p in products]
        sims = [s for s in sims if not s.empty]
        if not sims:
            return pd.DataFrame()
        all_ = pd.concat(sims, ignore_index=True)

        def mm(s):
            return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0
        all_["Speed Score"] = mm(all_["Lead Time Reduction (%)"])
        all_["Profit Score"] = mm(all_["Profit Impact (%)"])
        all_["Risk Penalty"] = 0.2 * (1 - all_["Confidence Score"] / 100)
        all_["Priority Score"] = (speed_weight * all_["Speed Score"] + (1 - speed_weight) * all_["Profit Score"]
                                  - all_["Risk Penalty"])
        all_["High Risk"] = (all_["Profit Impact (%)"] < -HIGH_RISK_MARGIN_DROP_PCT) | (all_["Confidence Score"] < 40)
        self.all_scenarios = all_
        recs = all_[~all_["is_current"]].copy()
        recs = recs[(recs["Lead Time Reduction (%)"] > 0) | (recs["Profit Impact ($)"] > 0)]
        recs = recs.sort_values(["Product", "Priority Score"], ascending=[True, False])
        if top_n:
            recs = recs.groupby("Product").head(top_n)
        return recs.sort_values("Priority Score", ascending=False).reset_index(drop=True)

    def kpis(self, recs: pd.DataFrame, n_products: int) -> dict:
        if recs.empty:
            return {"Lead Time Reduction (%)": 0.0, "Profit Impact Stability (%)": 100.0,
                    "Scenario Confidence Score": 0.0, "Recommendation Coverage (%)": 0.0}
        best = recs.sort_values("Priority Score", ascending=False).groupby("Product").head(1)
        w = best["Orders"]
        return {
            "Lead Time Reduction (%)": float(np.average(best["Lead Time Reduction (%)"], weights=w)),
            "Profit Impact Stability (%)": 100 * float((best["Profit Impact (%)"] >= -HIGH_RISK_MARGIN_DROP_PCT).mean()),
            "Scenario Confidence Score": float(np.average(best["Confidence Score"], weights=w)),
            "Recommendation Coverage (%)": 100 * best["Product"].nunique() / n_products,
        }
