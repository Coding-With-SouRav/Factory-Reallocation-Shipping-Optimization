"""Run the full offline pipeline: clean -> train/evaluate -> cluster -> recommend -> save artifacts.

    python train.py
"""
import joblib

from src.config import ARTIFACT_DIR
from src.pipeline import clean, cluster_routes, congested_combos, load_data, train_models
from src.simulate import Simulator

 
def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    raw = load_data()
    df = clean(raw)
    print(f"Rows: raw={len(raw):,} cleaned={len(df):,}")

    metrics, best, model, rf = train_models(df)
    print("\nModel comparison (20% hold-out):\n", metrics.round(3).to_string(index=False))
    print(f"\nSelected model: {best}")

    routes = cluster_routes(df)
    combos = congested_combos(df)
    sim = Simulator(df, model, rf)
    recs = sim.recommend(speed_weight=0.5)
    kpis = sim.kpis(recs, df["Product Name"].nunique())
    print("\nKPIs:", {k: round(v, 2) for k, v in kpis.items()})

    joblib.dump({"model": model, "rf": rf, "best": best}, ARTIFACT_DIR / "models.joblib")
    metrics.to_csv(ARTIFACT_DIR / "model_metrics.csv", index=False)
    routes.to_csv(ARTIFACT_DIR / "route_clusters.csv", index=False)
    combos.to_csv(ARTIFACT_DIR / "product_region_congestion.csv", index=False)
    recs.to_csv(ARTIFACT_DIR / "recommendations.csv", index=False)
    print(f"\nArtifacts written to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
