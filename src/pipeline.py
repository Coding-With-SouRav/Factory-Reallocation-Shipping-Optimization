import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import *

CAT = ["Product Name", "Region", "Ship Mode"]
NUM = ["distance_km", "Units", "unit_price", "order_year", "order_month"]
TARGET = "lead_time"


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def factory_distance(factory, state):
    """km between factory and customer state centroid (vectorised over Series)."""
    flat = factory.map(lambda f: FACTORIES[f][0]).astype(float)
    flon = factory.map(lambda f: FACTORIES[f][1]).astype(float)
    slat = state.map(lambda s: STATE_COORDS[s][0]).astype(float)
    slon = state.map(lambda s: STATE_COORDS[s][1]).astype(float)
    return haversine(flat, flon, slat, slon)


def freight_cost(distance, units, ship_mode):
    return FREIGHT_COST_PER_UNIT_KM * distance * units * ship_mode.map(SHIP_MODE_COST_MULT)


def load_data(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Product Name"] = df["Product Name"].str.strip()
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df[TARGET] = (df["Ship Date"] - df["Order Date"]).dt.days
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)
    df["distance_km"] = factory_distance(df["Factory"], df["State/Province"])
    df["unit_price"] = df["Sales"] / df["Units"]
    df["order_year"] = df["Order Date"].dt.year
    df["order_month"] = df["Order Date"].dt.month
    df["margin_pct"] = df["Gross Profit"] / df["Sales"] * 100
    df["freight_cost"] = freight_cost(df["distance_km"], df["Units"], df["Ship Mode"])
    return df


def clean(df: pd.DataFrame, lo=0.005, hi=0.995) -> pd.DataFrame:
    """Drop duplicates, invalid rows and extreme outliers (lead time & sales percentiles)."""
    d = df.drop_duplicates("Row ID").dropna(subset=CAT + NUM + [TARGET])
    d = d[(d[TARGET] > 0) & (d["Units"] > 0) & (d["Sales"] > 0)]
    for c in (TARGET, "Sales"):
        a, b = d[c].quantile([lo, hi])
        d = d[d[c].between(a, b)]
    return d.reset_index(drop=True)


def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
        ("num", StandardScaler(), NUM),
    ])


def _estimators():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1,
                                               random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.08,
                                                       random_state=RANDOM_STATE),
    }


def train_models(df: pd.DataFrame):
    """Fit the three models on a train split, evaluate on the hold-out, refit winner on all data.

    Selection rule (accuracy + interpretability): best tree model by RMSE, unless the linear
    baseline is within 2% of it, in which case the interpretable linear model wins.
    A Random Forest is always refit on all data too, because its tree spread gives the
    prediction-uncertainty used in the scenario confidence score.
    """
    X, y = df[CAT + NUM], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    rows = []
    for name, est in _estimators().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", est)]).fit(Xtr, ytr)
        p = pipe.predict(Xte)
        rows.append({"Model": name, "RMSE": mean_squared_error(yte, p) ** 0.5,
                     "MAE": mean_absolute_error(yte, p), "R2": r2_score(yte, p)})
    metrics = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best_tree = metrics[metrics.Model != "Linear Regression"].iloc[0]
    lin = metrics[metrics.Model == "Linear Regression"].iloc[0]
    best = "Linear Regression" if lin.RMSE <= best_tree.RMSE * 1.02 else best_tree.Model
    est = _estimators()
    final = Pipeline([("prep", make_preprocessor()), ("model", clone(est[best]))]).fit(X, y)
    rf = Pipeline([("prep", make_preprocessor()), ("model", clone(est["Random Forest"]))]).fit(X, y)
    return metrics, best, final, rf


def cluster_routes(df: pd.DataFrame, k=3):
    """Cluster factory->state routes by performance similarity."""
    r = df.groupby(["Factory", "State/Province"]).agg(
        orders=(TARGET, "size"), avg_lead=(TARGET, "mean"), lead_std=(TARGET, "std"),
        distance_km=("distance_km", "mean"), avg_profit=("Gross Profit", "mean")).reset_index()
    r["lead_std"] = r["lead_std"].fillna(0)
    r = r[r.orders >= 5].reset_index(drop=True)
    Z = StandardScaler().fit_transform(r[["avg_lead", "lead_std", "distance_km"]])
    r["cluster"] = KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(Z)
    order = list(r.groupby("cluster").avg_lead.mean().sort_values().index)
    names = ["Fast / reliable", "Moderate", "Consistently slow"]
    r["route_cluster"] = r["cluster"].map({c: names[i] for i, c in enumerate(order)})
    return r.drop(columns="cluster")


def congested_combos(df: pd.DataFrame):
    """Product x Region combos with above-median volume AND top-quartile lead time."""
    g = df.groupby(["Product Name", "Region"]).agg(orders=(TARGET, "size"), units=("Units", "sum"),
                                                   avg_lead=(TARGET, "mean")).reset_index()
    g["congested"] = (g.orders >= g.orders.median()) & (g.avg_lead >= g.avg_lead.quantile(0.75))
    return g.sort_values("avg_lead", ascending=False)
