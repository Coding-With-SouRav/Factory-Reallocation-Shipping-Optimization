# Factory Reallocation & Shipping Optimization – Nassau Candy Distributor

Predicts shipping lead time under alternative product→factory assignments, simulates reassignments,
and recommends the best ones by balancing speed and profit.

## Run
```bash
pip install -r requirements.txt
python train.py          # optional: trains models, writes artifacts/ (metrics, clusters, recommendations)
streamlit run app.py     # dashboard (trains in-memory on first load, ~10-60 s)
```

## Structure
| File | Purpose |
|---|---|
| `src/config.py` | Factory coordinates, product→factory map, state centroids, cost assumptions |
| `src/pipeline.py` | Cleaning, feature engineering (lead time, haversine distance), 3 models, route clustering |
| `src/simulate.py` | Scenario engine, ranking/optimisation, confidence & risk, KPIs |
| `train.py` | End-to-end offline run → `artifacts/` |
| `app.py` | Streamlit dashboard (Simulator, What-If, Recommendations, Risk panel, Model & Clusters) |

## Method (follows the brief)
1. **Prep** – lead time = Ship Date − Order Date; drop duplicates and 0.5/99.5-percentile outliers of lead time and sales; one-hot categoricals, standardise numerics.
2. **Models** – Linear Regression (baseline), Random Forest, Gradient Boosting; 80/20 hold-out; RMSE / MAE / R². Winner = best tree model unless linear is within 2% RMSE (interpretability).
3. **Clustering** – KMeans (k=3) on factory→state routes (avg lead, lead variability, distance) → Fast / Moderate / Consistently slow; product×region congestion flags.
4. **Simulation** – for each product, re-score its real historical orders as if made at each of the 5 factories (distance recomputed), predict lead time, estimate freight & profit change, uncertainty and support.
5. **Recommendation** – `score = w·speed + (1−w)·profit − 0.2·(1−confidence)` (min-max normalised); `w` is the dashboard slider. Top-N per product.

## Important assumptions & data caveats (read before presenting results)
- **Lead times are ~900–1,640 days.** Ship dates in the file fall in 2026–2030 for 2024–25 orders, so the raw lead time is dominated by an order-year artefact (≈ +400 days for 2025 orders). Order year/month are included as control features so the model absorbs it; absolute days should not be read as real transit times. Please verify the date fields with the data owner.
- **Factory is not a model input.** Every product maps to exactly one factory in history, so factory is perfectly collinear with product and its counterfactual effect cannot be learned. A factory swap acts through `distance_km` (factory → customer state centroid), which does vary across orders.
- **Distance has a very weak effect on lead time** in this data, so predicted lead-time gains are small (~1 day). Most of the value of reassignments shows up in the freight/profit side.
- **Freight cost is a proxy** (`FREIGHT_COST_PER_UNIT_KM × km × units × ship-mode multiplier`, in `config.py`) – the dataset has no freight cost. Factory capacity and unit-cost differences are not modelled. Treat profit impact as directional and tune the assumptions.
- Customer location uses state/province centroids (no customer coordinates in the data).
- Confidence score (0–100) = 50% historical support (orders shipped from that factory into those regions) + 50% low Random-Forest tree disagreement. Scenarios with confidence < 40 or profit impact < −2% are flagged high-risk.
