"""Task 2 - Price prediction (regression).

Run:  python src/train_regression.py
Outputs: models/best_regressor.joblib, reports/results/regression_*.csv, reports/figures/1*_*.png
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATEGORICAL, FIG_DIR, MIN_PLAUSIBLE_PRICE, MODEL_DIR, NUMERIC, RANDOM_STATE,
                    RES_DIR, TEST_SIZE, clean_rows, load_raw, make_preprocessor)

for d in (FIG_DIR, RES_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------ data
df = clean_rows(load_raw())
n_before = len(df)
n_zero = int((df.Price == 0).sum())
n_low = int(((df.Price > 0) & (df.Price < MIN_PLAUSIBLE_PRICE)).sum())
# Decision on Price == 0: these are "Negotiable" ads with NO price. 0 is not a real price,
# it is a missing target, so we cannot learn from (or evaluate on) them -> remove them.
# Prices < 5,000 SAR are monthly instalments, not car prices -> also removed.
df = df[df.Price >= MIN_PLAUSIBLE_PRICE].reset_index(drop=True)
print(f"Rows: {n_before} -> removed {n_zero} with Price=0 and {n_low} with Price<{MIN_PLAUSIBLE_PRICE} -> {len(df)}")

# Inputs: all descriptive features. Negotiable is excluded: after the filter it is always False.
FEATURES = NUMERIC + CATEGORICAL
X, y = df[FEATURES], df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
print("Train/test:", X_train.shape, X_test.shape)


def make_model(estimator):
    """Preprocessing + model; the model learns log(price) because price is strongly
    right-skewed (skew 3.6). Predictions are converted back to SAR automatically."""
    pipe = Pipeline([("prep", make_preprocessor(NUMERIC, CATEGORICAL)), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


R = "regressor__"
MINF = R + "prep__columns__cat__onehot__min_frequency"
models = {
    "Linear Regression": (LinearRegression(), {
        MINF: [1, 5, 10, 20], R + "model__fit_intercept": [True, False]}),
    "Decision Tree": (DecisionTreeRegressor(random_state=RANDOM_STATE), {
        R + "model__max_depth": [10, 15, 20, None],
        R + "model__min_samples_leaf": [1, 3, 5, 10]}),
    "Random Forest": (RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), {
        R + "model__max_depth": [20, None],
        R + "model__min_samples_leaf": [1, 2],
        R + "model__max_features": [0.3, 0.6, 1.0]}),
    "SVR": (SVR(kernel="rbf"), {
        R + "model__C": [0.3, 1, 3, 10, 100],
        R + "model__epsilon": [0.05, 0.1],
        R + "model__gamma": ["scale", 0.01]}),
    "MLP": (MLPRegressor(max_iter=1000, early_stopping=True, random_state=RANDOM_STATE), {
        R + "model__hidden_layer_sizes": [(64,), (128, 64), (64, 32)],
        R + "model__alpha": [1e-4, 1e-3, 1e-2],
        R + "model__learning_rate_init": [1e-3, 5e-3]}),
}


def metrics(y_true, y_pred):
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "MAPE_%": float(np.mean(np.abs(y_true - y_pred) / y_true) * 100)}


cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted, preds = [], {}, {}

base = DummyRegressor(strategy="median").fit(X_train, y_train)
rows.append({"model": "Baseline (median)", "cv_MAE": np.nan, "best_params": "-", "fit_sec": 0,
             **metrics(y_test, base.predict(X_test))})

for name, (est, grid) in models.items():
    t0 = time.time()
    gs = GridSearchCV(make_model(est), grid, cv=cv, scoring="neg_mean_absolute_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    secs = time.time() - t0
    p = gs.best_estimator_.predict(X_test)
    tr = gs.best_estimator_.predict(X_train)
    fitted[name], preds[name] = gs.best_estimator_, p
    params = {k.replace(R, "").replace("model__", ""): v for k, v in gs.best_params_.items()}
    m = metrics(y_test, p)
    rows.append({"model": name, "cv_MAE": -gs.best_score_, "best_params": json.dumps(params, default=str),
                 "fit_sec": round(secs, 1), **m, "train_R2": r2_score(y_train, tr)})
    print(f"{name:18s} cv_MAE={-gs.best_score_:9.0f}  test MAE={m['MAE']:9.0f}  RMSE={m['RMSE']:9.0f}  "
          f"R2={m['R2']:.3f}  ({secs:.0f}s)  {params}")
    pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
        .sort_values("rank_test_score").to_csv(RES_DIR / f"regression_grid_{name.replace(' ', '_')}.csv", index=False)

res = pd.DataFrame(rows)
res.to_csv(RES_DIR / "regression_results.csv", index=False)
print(res.drop(columns="best_params").round(3).to_string(index=False))

# Model selection uses the CROSS-VALIDATION score (not the test set), then we report test.
best_name = res.dropna(subset=["cv_MAE"]).sort_values("cv_MAE").iloc[0]["model"]
best = fitted[best_name]
joblib.dump(best, MODEL_DIR / "best_regressor.joblib")
for name, m in fitted.items():
    joblib.dump(m, MODEL_DIR / f"regressor_{name.replace(' ', '_').lower()}.joblib", compress=3)
json.dump({"model": best_name, "features": FEATURES,
           "test_metrics": {k: float(v) for k, v in metrics(y_test, preds[best_name]).items()}},
          open(MODEL_DIR / "best_regressor.json", "w"), indent=2)
print("Best (by CV MAE):", best_name)

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
for a, (name, p) in zip(axes, preds.items()):
    a.scatter(y_test, p, s=6, alpha=0.4)
    lim = [5e3, 1.2e6]
    a.plot(lim, lim, "r--", lw=1)
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlim(lim); a.set_ylim(lim)
    a.set_title(f"{name}\nR²={r2_score(y_test, p):.3f}  MAE={mean_absolute_error(y_test, p):,.0f}")
    a.set_xlabel("Actual price (SAR)"); a.set_ylabel("Predicted")
plt.tight_layout(); plt.savefig(FIG_DIR / "10_regression_pred_vs_actual.png", dpi=130); plt.close()

plot = res[res.model != "Baseline (median)"].set_index("model")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
plot["MAE"].plot.barh(ax=ax[0], color="#4C72B0", title="Test MAE (SAR, lower=better)")
plot["RMSE"].plot.barh(ax=ax[1], color="#DD8452", title="Test RMSE (SAR, lower=better)")
plot["R2"].plot.barh(ax=ax[2], color="#55A868", title="Test R² (higher=better)")
plt.tight_layout(); plt.savefig(FIG_DIR / "11_regression_comparison.png", dpi=130); plt.close()

# Feature importance of the Random Forest, summed back to original columns
rf = fitted["Random Forest"].regressor_
names = rf.named_steps["prep"].named_steps["columns"].get_feature_names_out()
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=names)
orig = imp.groupby(lambda n: next(f for f in FEATURES if n.split("__", 1)[1].startswith(f))).sum()
orig = orig.sort_values()
orig.to_csv(RES_DIR / "regression_rf_feature_importance.csv")
orig.plot.barh(figsize=(7, 4.5), color="#4C72B0", title="Random Forest feature importance (grouped)")
plt.tight_layout(); plt.savefig(FIG_DIR / "12_regression_feature_importance.png", dpi=130); plt.close()

# ------------------------------------------------------------------ error analysis (best model)
err = X_test.copy()
err["Actual"] = y_test
err["Predicted"] = preds[best_name].round(0)
err["AbsError"] = (err.Actual - err.Predicted).abs()
err["PctError"] = err.AbsError / err.Actual * 100
err.sort_values("AbsError", ascending=False).head(15).to_csv(RES_DIR / "regression_worst_errors.csv")
err["price_band"] = pd.cut(err.Actual, [0, 30e3, 60e3, 100e3, 200e3, 2e6],
                           labels=["<30k", "30-60k", "60-100k", "100-200k", ">200k"])
band = err.groupby("price_band", observed=True).agg(n=("Actual", "size"), MAE=("AbsError", "mean"),
                                                    median_pct_error=("PctError", "median"))
band.to_csv(RES_DIR / "regression_error_by_price_band.csv")
counts = X_train.Type.value_counts()
err["type_freq_in_train"] = err.Type.map(counts).fillna(0)
err["type_bucket"] = pd.cut(err.type_freq_in_train, [-1, 9, 30, 100, 1e5],
                            labels=["<10 (rare)", "10-30", "31-100", ">100"])
rare = err.groupby("type_bucket", observed=True).agg(n=("Actual", "size"), MAE=("AbsError", "mean"),
                                                     median_pct_error=("PctError", "median"))
rare.to_csv(RES_DIR / "regression_error_by_model_rarity.csv")
print("\nError by price band:\n", band.round(1).to_string())
print("\nError by how often the car model appears in training:\n", rare.round(1).to_string())
print("\nWorst 8 predictions:\n", err.sort_values("AbsError", ascending=False).head(8)
      [["Make", "Type", "Year", "Mileage", "Actual", "Predicted", "PctError"]].round(0).to_string())

plt.figure(figsize=(7, 4))
sns.histplot(np.log(err.Predicted / err.Actual), bins=60)
plt.axvline(0, color="r", ls="--"); plt.xlabel("log(predicted / actual)")
plt.title(f"{best_name}: residual distribution (test set)")
plt.tight_layout(); plt.savefig(FIG_DIR / "13_regression_residuals.png", dpi=130); plt.close()
