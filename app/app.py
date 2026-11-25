"""Task 4 - Flask API serving the best regression and classification models.

Routes
  GET  /                      health check + model info
  POST /predict/price         -> best regression model (price in SAR)
  POST /predict/class         -> best classification model
                                 ?target=gear_type (default) or ?target=options
Body: one JSON object (a car) or a list of objects.
"""
import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))   # the saved pipelines use common.GroupMedianImputer
import common  # noqa: E402,F401

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("car-api")

MODELS = ROOT / "models"
regressor = joblib.load(MODELS / "best_regressor.joblib")
reg_info = json.load(open(MODELS / "best_regressor.json"))
classifiers = {
    "gear_type": joblib.load(MODELS / "best_classifier_gear_type.joblib"),
    "options": joblib.load(MODELS / "best_classifier_options.joblib"),
}
clf_info = json.load(open(MODELS / "best_classifiers.json"))
clf_info = {"gear_type": clf_info["Gear_Type"], "options": clf_info["Options"]}
log.info("Loaded regressor=%s, classifiers=%s", reg_info["model"],
         {k: v["model"] for k, v in clf_info.items()})

app = Flask(__name__)
NUMERIC_FIELDS = {"Year", "Engine_Size", "Mileage", "Price"}
REQUIRED = {"Make", "Type", "Year"}   # everything else may be missing and is imputed


def to_frame(payload, features):
    rows = payload if isinstance(payload, list) else [payload]
    if not rows or not all(isinstance(r, dict) for r in rows):
        raise ValueError("Body must be a JSON object or a list of objects")
    for i, r in enumerate(rows):
        missing = REQUIRED - r.keys()
        if missing:
            raise ValueError(f"Record {i}: missing required fields {sorted(missing)}")
    df = pd.DataFrame(rows).reindex(columns=features)
    for c in features:
        if c in NUMERIC_FIELDS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype(object).where(df[c].notna(), np.nan)
    return df, len(rows)


@app.get("/")
def health():
    return jsonify(status="ok",
                   regression_model=reg_info["model"],
                   classification_models={k: v["model"] for k, v in clf_info.items()})


@app.post("/predict/price")
def predict_price():
    t0 = time.time()
    try:
        X, n = to_frame(request.get_json(force=True), reg_info["features"])
    except Exception as e:
        log.warning("Bad /predict/price request: %s", e)
        return jsonify(error=str(e)), 400
    pred = regressor.predict(X)
    out = [{"predicted_price_SAR": round(float(p), -2)} for p in pred]
    log.info("/predict/price n=%d -> %s (%.0f ms)", n, [o["predicted_price_SAR"] for o in out],
             (time.time() - t0) * 1000)
    return jsonify(model=reg_info["model"], predictions=out)


@app.post("/predict/class")
def predict_class():
    t0 = time.time()
    target = request.args.get("target", "gear_type").lower()
    if target not in classifiers:
        return jsonify(error=f"target must be one of {list(classifiers)}"), 400
    model, info = classifiers[target], clf_info[target]
    try:
        X, n = to_frame(request.get_json(force=True), info["features"])
    except Exception as e:
        log.warning("Bad /predict/class request: %s", e)
        return jsonify(error=str(e)), 400
    labels = model.predict(X)
    out = [{"predicted_class": str(l)} for l in labels]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        for o, p in zip(out, proba):
            o["probabilities"] = {str(c): round(float(v), 3) for c, v in zip(model.classes_, p)}
    log.info("/predict/class target=%s n=%d -> %s (%.0f ms)", target, n,
             [o["predicted_class"] for o in out], (time.time() - t0) * 1000)
    return jsonify(target=target, model=info["model"], predictions=out)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
