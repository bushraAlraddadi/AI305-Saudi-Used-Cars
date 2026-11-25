"""Task 3 - Classification of two categorical features: Gear_Type and Options.

Run:  python src/train_classification.py
Outputs: models/best_classifier_<target>.joblib, reports/results/classification_*.csv,
         reports/figures/2*_*.png
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
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score, recall_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATEGORICAL, FIG_DIR, MIN_PLAUSIBLE_PRICE, MODEL_DIR, NUMERIC, RANDOM_STATE,
                    RES_DIR, TEST_SIZE, clean_rows, load_raw, make_preprocessor)

for d in (FIG_DIR, RES_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")

TARGETS = {
    # binary, imbalanced (86% / 14%) -> tests our imbalance handling
    "Gear_Type": ["Automatic", "Manual"],
    # 3 ordered classes, fairly balanced -> harder, multi-class
    "Options": ["Standard", "Semi Full", "Full"],
}

base = clean_rows(load_raw())
# Price is a useful input (e.g. Full-option cars cost more), but 1/3 of ads have no price.
# Unpriced (0) and instalment (<5000) prices -> NaN; the imputer fills them and adds a
# "price was missing" indicator column, so the model can still use those rows.
base.loc[base.Price < MIN_PLAUSIBLE_PRICE, "Price"] = np.nan


def models_and_grids():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000), {
            "model__C": [0.01, 0.1, 1, 10],
            "model__class_weight": [None, "balanced"]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=RANDOM_STATE), {
            "model__max_depth": [5, 10, 20, None],
            "model__min_samples_leaf": [1, 5, 10],
            "model__class_weight": [None, "balanced"]}),
        "Random Forest": (RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1), {
            "model__max_depth": [20, None],
            "model__min_samples_leaf": [1, 3],
            "model__class_weight": [None, "balanced"]}),
        "SVM": (SVC(kernel="rbf"), {
            "model__C": [0.3, 1, 10],
            "model__gamma": ["scale", 0.01],
            "model__class_weight": [None, "balanced"]}),
        # MLP has no class_weight option -> we pass balanced sample weights instead (see below)
        "MLP": (MLPClassifier(max_iter=1000, early_stopping=True, random_state=RANDOM_STATE), {
            "model__hidden_layer_sizes": [(64,), (128, 64)],
            "model__alpha": [1e-4, 1e-3, 1e-2]}),
    }


def scores(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred),
            "balanced_acc": balanced_accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted")}


summary = {}
for i, (target, labels) in enumerate(TARGETS.items()):
    print(f"\n================ TARGET: {target} ================")
    df = base.dropna(subset=[target]).reset_index(drop=True)      # target must be known
    cat_in = [c for c in CATEGORICAL if c != target]
    num_in = NUMERIC + ["Price"]
    features = num_in + cat_in
    X, y = df[features], df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print("Class distribution (train):", y_train.value_counts(normalize=True).round(3).to_dict())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rows, fitted, preds = [], {}, {}
    dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    rows.append({"model": "Baseline (majority)", "cv_f1_macro": np.nan, "best_params": "-",
                 "fit_sec": 0, **scores(y_test, dummy.predict(X_test))})

    for name, (est, grid) in models_and_grids().items():
        pipe = Pipeline([("prep", make_preprocessor(num_in, cat_in, add_indicator=True)), ("model", est)])
        fit_params = {}
        if name == "MLP":
            fit_params["model__sample_weight"] = compute_sample_weight("balanced", y_train)
        t0 = time.time()
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="f1_macro", n_jobs=-1)
        gs.fit(X_train, y_train, **fit_params)
        secs = time.time() - t0
        p = gs.best_estimator_.predict(X_test)
        fitted[name], preds[name] = gs.best_estimator_, p
        params = {k.replace("model__", ""): v for k, v in gs.best_params_.items()}
        s = scores(y_test, p)
        rows.append({"model": name, "cv_f1_macro": gs.best_score_,
                     "best_params": json.dumps(params, default=str), "fit_sec": round(secs, 1), **s,
                     "train_f1_macro": f1_score(y_train, gs.best_estimator_.predict(X_train), average="macro")})
        print(f"{name:20s} cv_F1={gs.best_score_:.3f}  test acc={s['accuracy']:.3f}  "
              f"F1macro={s['f1_macro']:.3f}  ({secs:.0f}s) {params}")
        pd.DataFrame(gs.cv_results_)[["params", "mean_test_score", "std_test_score", "rank_test_score"]] \
            .sort_values("rank_test_score") \
            .to_csv(RES_DIR / f"classification_{target}_grid_{name.replace(' ', '_')}.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(RES_DIR / f"classification_{target}_results.csv", index=False)
    print(res.drop(columns="best_params").round(3).to_string(index=False))

    best_name = res.dropna(subset=["cv_f1_macro"]).sort_values("cv_f1_macro", ascending=False).iloc[0]["model"]
    joblib.dump(fitted[best_name], MODEL_DIR / f"best_classifier_{target.lower()}.joblib", compress=3)
    rep = classification_report(y_test, preds[best_name], labels=labels, digits=3)
    (RES_DIR / f"classification_{target}_best_report.txt").write_text(f"Best model: {best_name}\n\n{rep}")
    print("Best (by CV macro-F1):", best_name, "\n", rep)
    summary[target] = {"model": best_name, "features": features, "classes": labels,
                       "test_metrics": {k: float(v) for k, v in scores(y_test, preds[best_name]).items()}}

    # confusion matrices of all 5 models
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6))
    for a, (name, p) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, p, labels=labels)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a,
                    xticklabels=labels, yticklabels=labels)
        a.set_title(f"{name}\nF1-macro={f1_score(y_test, p, average='macro'):.3f}")
        a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    plt.tight_layout(); plt.savefig(FIG_DIR / f"2{i}_confusion_{target}.png", dpi=130); plt.close()

    plot = res[~res.model.str.startswith("Baseline")].set_index("model")[
        ["accuracy", "precision_macro", "recall_macro", "f1_macro"]]
    plot.plot.bar(figsize=(10, 4.5), rot=0, ylim=(0, 1), title=f"{target}: test metrics by model")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(FIG_DIR / f"2{i}_metrics_{target}.png", dpi=130); plt.close()

    # error analysis for the best model
    err = X_test.copy()
    err["actual"], err["predicted"] = y_test.values, preds[best_name]
    wrong = err[err.actual != err.predicted]
    wrong.to_csv(RES_DIR / f"classification_{target}_misclassified.csv", index=False)
    counts = X_train.Type.value_counts()
    err["rare_model"] = err.Type.map(counts).fillna(0) < 10
    err["correct"] = err.actual == err.predicted
    by_rare = err.groupby("rare_model").correct.mean()
    print(f"Accuracy on rare car models (<10 in train): {by_rare.get(True, np.nan):.3f} "
          f"vs common: {by_rare.get(False, np.nan):.3f}")
    print("Most frequent mistakes (actual -> predicted):")
    print(wrong.groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(4).to_string())
    print("Car models with most mistakes:", wrong.Type.value_counts().head(6).to_dict())
    summary[target]["error_analysis"] = {
        "acc_rare_models": float(by_rare.get(True, np.nan)),
        "acc_common_models": float(by_rare.get(False, np.nan)),
        "top_confusions": {f"{a}->{b}": int(n) for (a, b), n in
                           wrong.groupby(["actual", "predicted"]).size().sort_values(ascending=False).head(4).items()},
        "models_with_most_errors": wrong.Type.value_counts().head(6).to_dict()}

json.dump(summary, open(MODEL_DIR / "best_classifiers.json", "w"), indent=2)
