"""Task 1 - Data Exploration Analysis.

Run:  python src/eda.py
Outputs: reports/figures/*.png and reports/results/eda_*.csv / eda_summary.txt
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CATEGORICAL, FIG_DIR, MAX_PLAUSIBLE_MILEAGE, MIN_PLAUSIBLE_PRICE,
                    NUMERIC, RES_DIR, load_raw)

FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")
lines = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    lines.append(s)


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


def cramers_v(x, y):
    """Association between two categorical variables (0 = none, 1 = perfect)."""
    t = pd.crosstab(x, y).to_numpy()
    n = t.sum()
    expected = t.sum(1, keepdims=True) @ t.sum(0, keepdims=True) / n
    chi2 = ((t - expected) ** 2 / np.where(expected == 0, 1, expected)).sum()
    r, k = t.shape
    phi2 = max(0, chi2 / n - (k - 1) * (r - 1) / (n - 1))       # bias-corrected
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)
    d = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2 / d)) if d > 0 else 0.0


def correlation_ratio(cat, num):
    """Eta: how much of a numeric variable's variance is explained by a categorical one."""
    d = pd.DataFrame({"c": cat, "n": num}).dropna()
    grand = d["n"].mean()
    between = d.groupby("c")["n"].agg(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    total = ((d["n"] - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else 0.0


# ---------------------------------------------------------------- 1. structure
raw = load_raw()
log("=== 1. STRUCTURE ===")
log("Stored as a single CSV table; one row = one used-car ad on syarah.com")
log("Shape:", raw.shape)
log(raw.dtypes.to_string())
log(raw.head().to_string())

# ---------------------------------------------------------------- 2. issues in raw data
log("\n=== 2. DATA ISSUES (raw) ===")
log("Missing values per column:\n" + raw.isna().sum().to_string())
log("Condition values:", raw["Condition"].value_counts().to_dict(), "-> constant, dropped")
log("Link unique:", raw["Link"].nunique(), "of", len(raw), "-> identifier, dropped")
neg = (raw["Price"] == "Negotiable").sum()
log(f"Price stored as text; 'Negotiable' (no price) rows: {neg} ({neg/len(raw):.1%}) -> set to 0")
log("Negotiable flag vs missing price:\n" +
    pd.crosstab(raw["Negotiable"], raw["Price"] == "Negotiable").to_string())
log("Exact duplicates incl. Link:", raw.duplicated().sum(),
    "| duplicates ignoring Link:", raw.drop(columns="Link").duplicated().sum())

df = raw.drop(columns=["Link", "Condition"]).copy()
df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
df = df.drop_duplicates().reset_index(drop=True)
log("Rows after dropping duplicates:", len(df))

priced = df[df["Price"] > 0]
log(f"Implausible prices 0 < Price < {MIN_PLAUSIBLE_PRICE}: {(priced.Price < MIN_PLAUSIBLE_PRICE).sum()}")
log(priced[priced.Price < MIN_PLAUSIBLE_PRICE].sort_values("Price")
    [["Make", "Type", "Year", "Mileage", "Price"]].head(10).to_string())
log(f"Implausible mileage > {MAX_PLAUSIBLE_MILEAGE:,} km: {(df.Mileage > MAX_PLAUSIBLE_MILEAGE).sum()}",
    "| top values:", sorted(df.Mileage.nlargest(5).tolist()))
log("Very old cars (Year < 1980):", (df.Year < 1980).sum(), "-> rare but plausible (classic cars), kept")
log("Engine_Size missing:", df.Engine_Size.isna().sum(),
    "-> imputed with the median of the same model (fitted on train only)")

# ---------------------------------------------------------------- 3. descriptive stats
log("\n=== 3. DESCRIPTIVE STATISTICS ===")
num_cols = NUMERIC + ["Price"]
desc = df[num_cols].describe().T
desc["skew"] = df[num_cols].skew()
desc["missing"] = df[num_cols].isna().sum()
log(desc.round(2).to_string())
desc.to_csv(RES_DIR / "eda_numeric_stats.csv")
log("Price (only priced ads, Price >= 5000):\n" +
    priced[priced.Price >= MIN_PLAUSIBLE_PRICE].Price.describe().round(0).to_string())

cat_rows = []
for c in CATEGORICAL + ["Negotiable"]:
    vc = df[c].value_counts(dropna=False)
    cat_rows.append({"feature": c, "n_unique": df[c].nunique(), "missing": int(df[c].isna().sum()),
                     "top": vc.index[0], "top_share": round(vc.iloc[0] / len(df), 3),
                     "imbalance_ratio_max_min": round(vc.iloc[0] / vc.iloc[-1], 1)})
cat_stats = pd.DataFrame(cat_rows)
log(cat_stats.to_string(index=False))
cat_stats.to_csv(RES_DIR / "eda_categorical_stats.csv", index=False)
for c in ["Origin", "Options", "Fuel_Type", "Gear_Type", "Color"]:
    log(f"{c}: {df[c].value_counts(dropna=False).to_dict()}")

# distributions of numeric features
fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
sns.histplot(df.Year, bins=40, ax=ax[0]).set_title("Year")
sns.histplot(df.Engine_Size.dropna(), bins=30, ax=ax[1]).set_title("Engine size (L)")
sns.histplot(df.Mileage[df.Mileage <= MAX_PLAUSIBLE_MILEAGE], bins=50, ax=ax[2]).set_title("Mileage (km, <=1M)")
sns.histplot(priced.Price[priced.Price >= MIN_PLAUSIBLE_PRICE], bins=60, ax=ax[3]).set_title("Price (SAR, priced ads)")
save("01_numeric_distributions.png")

# balance of categorical features
fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(axes.ravel(), ["Origin", "Options", "Fuel_Type", "Gear_Type", "Color", "Region"]):
    vc = df[c].fillna("Missing").value_counts().head(12)
    sns.barplot(x=vc.values, y=vc.index.astype(str), ax=a, color="#4C72B0")
    a.set_title(f"{c} (n_unique={df[c].nunique()})")
save("02_categorical_balance.png")

vc = df.Make.value_counts().head(20)
plt.figure(figsize=(8, 6))
sns.barplot(x=vc.values, y=vc.index, color="#4C72B0").set_title("Top 20 makes")
save("03_top_makes.png")

# ---------------------------------------------------------------- 4. correlation with price
log("\n=== 4. CORRELATION WITH PRICE (priced ads, plausible values) ===")
p = df[(df.Price >= MIN_PLAUSIBLE_PRICE) & (df.Mileage <= MAX_PLAUSIBLE_MILEAGE)].copy()
p["Negotiable"] = p["Negotiable"].astype(int)
pear = p[num_cols].corr(method="pearson")
spear = p[num_cols].corr(method="spearman")
corr_tbl = pd.DataFrame({"pearson": pear["Price"], "spearman": spear["Price"]}).drop("Price")
log(corr_tbl.round(3).to_string())
corr_tbl.to_csv(RES_DIR / "eda_price_correlation.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[0]).set_title("Pearson")
sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax[1]).set_title("Spearman")
save("04_correlation_heatmaps.png")

# ---------------------------------------------------------------- 5. how each feature affects price
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for a, c in zip(ax, NUMERIC):
    sns.scatterplot(data=p, x=c, y="Price", s=8, alpha=0.35, ax=a)
    a.set_yscale("log"); a.set_title(f"Price vs {c} (log scale)")
save("05_price_vs_numeric.png")

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
for a, c in zip(axes.ravel(), ["Options", "Origin", "Gear_Type", "Fuel_Type", "Color", "Region"]):
    order = p.groupby(c).Price.median().sort_values(ascending=False).index[:10]
    sns.boxplot(data=p[p[c].isin(order)], x="Price", y=c, order=order, ax=a,
                showfliers=False, color="#8FB3D9")
    a.set_title(f"Price by {c}")
save("06_price_by_categorical.png")

top_makes = p.Make.value_counts().head(15).index
order = p[p.Make.isin(top_makes)].groupby("Make").Price.median().sort_values(ascending=False).index
plt.figure(figsize=(9, 6))
sns.boxplot(data=p[p.Make.isin(top_makes)], x="Price", y="Make", order=order, showfliers=False, color="#8FB3D9")
plt.title("Price by Make (15 most common)")
save("07_price_by_make.png")

eff = []
for c in CATEGORICAL:
    eff.append({"feature": c, "eta_with_price": round(correlation_ratio(p[c], p.Price), 3),
                "median_price_range": f"{p.groupby(c).Price.median().min():,.0f} - {p.groupby(c).Price.median().max():,.0f}"})
for c in NUMERIC:
    eff.append({"feature": c, "eta_with_price": round(abs(spear.loc[c, "Price"]), 3),
                "median_price_range": "(|Spearman|)"})
eff = pd.DataFrame(eff).sort_values("eta_with_price", ascending=False)
log("\n=== 5. EFFECT OF EACH FEATURE ON PRICE ===")
log(eff.to_string(index=False))
eff.to_csv(RES_DIR / "eda_feature_effect_on_price.csv", index=False)
log("Median price by Options:", p.groupby("Options").Price.median().to_dict())
log("Median price by Gear_Type:", p.groupby("Gear_Type").Price.median().to_dict())
log("Median price by Origin:", p.groupby("Origin").Price.median().to_dict())

# ---------------------------------------------------------------- 6. each categorical as target
log("\n=== 6. WHICH FEATURES SEPARATE EACH CATEGORICAL TARGET ===")
log("Cramer's V for categorical inputs, correlation ratio (eta) for numeric inputs. 0=no relation, 1=perfect.")
targets = ["Make", "Origin", "Color", "Options", "Fuel_Type", "Gear_Type", "Region", "Negotiable"]
inputs = CATEGORICAL + NUMERIC + ["Price"]
d6 = df.copy()
d6.loc[d6.Price < MIN_PLAUSIBLE_PRICE, "Price"] = np.nan   # unpriced ads are not informative
mat = pd.DataFrame(index=targets, columns=inputs, dtype=float)
for t in targets:
    for f in inputs:
        if f == t:
            continue
        sub = d6[[t, f]].dropna()
        if f in NUMERIC + ["Price"]:
            mat.loc[t, f] = correlation_ratio(sub[t], sub[f])
        else:
            mat.loc[t, f] = cramers_v(sub[t], sub[f])
mat.round(3).to_csv(RES_DIR / "eda_target_vs_features.csv")
for t in targets:
    log(f"{t:>10}: " + ", ".join(f"{k}={v:.2f}" for k, v in mat.loc[t].dropna().sort_values(ascending=False).head(4).items()))
plt.figure(figsize=(12, 5))
sns.heatmap(mat.astype(float), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
plt.title("How well each feature (columns) distinguishes each categorical target (rows)")
save("08_categorical_target_association.png")

# box plots for the two classification targets chosen in Task 3
fig, ax = plt.subplots(1, 4, figsize=(17, 4))
sns.boxplot(data=d6, x="Gear_Type", y="Year", ax=ax[0], showfliers=False)
sns.boxplot(data=d6, x="Gear_Type", y="Engine_Size", ax=ax[1], showfliers=False)
sns.boxplot(data=d6, x="Options", y="Price", ax=ax[2], showfliers=False, order=["Standard", "Semi Full", "Full"])
sns.boxplot(data=d6, x="Options", y="Year", ax=ax[3], showfliers=False, order=["Standard", "Semi Full", "Full"])
for a in ax: a.set_title(f"{a.get_ylabel()} by {a.get_xlabel()}")
save("09_targets_boxplots.png")

# ---------------------------------------------------------------- 7. split note
log("\n=== 7. TRAIN/TEST SPLIT ===")
log("80/20 split with random_state=42, stratified by the target for classification.")
log("The split is done right after row-level cleaning and BEFORE any imputer/scaler/encoder is fitted;")
log("all fitted preprocessing lives in sklearn Pipelines (see train_*.py).")

(RES_DIR / "eda_summary.txt").write_text("\n".join(lines), encoding="utf-8")
print("\nSaved figures to", FIG_DIR)
