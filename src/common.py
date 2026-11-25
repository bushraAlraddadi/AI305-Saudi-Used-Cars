"""Shared helpers: loading, row-level cleaning, feature lists and preprocessing.

Important rule followed everywhere: anything that *learns* from the data
(imputation values, scaling statistics, one-hot categories) lives inside a
scikit-learn Pipeline, so it is fitted on the training split only.
Only row-level, rule-based cleaning (no statistics) happens before the split.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "UsedCarsSA_Unclean_EN.csv"
FIG_DIR = ROOT / "reports" / "figures"
RES_DIR = ROOT / "reports" / "results"
MODEL_DIR = ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Thresholds for implausible values (decided during EDA, see eda.py)
MAX_PLAUSIBLE_MILEAGE = 1_000_000   # km; above this is almost surely a typo (e.g. 20,000,000)
MIN_PLAUSIBLE_PRICE = 5_000         # SAR; lower "prices" are monthly instalments (e.g. a 2020 Elantra at 1,221)

NUMERIC = ["Year", "Engine_Size", "Mileage"]
CATEGORICAL = ["Make", "Type", "Origin", "Color", "Options",
               "Fuel_Type", "Gear_Type", "Region"]


def load_raw(path=DATA_FILE) -> pd.DataFrame:
    """Load the original 8,248-row file. Falls back to kagglehub if the CSV is missing."""
    path = Path(path)
    if not path.exists():
        import kagglehub
        folder = Path(kagglehub.dataset_download("turkibintalib/saudi-arabia-used-cars-dataset"))
        path = folder / "UsedCarsSA_Unclean_EN.csv"
    return pd.read_csv(path)


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-based cleaning that does not learn anything from the data."""
    df = df.copy()
    # Link is a unique URL (identifier, no predictive value); Condition is constant ("Used").
    df = df.drop(columns=["Link", "Condition"])
    # Price is stored as text because unpriced ads say "Negotiable". Convert them to 0.
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce").fillna(0).astype(int)
    # Exact duplicate ads (same car posted twice) would leak between train and test.
    df = df.drop_duplicates().reset_index(drop=True)
    # Impossible mileage -> treat as missing (imputed later inside the pipeline).
    df.loc[df["Mileage"] > MAX_PLAUSIBLE_MILEAGE, "Mileage"] = np.nan
    return df


class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Fill Engine_Size with the median of the same car model (Type),
    falling back to the Make median, then the global median.
    Fitted on training data only."""

    def __init__(self, target="Engine_Size", groups=("Type", "Make")):
        self.target = target
        self.groups = groups

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.maps_ = {g: X.groupby(g)[self.target].median().dropna().to_dict()
                      for g in self.groups}
        self.global_ = float(X[self.target].median())
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for g in self.groups:
            miss = X[self.target].isna()
            X.loc[miss, self.target] = X.loc[miss, g].map(self.maps_[g])
        X[self.target] = X[self.target].fillna(self.global_).astype(float)
        return X


def make_preprocessor(numeric, categorical, add_indicator=False):
    """Impute -> scale numeric; impute -> one-hot categorical.
    Rare categories (< 10 cars) are grouped into one 'infrequent' column,
    which also lets the API accept car models it has never seen.
    add_indicator=True adds a 0/1 column saying a numeric value was missing
    (used for Price in classification: missing = 'Negotiable' ad)."""
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=add_indicator)),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=10, sparse_output=False)),
    ])
    ct = ColumnTransformer([
        ("num", num_pipe, numeric),
        ("cat", cat_pipe, categorical),
    ])
    steps = []
    if "Engine_Size" in numeric and "Type" in categorical and "Make" in categorical:
        steps.append(("engine_imputer", GroupMedianImputer()))
    steps.append(("columns", ct))
    return Pipeline(steps)
