# src/style.py — LOCKED conventions (change only with the whole group)
#
# Every notebook and both dashboards import from here:
#     from src.style import *
# so names, colours, era order, seed and the train/test split are identical everywhere.

import ast
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]           # repo root, wherever it is cloned
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

CLEAN_PATH = DATA_DIR / "tracks_clean.csv"           # written by analysis.ipynb (cleaning section)
MODEL_DATA_PATH = DATA_DIR / "model_data.parquet"    # Part 1.2 (Jason)
SPLIT_PATH = DATA_DIR / "split.csv"                  # Part 1.2 (Jason)

# ---- Reproducibility -----------------------------------------------------------
SEED = 42

# ---- Features (sound only: no popularity, followers, year or decade) ---------------
FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
            "key", "mode", "time_signature"]
PCA_FEATURES = [f for f in FEATURES if f not in ("key", "mode", "time_signature")]  # 10 continuous
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]       # Spotify pitch class 0-11

# ---- Target: 4 eras ------------------------------------------------------------
ERA_BINS = [-np.inf, 1959, 1979, 1999, np.inf]      # <=1959 | 1960-79 | 1980-99 | >=2000
ERA_ORDER = [0, 1, 2, 3]                             # class codes, oldest -> newest
ERA_LABELS = ["Pre-1960", "1960–79", "1980–99", "2000+"]
ERA_COLORS = ["#9ecae1", "#4292c6", "#2171b5", "#08306b"]   # light -> dark = old -> new
ERA_COLOR_MAP = dict(zip(ERA_LABELS, ERA_COLORS))           # for plotly color_discrete_map


def _first_id(v):
    """First-listed artist id. Accepts a list or the CSV string form "['id1', 'id2']"."""
    if isinstance(v, (list, tuple, np.ndarray)):
        return v[0] if len(v) else None
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)):
            return parsed[0] if parsed else None
        return parsed
    return None


def load_clean(path=CLEAN_PATH):
    """Read the team's cleaned tracks file."""
    return pd.read_csv(path, low_memory=False)


def music_only(clean_df):
    """Same rows as the EDA's analysis copy (analysis.ipynb 2.8 and 3A):
    drop spoken word, then keep the 1st-99th percentile duration window of what is left."""
    df = clean_df.loc[~clean_df["is_spoken_word"].astype(bool)]
    dur = df["duration_min"] if "duration_min" in df else df["duration_ms"] / 60_000
    lo, hi = dur.quantile([0.01, 0.99])
    return df.loc[dur.between(lo, hi)]


def make_model_table(clean_df):
    df = music_only(clean_df)
    df = df.sort_values("id").reset_index(drop=True)                  # fixed row order
    year = df["year"] if "year" in df else df["release_year"]         # CSV calls it release_year
    df["year"] = year.astype(int)
    df["era"] = pd.cut(df["year"], ERA_BINS, labels=False).astype(int)
    if "primary_artist_id" in df:                                     # already = first-listed id
        df["artist_id"] = df["primary_artist_id"]
    else:
        df["artist_id"] = df["id_artists"].map(_first_id)
    missing = df["artist_id"].isna().sum()
    assert missing == 0, f"{missing} rows have no artist id"
    # The cleaned CSV stores key as note names and mode as Minor/Major (for charts).
    # Map back to Spotify's numeric codes so every model gets numbers.
    # is_numeric_dtype, not `dtype == object`: pandas 3 reads text as the new str dtype.
    if not pd.api.types.is_numeric_dtype(df["key"]):
        df["key"] = df["key"].map({k: i for i, k in enumerate(KEY_NAMES)})
    if not pd.api.types.is_numeric_dtype(df["mode"]):
        df["mode"] = df["mode"].map({"Minor": 0, "Major": 1})
    assert df[["key", "mode"]].notna().all().all(), "unknown key/mode label"  # else NaN, imputed silently
    df[FEATURES] = df[FEATURES].astype(float)                         # fails loudly if any text is left
    return df[["id", "name", "artist_id", "year", "era"] + FEATURES]


def make_split(df):
    df = df.copy().reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    _, test_idx = next(outer.split(df, df.era, groups=df.artist_id))  # fold 0 = test
    df["split"] = "train"
    df.loc[test_idx, "split"] = "test"
    df["fold"] = -1                                                   # -1 = test rows
    tr = df.index[df.split == "train"]
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (_, va) in enumerate(inner.split(df.loc[tr], df.loc[tr, "era"],
                                            groups=df.loc[tr, "artist_id"])):
        df.loc[tr[va], "fold"] = k
    assert set(df.artist_id[df.split == "train"]).isdisjoint(df.artist_id[df.split == "test"])
    return df


def split_hash(split_df):
    """Short fingerprint of the test track ids. Same hash = same test set (counts alone can match by chance)."""
    test_ids = split_df.loc[split_df.split == "test", "id"].astype(str)
    return hashlib.md5("".join(sorted(test_ids)).encode()).hexdigest()[:8]


def check_split(split_df):
    """Print the counts and test-set hash everyone compares on day 1. Identical output = identical split."""
    print("rows:", len(split_df))
    print("test hash:", split_hash(split_df))
    print(split_df.split.value_counts().to_string(), "\n")
    print(split_df.fold.value_counts().sort_index().to_string(), "\n")
    print(pd.crosstab(split_df.split, split_df.era.map(dict(zip(ERA_ORDER, ERA_LABELS))),
                      normalize="index")[ERA_LABELS].round(3).to_string())
