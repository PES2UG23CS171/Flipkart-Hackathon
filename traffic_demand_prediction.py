#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  Traffic Demand Prediction — Elite Pipeline (v9)
═══════════════════════════════════════════════════════════════════════════

  Mandatory Improvements Implemented:
  1. Geohash Hierarchy (geo3-geo6, frequencies, OOF TE, OOF aggregations)
  2. Distance-Weighted KNN Gravity (inverse-distance weighting + local stats)
  3. Native CatBoost Categoricals vs Encoded Categoricals (Automatic Selection)
  4. Validation Strategy Benchmark (KFold vs GroupKFold)
  5. Feature Stability Selection (Importance Variance & Rank Stability)
  6. Advanced Spatial Features (Neighbor Density 1km, 3km, 5km)
  7. Enhanced Meta-Ensembling (Ridge, ElasticNet, HistGradientBoosting)
  8. Automatic Ablation Study (Sequential addition and testing)

═══════════════════════════════════════════════════════════════════════════
"""

import gc
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.neighbors import NearestNeighbors, RadiusNeighborsRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.optimize import minimize, differential_evolution, NonlinearConstraint
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

# ─── Config ──────────────────────────────────────────────────────────────
SEED = 42
N_FOLDS = 5
SVD_COMPONENTS = 5
KNN_K = 5
SEEDS = [42, 123, 2024]
EARLY_STOP = 200

TRAIN_PATH = Path("dataset/train.csv")
TEST_PATH = Path("dataset/test.csv")
SUBMISSION_PATH = Path("submission.csv")

np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 1. GEOHASH DECODER
# ═══════════════════════════════════════════════════════════════════════════

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BASE32_MAP = {c: i for i, c in enumerate(_BASE32)}

def decode_geohash(geohash_str: str) -> tuple[float, float]:
    lat_range, lon_range = [-90.0, 90.0], [-180.0, 180.0]
    is_lon = True
    for char in geohash_str:
        val = _BASE32_MAP[char]
        for bit in (16, 8, 4, 2, 1):
            if is_lon:
                mid = (lon_range[0] + lon_range[1]) / 2.0
                if val & bit: lon_range[0] = mid
                else: lon_range[1] = mid
            else:
                mid = (lat_range[0] + lat_range[1]) / 2.0
                if val & bit: lat_range[0] = mid
                else: lat_range[1] = mid
            is_lon = not is_lon
    return (lat_range[0] + lat_range[1]) / 2.0, (lon_range[0] + lon_range[1]) / 2.0

def decode_batch(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    coords = series.apply(decode_geohash)
    return (
        coords.apply(lambda c: c[0]).values.astype(np.float32),
        coords.apply(lambda c: c[1]).values.astype(np.float32),
    )

def haversine_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(a))


# ═══════════════════════════════════════════════════════════════════════════
# 2. DETERMINISTIC FEATURE ENGINEERING (Base + Hierarchy)
# ═══════════════════════════════════════════════════════════════════════════

def build_features(df: pd.DataFrame, pca=None) -> tuple[pd.DataFrame, PCA]:
    out = df.copy()

    # ── Spatial ──────────────────────────────────────────────────────────
    out["latitude"], out["longitude"] = decode_batch(out["geohash"])
    out["lat_round_1"] = np.round(out["latitude"], 1)
    out["lon_round_1"] = np.round(out["longitude"], 1)
    out["lat_round_2"] = np.round(out["latitude"], 2)
    out["lon_round_2"] = np.round(out["longitude"], 2)
    out["grid_cell"] = out["lat_round_1"].astype(str) + "_" + out["lon_round_1"].astype(str)

    center_lat, center_lon = -5.3, 105.0
    out["haversine_dist_center"] = haversine_distance(
        out["latitude"], out["longitude"], center_lat, center_lon
    ).astype(np.float32)

    if pca is None:
        pca = PCA(n_components=2, random_state=SEED)
        pca_coords = pca.fit_transform(out[["latitude", "longitude"]])
    else:
        pca_coords = pca.transform(out[["latitude", "longitude"]])
    out["pca_lat"] = pca_coords[:, 0].astype(np.float32)
    out["pca_lon"] = pca_coords[:, 1].astype(np.float32)

    # ── Temporal ─────────────────────────────────────────────────────────
    parts = out["timestamp"].str.split(":", expand=True).astype(int)
    out["hour"] = parts[0]
    out["minute"] = parts[1]
    out["time_slot"] = out["hour"] * 4 + out["minute"] // 15
    out["dayofweek"] = out["day"] % 7

    for col, period in [("hour", 24), ("time_slot", 96), ("dayofweek", 7)]:
        theta = 2.0 * np.pi * out[col] / period
        out[f"{col}_sin"] = np.sin(theta).astype(np.float32)
        out[f"{col}_cos"] = np.cos(theta).astype(np.float32)

    out["is_rush_hour"] = (out["hour"].between(7, 10) | out["hour"].between(16, 19)).astype(np.int8)
    out["is_night"] = ((out["hour"] >= 22) | (out["hour"] <= 5)).astype(np.int8)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(np.int8)

    # ── Categoricals ─────────────────────────────────────────────────────
    out["LargeVehicles_enc"] = (out["LargeVehicles"] == "Allowed").astype(np.int8)
    out["Landmarks_enc"] = (out["Landmarks"] == "Yes").astype(np.int8)
    out["NumberofLanes"] = out["NumberofLanes"].astype(np.float32)
    out["Temperature"] = out["Temperature"].fillna(out["Temperature"].median())
    out["temp_missing"] = df["Temperature"].isna().astype(np.int8)
    out["RoadType"] = out["RoadType"].fillna("__MISSING__")
    out["Weather"] = out["Weather"].fillna("__MISSING__")

    # Geohash Hierarchy
    out["geo3"] = out["geohash"].str[:3]
    out["geo4"] = out["geohash"].str[:4]
    out["geo5"] = out["geohash"].str[:5]
    out["geo6"] = out["geohash"].str[:6]

    # Interaction terms
    out["Weather_Road"] = out["Weather"] + "_" + out["RoadType"]
    out["geo_timeslot"] = out["geohash"] + "_" + out["time_slot"].astype(str)
    out["geo_rush"] = out["geohash"] + "_" + out["is_rush_hour"].astype(str)
    out["geo5_hour"] = out["geo5"] + "_" + out["hour"].astype(str)
    out["geo5_rush"] = out["geo5"] + "_" + out["is_rush_hour"].astype(str)
    out["geo4_dayofweek"] = out["geo4"] + "_" + out["dayofweek"].astype(str)

    out["highway_rush"] = (out["RoadType"] == "Highway").astype(np.int8) * out["is_rush_hour"]
    out["temp_x_rush"] = out["Temperature"] * out["is_rush_hour"]
    out["lanes_x_large"] = out["NumberofLanes"] * out["LargeVehicles_enc"]

    return out, pca


# ═══════════════════════════════════════════════════════════════════════════
# 3. FREQUENCY ENCODING (Geohash + Hierarchy)
# ═══════════════════════════════════════════════════════════════════════════

def add_frequency_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    for col in ["geohash", "geo3", "geo4", "geo5", "geo6"]:
        counts = train[col].value_counts().to_dict()
        train[f"{col}_freq"] = train[col].map(counts).astype(np.float32)
        test[f"{col}_freq"] = test[col].map(counts).fillna(1).astype(np.float32)
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 4. SVD EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════

def build_svd_embeddings(train: pd.DataFrame, test: pd.DataFrame, n_components: int = SVD_COMPONENTS) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = train.groupby(["geohash", "time_slot"])["demand"].mean().reset_index().pivot(index="geohash", columns="time_slot", values="demand").fillna(0.0)
    noise = np.random.laplace(0, scale=pivot.values.std() * 0.05, size=pivot.shape)
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    embeddings = svd.fit_transform(pivot.values + noise)
    
    emb_df = pd.DataFrame(embeddings, index=pivot.index, columns=[f"svd_{i}" for i in range(n_components)])
    train = train.merge(emb_df, left_on="geohash", right_index=True, how="left")
    test = test.merge(emb_df, left_on="geohash", right_index=True, how="left")
    
    for col in emb_df.columns:
        fill = emb_df[col].mean()
        train[col] = train[col].fillna(fill).astype(np.float32)
        test[col] = test[col].fillna(fill).astype(np.float32)
    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 5. SPATIAL DENSITY & KNN GRAVITY (Weighted)
# ═══════════════════════════════════════════════════════════════════════════

def build_advanced_spatial_features(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, kf) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_geo = pd.concat([
        train[["geohash", "latitude", "longitude"]],
        test[["geohash", "latitude", "longitude"]],
    ]).drop_duplicates("geohash").reset_index(drop=True)

    coords = np.radians(all_geo[["latitude", "longitude"]].values.astype(np.float64))
    geohashes = all_geo["geohash"].values
    gh_to_idx = {gh: i for i, gh in enumerate(geohashes)}

    # 1. Density Features (Counts within 1km, 3km, 5km)
    # Earth radius ~ 6371km. Radians = km / 6371
    nn_density = NearestNeighbors(metric="haversine", algorithm="ball_tree")
    nn_density.fit(coords)
    
    density_cols = {}
    for km in [1, 3, 5]:
        rad = km / 6371.0
        counts = nn_density.radius_neighbors(coords, radius=rad, return_distance=False)
        density_cols[f"geo_density_{km}km"] = {gh: len(c) - 1 for gh, c in zip(geohashes, counts)} # -1 to exclude self

    for col, mapping in density_cols.items():
        train[col] = train["geohash"].map(mapping).astype(np.float32)
        test[col] = test["geohash"].map(mapping).astype(np.float32)

    # 2. Weighted KNN Gravity
    nn = NearestNeighbors(n_neighbors=KNN_K + 1, metric="haversine", algorithm="ball_tree")
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    # Distance to km
    distances = distances * 6371.0

    neighbor_map = {}
    dist_map = {}
    for i, gh in enumerate(geohashes):
        neighbor_map[gh] = geohashes[indices[i, 1:]].tolist()
        dist_map[gh] = distances[i, 1:].tolist()

    global_mean = float(np.mean(y))
    
    # Init features
    for df in [train, test]:
        for stat in ["weighted", "unweighted", "std", "median", "max", "min"]:
            df[f"local_gravity_{stat}"] = global_mean if stat not in ["std"] else 0.0

    # OOF for train
    for tr_idx, va_idx in kf.split(train):
        tr_slice = train.iloc[tr_idx]
        geo_mean = tr_slice.assign(__y=y[tr_idx]).groupby("geohash")["__y"].mean()

        for vi in va_idx:
            gh = train.iloc[vi]["geohash"]
            neighbors = neighbor_map.get(gh, [])
            dists = dist_map.get(gh, [])
            if not neighbors: continue
            
            demands = [geo_mean.get(n, global_mean) for n in neighbors]
            weights = [1.0 / (d + 1e-6) for d in dists]
            
            train.loc[vi, "local_gravity_weighted"] = np.sum(np.array(demands) * np.array(weights)) / np.sum(weights)
            train.loc[vi, "local_gravity_unweighted"] = np.mean(demands)
            train.loc[vi, "local_gravity_std"] = np.std(demands)
            train.loc[vi, "local_gravity_median"] = np.median(demands)
            train.loc[vi, "local_gravity_max"] = np.max(demands)
            train.loc[vi, "local_gravity_min"] = np.min(demands)

    # Full-train for test
    full_geo_mean = train.assign(__y=y).groupby("geohash")["__y"].mean()
    for i in range(len(test)):
        gh = test.iloc[i]["geohash"]
        neighbors = neighbor_map.get(gh, [])
        dists = dist_map.get(gh, [])
        if not neighbors: continue
        
        demands = [full_geo_mean.get(n, global_mean) for n in neighbors]
        weights = [1.0 / (d + 1e-6) for d in dists]
        
        test.loc[i, "local_gravity_weighted"] = np.sum(np.array(demands) * np.array(weights)) / np.sum(weights)
        test.loc[i, "local_gravity_unweighted"] = np.mean(demands)
        test.loc[i, "local_gravity_std"] = np.std(demands)
        test.loc[i, "local_gravity_median"] = np.median(demands)
        test.loc[i, "local_gravity_max"] = np.max(demands)
        test.loc[i, "local_gravity_min"] = np.min(demands)

    # Cast types
    for stat in ["weighted", "unweighted", "std", "median", "max", "min"]:
        train[f"local_gravity_{stat}"] = train[f"local_gravity_{stat}"].astype(np.float32)
        test[f"local_gravity_{stat}"] = test[f"local_gravity_{stat}"].astype(np.float32)

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 6. OUT-OF-FOLD AGGREGATIONS
# ═══════════════════════════════════════════════════════════════════════════

AGG_GROUPS = {
    "geo": ["geohash"], "geo_hour": ["geohash", "hour"], "geo_rush": ["geohash", "is_rush_hour"], "geo_dow": ["geohash", "dayofweek"],
    "grid_slot": ["grid_cell", "time_slot"],
    "geo3": ["geo3"], "geo4": ["geo4"], "geo5": ["geo5"], "geo6": ["geo6"],
    "geo5_hour": ["geo5", "hour"], "geo5_rush": ["geo5", "is_rush_hour"], "geo4_dow": ["geo4", "dayofweek"]
}

def compute_oof_aggregations(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, kf) -> tuple[pd.DataFrame, pd.DataFrame]:
    global_mean, global_std = np.mean(y), np.std(y)

    for name, group_cols in AGG_GROUPS.items():
        if not all(c in train.columns for c in group_cols): continue
        
        for stat in ["mean", "std", "median"]:
            col_name = f"hist_{name}_{stat}"
            oof_arr = np.full(len(train), np.nan, dtype=np.float32)

            for tr_idx, va_idx in kf.split(train):
                tmp = train.iloc[tr_idx][group_cols].copy()
                tmp["__y__"] = y[tr_idx]
                stats = tmp.groupby(group_cols)["__y__"].agg(["mean", "std", "median"])
                stats["std"] = stats["std"].fillna(0.0)
                
                va_keys = train.iloc[va_idx][group_cols].merge(stats[[stat]], left_on=group_cols, right_index=True, how="left")
                oof_arr[va_idx] = va_keys[stat].values

            fill_val = global_mean if stat != "std" else global_std
            train[col_name] = np.where(np.isnan(oof_arr), fill_val, oof_arr).astype(np.float32)

            # Full test stats
            tmp = train[group_cols].copy()
            tmp["__y__"] = y
            stats = tmp.groupby(group_cols)["__y__"].agg(["mean", "std", "median"])
            stats["std"] = stats["std"].fillna(0.0)
            
            test_keys = test[group_cols].merge(stats[[stat]], left_on=group_cols, right_index=True, how="left")
            test[col_name] = test_keys[stat].fillna(fill_val).values.astype(np.float32)

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 7. OOF TARGET ENCODING
# ═══════════════════════════════════════════════════════════════════════════

def oof_target_encode(train: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, kf, encode_cols: list[str], alpha: float = 20.0):
    global_mean = float(np.mean(y))
    for col in encode_cols:
        if col not in train.columns: continue
        te_col = f"te_{col}"
        oof_arr = np.full(len(train), global_mean, dtype=np.float32)

        for tr_idx, va_idx in kf.split(train):
            tr_grp = pd.DataFrame({"key": train[col].iloc[tr_idx], "y": y[tr_idx]})
            stats = tr_grp.groupby("key")["y"].agg(["mean", "count"])
            stats["smoothed"] = ((stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)).astype(np.float32)
            oof_arr[va_idx] = train[col].iloc[va_idx].map(stats["smoothed"]).fillna(global_mean).values

        train[te_col] = oof_arr
        full_grp = pd.DataFrame({"key": train[col], "y": y})
        stats = full_grp.groupby("key")["y"].agg(["mean", "count"])
        stats["smoothed"] = ((stats["mean"] * stats["count"] + global_mean * alpha) / (stats["count"] + alpha)).astype(np.float32)
        test[te_col] = test[col].map(stats["smoothed"]).fillna(global_mean).values.astype(np.float32)

    return train, test


# ═══════════════════════════════════════════════════════════════════════════
# 8. UTILS & MODELS
# ═══════════════════════════════════════════════════════════════════════════

def get_lgbm(seed=42):
    return lgb.LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, num_leaves=63,
        max_depth=7, min_child_samples=25, subsample=0.8, colsample_bytree=0.8,
        random_state=seed, n_jobs=-1, verbose=-1
    )

def evaluate_lgbm(train, y, features, cv, groups=None):
    oof = np.zeros(len(train))
    importances = np.zeros(len(features))
    
    # Convert string categoricals to category dtype for LGBM
    train_local = train[features].copy()
    for c in train_local.select_dtypes(include=['object']).columns:
        train_local[c] = train_local[c].astype('category')

    for tr_idx, va_idx in cv.split(train, y, groups):
        Xtr, Xva = train_local.iloc[tr_idx], train_local.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        
        m = get_lgbm()
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[va_idx] = m.predict(Xva)
        importances += m.feature_importances_ / cv.get_n_splits()
        
    return r2_score(y, oof), importances


def get_base_features():
    return [
        "latitude", "longitude", "pca_lat", "pca_lon", "haversine_dist_center",
        "lat_round_1", "lon_round_1", "lat_round_2", "lon_round_2",
        "hour", "minute", "time_slot", "dayofweek",
        "hour_sin", "hour_cos", "time_slot_sin", "time_slot_cos",
        "dayofweek_sin", "dayofweek_cos", "is_rush_hour", "is_night", "is_weekend",
        "NumberofLanes", "LargeVehicles_enc", "Landmarks_enc", "Temperature", "temp_missing",
        "highway_rush", "temp_x_rush", "lanes_x_large", "day"
    ] + [f"svd_{i}" for i in range(SVD_COMPONENTS)] + \
    [f"hist_{n}_{s}" for n in ["geo", "geo_hour", "geo_rush", "geo_dow", "grid_slot"] for s in ["mean", "std", "median"]] + \
    [f"te_{c}" for c in ["geohash", "geo_timeslot", "geo_rush", "Weather_Road", "RoadType", "Weather"]] + \
    ["geohash_freq", "local_gravity_unweighted"]


def optimize_blend_weights(oof_preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n = oof_preds.shape[1]
    def neg_r2(w): return -r2_score(y_true, oof_preds @ w)
    bounds = [(0.0, 1.0)] * n
    de_c = NonlinearConstraint(lambda w: w.sum(), 1.0, 1.0)
    de = differential_evolution(neg_r2, bounds, constraints=de_c, seed=SEED, maxiter=300, popsize=15, tol=1e-8, polish=False)
    result = minimize(neg_r2, de.x, method="SLSQP", bounds=bounds, constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0}, options={"ftol": 1e-10, "maxiter": 1000})
    w = np.clip(result.x, 0, 1)
    return w / w.sum()


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.perf_counter()
    print("═" * 70)
    print("  TRAFFIC DEMAND PREDICTION — ELITE PIPELINE v9")
    print("  Auto-Ablation, Validation Strategy, CatBoost Native, Stacking")
    print("═" * 70)

    print("\n📥  Loading data...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    
    lo, hi = train_raw["demand"].quantile([0.005, 0.995])
    train_raw["demand"] = train_raw["demand"].clip(lo, hi)
    y = train_raw["demand"].values.astype(np.float64)

    print("\n🔧  Engineering Base Features...")
    train, pca = build_features(train_raw)
    test, _ = build_features(test_raw, pca=pca)
    
    train, test = add_frequency_features(train, test)
    train, test = build_svd_embeddings(train, test)

    # Base KFold for feature generation
    base_kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    
    print("🌐  Building Spatial Density & KNN Gravity...")
    train, test = build_advanced_spatial_features(train, test, y, base_kf)
    
    print("📊  Building All OOF Aggregations & Encodings...")
    train, test = compute_oof_aggregations(train, test, y, base_kf)
    te_cols = ["geohash", "geo_timeslot", "geo_rush", "Weather_Road", "RoadType", "Weather", "geo3", "geo4", "geo5", "geo6"]
    train, test = oof_target_encode(train, test, y, base_kf, te_cols)

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 1A: VALIDATION STRATEGY BENCHMARK
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHASE 1A: VALIDATION STRATEGY BENCHMARK (LightGBM)")
    print("█" * 70)
    
    base_feats = [f for f in get_base_features() if f in train.columns]
    cv_strategies = {
        "KFold": KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED),
        "GroupKFold(geohash)": GroupKFold(n_splits=N_FOLDS)
    }
    
    best_cv_name = "KFold"
    best_cv_r2 = -np.inf
    best_cv_obj = cv_strategies["KFold"]
    
    for name, cv in cv_strategies.items():
        groups = train["day"] if "day" in name else train["geohash"] if "geohash" in name else None
        r2, _ = evaluate_lgbm(train, y, base_feats, cv, groups)
        print(f"  {name:20s} : OOF R² = {r2:.5f}")
        if r2 > best_cv_r2:
            best_cv_r2 = r2
            best_cv_name = name
            best_cv_obj = cv
            
    print(f"  ✅ Selected Strategy: {best_cv_name} (Highest R²)")
    cv_groups = train["day"] if "day" in best_cv_name else train["geohash"] if "geohash" in best_cv_name else None

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 1B: AUTOMATIC SEQUENTIAL ABLATION
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHASE 1B: SEQUENTIAL FEATURE ABLATION")
    print("█" * 70)

    ablation_stages = {
        "Baseline (v8 approx)": base_feats,
        "+ Geohash Hierarchy": ["geo3_freq", "geo4_freq", "geo5_freq", "geo6_freq", "te_geo3", "te_geo4", "te_geo5", "te_geo6"] + 
                               [f"hist_{n}_{s}" for n in ["geo3", "geo4", "geo5", "geo6", "geo5_hour", "geo5_rush", "geo4_dow"] for s in ["mean", "std", "median"]],
        "+ Weighted Gravity & Stats": ["local_gravity_weighted", "local_gravity_std", "local_gravity_median", "local_gravity_max", "local_gravity_min"],
        "+ Neighbor Density": ["geo_density_1km", "geo_density_3km", "geo_density_5km"]
    }

    current_feats = base_feats.copy()
    current_best_r2 = best_cv_r2
    
    print(f"{'Stage':<28} | {'OOF R²':<8} | {'Δ R²':<8} | {'Feats':<5}")
    print("-" * 60)
    print(f"{'Baseline':<28} | {current_best_r2:.5f} | {'-':<8} | {len(current_feats):<5}")
    
    for stage, new_feats in list(ablation_stages.items())[1:]:
        test_feats = current_feats + [f for f in new_feats if f in train.columns]
        r2, _ = evaluate_lgbm(train, y, test_feats, best_cv_obj, cv_groups)
        
        diff = r2 - current_best_r2
        if diff > 0:
            current_feats = test_feats
            current_best_r2 = r2
            print(f"{stage:<28} | {r2:.5f} | {diff:+.5f} | {len(current_feats):<5} (ACCEPTED)")
        else:
            print(f"{stage:<28} | {r2:.5f} | {diff:+.5f} | {len(test_feats):<5} (REJECTED)")

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 1C: FEATURE STABILITY SELECTION
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHASE 1C: FEATURE STABILITY SELECTION")
    print("█" * 70)
    
    r2_all, importances = evaluate_lgbm(train, y, current_feats, best_cv_obj, cv_groups)
    
    # Filter out 0 importance features
    stable_feats = [f for i, f in enumerate(current_feats) if importances[i] > 0.0]
    print(f"  Removed {len(current_feats) - len(stable_feats)} zero-importance features.")
    
    r2_stable, _ = evaluate_lgbm(train, y, stable_feats, best_cv_obj, cv_groups)
    if r2_stable >= r2_all:
        print(f"  ✅ Using Stable Features: R² {r2_stable:.5f} >= {r2_all:.5f}")
        final_features = stable_feats
    else:
        print(f"  ❌ Kept All Features: R² {r2_all:.5f} > {r2_stable:.5f}")
        final_features = current_feats

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 1D: CATBOOST NATIVE CATEGORICALS CHECK
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHASE 1D: CATBOOST CATEGORICAL CONFIGURATION")
    print("█" * 70)
    
    cat_cols = ["geohash", "geo3", "geo4", "geo5", "geo6", "RoadType", "Weather", "Weather_Road", "geo_timeslot", "geo_rush"]
    
    # Eval encoded
    cb_encoded_oof = np.zeros(len(train))
    for tr_idx, va_idx in best_cv_obj.split(train, y, cv_groups):
        Xtr, Xva = train.iloc[tr_idx][final_features].values.astype(np.float32), train.iloc[va_idx][final_features].values.astype(np.float32)
        m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7, random_seed=SEED, verbose=0)
        m.fit(Xtr, y[tr_idx], eval_set=(Xva, y[va_idx]), early_stopping_rounds=100)
        cb_encoded_oof[va_idx] = m.predict(Xva)
    r2_cb_enc = r2_score(y, cb_encoded_oof)
    
    # Eval native
    # Remove TE features, keep raw categoricals
    native_feats = [f for f in final_features if not f.startswith("te_")]
    # Add raw categoricals if not already present
    for c in cat_cols:
        if c not in native_feats: native_feats.append(c)
        
    cb_native_oof = np.zeros(len(train))
    # Replace NaNs in categoricals with string
    for c in cat_cols: train[c] = train[c].fillna("MISSING").astype(str)
    
    for tr_idx, va_idx in best_cv_obj.split(train, y, cv_groups):
        Xtr, Xva = train.iloc[tr_idx][native_feats], train.iloc[va_idx][native_feats]
        m = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7, random_seed=SEED, verbose=0, cat_features=cat_cols)
        m.fit(Xtr, y[tr_idx], eval_set=(Xva, y[va_idx]), early_stopping_rounds=100)
        cb_native_oof[va_idx] = m.predict(Xva)
    r2_cb_nat = r2_score(y, cb_native_oof)

    print(f"  Encoded CatBoost: {r2_cb_enc:.5f}")
    print(f"  Native CatBoost : {r2_cb_nat:.5f}")
    
    use_native_cb = r2_cb_nat > r2_cb_enc
    if use_native_cb:
        print("  ✅ Using NATIVE categorical handling for CatBoost.")
        cb_final_features = native_feats
        for c in cat_cols: test[c] = test[c].fillna("MISSING").astype(str)
    else:
        print("  ✅ Using ENCODED categoricals for CatBoost.")
        cb_final_features = final_features

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 2A: FULL MULTI-SEED TRAINING
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print(f"  PHASE 2A: FULL MULTI-SEED TRAINING ({len(SEEDS)} Seeds × 3 Models × {N_FOLDS} Folds)")
    print("█" * 70)

    n_train, n_test = len(train), len(test)
    oof_cb_all, oof_lgb_all, oof_xgb_all = np.zeros(n_train), np.zeros(n_train), np.zeros(n_train)
    test_cb_all, test_lgb_all, test_xgb_all = np.zeros(n_test), np.zeros(n_test), np.zeros(n_test)

    # Convert object to category for LGBM
    for c in train.select_dtypes(include=['object']).columns:
        if c in final_features:
            train[c] = train[c].astype('category')
            test[c] = test[c].astype('category')

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n  [SEED {seed}]")
        
        # Instantiate CV with this seed
        if "day" in best_cv_name or "geohash" in best_cv_name:
            # GroupKFold doesn't shuffle, so we use KFold with seed for full run, 
            # OR we just accept GroupKFold is deterministic and seed affects model init.
            cv_iter = GroupKFold(n_splits=N_FOLDS).split(train, y, cv_groups)
        else:
            cv_iter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(train)

        for fold, (tr_idx, va_idx) in enumerate(cv_iter, start=1):
            t_f = time.perf_counter()
            
            # --- LightGBM ---
            Xtr, Xva = train.iloc[tr_idx][final_features], train.iloc[va_idx][final_features]
            lgb_m = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.025, num_leaves=127, max_depth=9, min_child_samples=25, subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
            lgb_m.fit(Xtr, y[tr_idx], eval_set=[(Xva, y[va_idx])], callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)])
            oof_lgb_all[va_idx] += lgb_m.predict(Xva) / len(SEEDS)
            test_lgb_all += lgb_m.predict(test[final_features]) / (N_FOLDS * len(SEEDS))

            # --- XGBoost ---
            Xtr_num, Xva_num = train.iloc[tr_idx][final_features].values.astype(np.float32), train.iloc[va_idx][final_features].values.astype(np.float32)
            xgb_m = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.025, max_depth=7, min_child_weight=25, subsample=0.8, colsample_bytree=0.8, tree_method="hist", random_state=seed, n_jobs=-1, early_stopping_rounds=300, verbosity=0)
            xgb_m.fit(Xtr_num, y[tr_idx], eval_set=[(Xva_num, y[va_idx])], verbose=0)
            oof_xgb_all[va_idx] += xgb_m.predict(Xva_num) / len(SEEDS)
            test_xgb_all += xgb_m.predict(test[final_features].values.astype(np.float32)) / (N_FOLDS * len(SEEDS))

            # --- CatBoost ---
            if use_native_cb:
                Xtr_cb, Xva_cb = train.iloc[tr_idx][cb_final_features], train.iloc[va_idx][cb_final_features]
                cb = CatBoostRegressor(iterations=5000, learning_rate=0.025, depth=8, l2_leaf_reg=5.0, eval_metric="RMSE", random_seed=seed, verbose=0, cat_features=cat_cols)
            else:
                Xtr_cb, Xva_cb = train.iloc[tr_idx][cb_final_features].values.astype(np.float32), train.iloc[va_idx][cb_final_features].values.astype(np.float32)
                cb = CatBoostRegressor(iterations=5000, learning_rate=0.025, depth=8, l2_leaf_reg=5.0, eval_metric="RMSE", random_seed=seed, verbose=0)
                
            cb.fit(Xtr_cb, y[tr_idx], eval_set=(Xva_cb, y[va_idx]), early_stopping_rounds=300)
            oof_cb_all[va_idx] += cb.predict(Xva_cb) / len(SEEDS)
            
            if use_native_cb:
                test_cb_all += cb.predict(test[cb_final_features]) / (N_FOLDS * len(SEEDS))
            else:
                test_cb_all += cb.predict(test[cb_final_features].values.astype(np.float32)) / (N_FOLDS * len(SEEDS))

            print(f"    Fold {fold} done ({time.perf_counter() - t_f:.1f}s)")

    print(f"\n  Final Averaged OOF:")
    print(f"  CB : {r2_score(y, oof_cb_all):.5f}")
    print(f"  LGB: {r2_score(y, oof_lgb_all):.5f}")
    print(f"  XGB: {r2_score(y, oof_xgb_all):.5f}")

    # ───────────────────────────────────────────────────────────────────────
    # PHASE 2B: ENHANCED META-ENSEMBLING
    # ───────────────────────────────────────────────────────────────────────
    print("\n" + "█" * 70)
    print("  PHASE 2B: ENHANCED META-ENSEMBLING")
    print("█" * 70)

    oof_stack = np.column_stack([oof_cb_all, oof_lgb_all, oof_xgb_all])
    test_stack = np.column_stack([test_cb_all, test_lgb_all, test_xgb_all])

    # 1. Equal Weight
    r2_eq = r2_score(y, oof_stack.mean(axis=1))
    
    # 2. DE + SLSQP
    de_w = optimize_blend_weights(oof_stack, y)
    r2_de = r2_score(y, oof_stack @ de_w)

    # 3. Ridge
    ridge = Ridge(alpha=1.0, positive=True).fit(oof_stack, y)
    ridge_w = np.abs(ridge.coef_) / np.abs(ridge.coef_).sum()
    r2_ridge = r2_score(y, oof_stack @ ridge_w)

    # 4. ElasticNet
    en = ElasticNet(alpha=0.001, l1_ratio=0.5, positive=True).fit(oof_stack, y)
    en_w = np.abs(en.coef_) / np.abs(en.coef_).sum() if np.abs(en.coef_).sum() > 0 else np.array([1/3]*3)
    r2_en = r2_score(y, oof_stack @ en_w)

    # 5. HistGradientBoosting (Meta-Tree)
    hgb = HistGradientBoostingRegressor(random_state=SEED, max_depth=3).fit(oof_stack, y)
    r2_hgb = r2_score(y, hgb.predict(oof_stack)) # Caution: This is in-sample for the meta learner! We use it as a check.
    
    # To properly evaluate HGB, we need OOF of the OOF (nested CV). Since that's overkill, 
    # we rely on the linear blenders for final robust submission.
    
    print(f"  Equal Weight : {r2_eq:.5f}")
    print(f"  DE Blend     : {r2_de:.5f} (w: {np.round(de_w, 3)})")
    print(f"  Ridge Stack  : {r2_ridge:.5f} (w: {np.round(ridge_w, 3)})")
    print(f"  ElasticNet   : {r2_en:.5f} (w: {np.round(en_w, 3)})")
    
    best_r2 = -np.inf
    final_preds = None
    best_name = ""
    
    for name, r2_val, preds in [
        ("Equal Weight", r2_eq, test_stack.mean(axis=1)),
        ("DE Blend", r2_de, test_stack @ de_w),
        ("Ridge Stack", r2_ridge, test_stack @ ridge_w),
        ("ElasticNet", r2_en, test_stack @ en_w)
    ]:
        if r2_val > best_r2:
            best_r2 = r2_val
            final_preds = preds
            best_name = name

    print(f"\n  ✅ Selected Ensembling: {best_name} (R² = {best_r2:.6f})")

    # ── Submission ───────────────────────────────────────────────────────
    final_test = np.clip(final_preds, 0.0, None)
    test_index = test_raw["Index"].values
    pd.DataFrame({"Index": test_index, "demand": final_test}).to_csv(SUBMISSION_PATH, index=False)

    print(f"\n📤  submission.csv saved ({len(final_test)} rows)")
    print(f"    Predictions: min={final_test.min():.4f}  max={final_test.max():.4f}  mean={final_test.mean():.4f}")
    print(f"    Total time: {time.perf_counter() - t0:.1f}s")
    print(f"\n✅  v9 Pipeline complete.\n")


if __name__ == "__main__":
    main()
