#!/usr/bin/env python
# Multi-objective optimisation of the ternary filter (land & ocean)
# Author: william.r.keely@jpl.nasa.gov

import argparse, warnings, optuna, pickle, os
from pathlib import Path
import pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category = FutureWarning)

DATA = dict(
    LND = Path("/path/to/LAND"),   # ← edit
    OCN = Path("/path/to/OCEAN"),  # ← edit
)
FRAC_DS = 0.10                    # down-sample fraction when caching

_CACHE = Path("_cache")
_CACHE.mkdir(exist_ok = True)

_TEMPLATE = dict(
    land  = "PreLoadB112_ununbalanced_5M_LndNDGL_qfNone_{Y}.pkl",
    ocean = "PreLoadB112_ununbalanced_5M_SeaGL_qfNone_{Y}.pkl",
)

def _raw_path(surface: str, year: int) -> Path:
    key = "LND" if surface == "land" else "OCN"
    return DATA[key] / _TEMPLATE[surface].format(Y = year)

def _cache_path(surface: str, year: int) -> Path:
    return _CACHE / f"{surface}_{year}.parquet"

def load_one(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path).copy()

def load_year(surface: str, year: int, *, frac: float = FRAC_DS, seed: int = 42) -> pd.DataFrame:
    cpath = _cache_path(surface, year)
    if cpath.exists():
        return pd.read_parquet(cpath)
    df = load_one(_raw_path(surface, year))
    if frac < 1.0:
        df = df.sample(frac = frac, random_state = seed)
    df.to_parquet(cpath, compression = "snappy", index = False)
    return df

def train_rf(df, y, feats, *, cw = (1, 1), w = None, n_estimators = 200, max_depth = 12):
    rf = RandomForestClassifier(
        n_estimators = n_estimators,
        max_depth = max_depth,
        max_samples = len(df) // 2,
        min_samples_leaf = 10,
        max_features = "sqrt",
        class_weight = {0: cw[0], 1: cw[1]},
        n_jobs = -1,
        random_state = 42,
    )
    rf.fit(df[feats], y, sample_weight = w)
    return rf

def tc_label(s, pct):
    thr = np.nanpercentile(np.abs(s), pct)
    return (np.abs(s) > thr).astype(int)

def sa_label(s, pct):
    thr = np.nanpercentile(np.abs(s), pct)
    return (np.abs(s) > thr).astype(int)

def weight_tccon(df):
    cnt = df["tccon_name"].value_counts()
    df["weights"] = 1.0 / df["tccon_name"].map(cnt)
    df["weights"] /= df["weights"].sum()
    return df

def sounding_fraction(df, res = 4):
    lat = df["latitude"].values
    lon = df["longitude"].values
    bins_lat = np.arange(-90, 90, res)
    bins_lon = np.arange(-180, 180, res)
    idx_lat = np.digitize(lat, bins_lat) - 1
    idx_lon = np.digitize(lon, bins_lon) - 1
    grid = np.zeros((len(bins_lat), len(bins_lon)))
    for i, j in zip(idx_lat, idx_lon):
        grid[i, j] += 1
    counts = grid[idx_lat, idx_lon]
    return counts / counts.max() * 100

def SA_label_density_weighting(df, gamma = 0.0, res = 4):
    if "sounding_fraction" not in df.columns:
        df["sounding_fraction"] = sounding_fraction(df, res = res)
    df["SA_label_density_weighting"] = np.abs(df["xco2raw_SA_bias"]) * (1 + gamma * df["sounding_fraction"])
    return df

FEATS_LAND = ["albedo_sco2", "rms_rel_sco2", "bias_correction_uncert"]
FEATS_OCEAN = ["albedo_wco2", "rms_rel_wco2", "bias_correction_uncert"]

def _trial_surface(trial, surface):
    yrs_train = [2015, 2016]
    year_test = 2021

    train = pd.concat([load_year(surface, y) for y in yrs_train], ignore_index = True)
    test = load_year(surface, year_test)

    pct = trial.suggest_int("pct", 40, 80, 5)
    cw0 = trial.suggest_float("cw0", 0.8, 1.3, 0.1)
    cw1 = trial.suggest_float("cw1", 0.8, 1.3, 0.1)
    n_est = trial.suggest_int("n_est", 120, 400, 40)
    max_d = trial.suggest_int("max_d", 10, 28, 2)
    gamma = trial.suggest_float("gamma", 0.0, 5.0, 0.5)
    abs_th = trial.suggest_float("abs_th", 0.8, 1.8, 0.05)

    feats = FEATS_LAND if surface == "land" else FEATS_OCEAN
    train = SA_label_density_weighting(train, gamma = gamma, res = 4)

    tc = weight_tccon(train[train["xco2tccon"] > 0].copy())
    y_tc = tc_label(tc["xco2MLcorr"] - tc["xco2tccon"], pct)

    sa_metric = train["SA_label_density_weighting"] if surface == "land" else train["xco2raw_SA_bias"]
    y_sa = sa_label(sa_metric, pct)

    rf_tc = train_rf(tc, y_tc, feats, cw = (cw0, cw1), w = tc["weights"], n_estimators = n_est, max_depth = max_d)
    rf_sa = train_rf(train, y_sa, feats, cw = (cw0, cw1), n_estimators = n_est, max_depth = max_d)

    tc_pred = rf_tc.predict(test[feats])
    sa_pred = rf_sa.predict(test[feats])

    σb = test["bias_correction_uncert"]
    σr = test["xco2_uncertainty"]
    abst_flag = (np.sqrt(np.maximum(σb ** 2 - σr ** 2, 0.0)) > abs_th).astype(int)

    qf = np.full(len(test), 2, int)
    qf[(tc_pred == 0) & (sa_pred == 0) & (abst_flag == 0)] = 0

    if surface == "land":
        cond = (tc_pred == 0) & ((sa_pred == 0) | (abst_flag == 0))
    else:
        cond = (tc_pred == 0) | (sa_pred == 0) | (abst_flag == 0)
    qf[cond] = np.minimum(qf[cond], 1)
    test["new_qf"] = qf

    rmse = np.sqrt(np.nanmean((test.loc[qf == 0, "xco2MLcorr"] - test.loc[qf == 0, "xco2_model"]) ** 2))
    throughput = (qf == 0).mean() * 100
    sigma = test.loc[qf == 0, "bias_correction_uncert"].std()
    return rmse, -throughput, sigma

def land_trial(t): return _trial_surface(t, "land")
def ocean_trial(t): return _trial_surface(t, "ocean")

def parse_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--n_trials", type = int, default = 25)
    p.add_argument("--land", action = "store_true")
    p.add_argument("--ocean", action = "store_true")
    return p.parse_args()

def _run_surface(obj_fn, n_trials, label):
    study = optuna.create_study(
        directions = ["minimize", "minimize", "minimize"],
        sampler = optuna.samplers.MOTPESampler(seed = 11),
    )
    study.optimize(obj_fn, n_trials = n_trials)
    best = sorted(study.best_trials, key = lambda tr: tr.number)[:10]
    print(f"\n--- {label} best (top10) ---")
    for tr in best:
        r, tp, s = tr.values
        print(f"#{tr.number:3d} RMSE = {r:.2f} TP = {-tp:.1f}% σ = {s:.2f}")
    df = pd.DataFrame([tr.values for tr in study.trials], columns = ["RMSE", "-TP", "σ"])
    ax = df.plot.scatter("RMSE", "σ", c = "-TP", cmap = "viridis", figsize = (5, 4))
    ax.set_title(label)
    ax.set_xlabel("RMSE")
    ax.set_ylabel("σ")
    ax.figure.tight_layout()
    ax.figure.savefig(f"pareto_{label.lower()}.png", dpi = 200)

def main():
    args = parse_cli()
    if not (args.land or args.ocean):
        args.land = args.ocean = True
    if args.land:
        _run_surface(land_trial, args.n_trials, "LAND")
    if args.ocean:
        _run_surface(ocean_trial, args.n_trials, "OCEAN")

if __name__ == "__main__":
    main()
