"""
Utils for training and applying Ternary Quality Filter for OCO-2 v11.2 Lite Files


Author: william.r.keely@jpl.nasa.gov

"""



import numpy as np
import pandas as pd
import collections

import os

import pickle
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KDTree



import seaborn as sns

# multi objective optimization
import optuna
import plotly.express as px
from pathlib import Path

import shap
# supress warnings
import warnings
warnings.filterwarnings("ignore")

def make_prediction(M, X, model, UQ=False, X_train=None, y_mean=0.0, y_std=1.0):
    """
    Predict bias with optional 1-σ proxy for Random-Forest models.

    For RF with UQ=True we build the (n_samples, n_trees) prediction
    matrix, take the 16th/84th percentiles across trees, and return
    the half-width (upper-lower) as a spread measure.
    """
    if model == "RF":
        if UQ:
            # stack predictions from every tree → shape (n_samples, n_trees)
            preds = np.column_stack([t.predict(X) for t in M.estimators_])
            bias  = preds.mean(axis=1)
            spread = np.quantile(preds, 0.84, axis=1) - np.quantile(preds, 0.16, axis=1)
            return bias, spread
        else:
            bias = M.predict(X)
            return bias, bias                              # dummy σ

    if model in {"XGB", "NN"}:
        bias = M.predict(X)
        return bias, bias

    if model == "GPR":
        if UQ:
            bias, sigma = M.predict(X, return_std=True)
            sigma = sigma * y_std + y_mean
        else:
            bias = np.concatenate([M.predict(X[i:i + 100_000])
                                    for i in range(0, len(X), 100_000)])
            sigma = bias
        return bias * y_std + y_mean, sigma

    if model in {"Ridge", "Ransac"}:
        bias = M.predict(X) * y_std + y_mean
        return bias, bias


    raise ValueError(f"Unknown model type: {model}")

def bias_correct(model_path, df, vars_to_fix, uq=False, proxy_name="TCCON"):
    """
    Apply the trained bias-correction model saved at *model_path* to *df*.
    Updates the columns in *vars_to_fix* in-place and (optionally) writes
    `bias_correction_uncert`.
    """
    meta = pd.read_pickle(model_path)
    X_mean, X_std = meta["X_mean"], meta["X_std"]
    y_mean, y_std = meta["y_mean"], meta["y_std"]
    model, feats, mdl_name = meta["TrainedModel"], meta["features"], meta["model"]

    X = df[feats]
    if mdl_name != "RF":                      # RF is already in native scale
        X = (X - X_mean) / X_std

    bias, sigma = make_prediction(model, X, mdl_name, UQ=uq, y_mean=y_mean, y_std=y_std)

    for v in vars_to_fix:
        df[v] = df[v] - bias

    if uq:
        df["bias_correction_uncert"] = sigma

    return df

def calc_SA_bias(xco2, sa_id):
    """
    Return (xco2 - SA median) with vectorised logic.
    Assumes *sa_id* is already sorted.
    """
    print("recalculating SA bias …")
    out = np.empty_like(xco2, dtype=float)
    idx_start = 0
    for sa, cnt in zip(*np.unique(sa_id, return_counts=True)):
        sl = slice(idx_start, idx_start + cnt)
        vals = xco2[sl]
        out[sl] = vals - np.median(vals) if cnt > 10 else np.nan
        idx_start += cnt
    return out


def train_rf(df, y, feats, *, cw=(1., 1.), w=None, rs=42):
    rf = RandomForestClassifier(
        n_estimators = 100,
        max_depth = 12,
        max_samples = len(df) // 2,
        min_samples_leaf = 10,
        max_features = "sqrt",
        class_weight = {0: cw[0], 1: cw[1]},
        n_jobs = -1,
        random_state= rs,
        verbose = 0,
    )
    rf.fit(df[feats], y, sample_weight=w)
    return rf


def sounding_fraction(df: pd.DataFrame, res_deg: int = 4) -> np.ndarray:
    """
    Fraction of soundings per equal-angle grid cell, scaled 0-100.
    """
    print("  • computing sounding-density grid …")
    lon, lat = df["longitude"].to_numpy(), df["latitude"].to_numpy()
    lon_e = np.arange(-180, 180 + res_deg, res_deg)
    lat_e = np.arange( -90,  90 + res_deg, res_deg)
    counts, _, _ = np.histogram2d(lon, lat, bins=[lon_e, lat_e])
    # look-up table → each sounding gets its cell count
    # (np.digitize gives 1-based indices)
    col = np.digitize(lon, lon_e) - 1
    row = np.digitize(lat, lat_e) - 1
    cell = counts.T[row, col]
    # convert to “% of mean monthly” like the original script
    n_month = len(df) / 12.0
    frac = (cell / n_month) * 100.0
    return frac / np.nanmax(frac) * 100     # normalise 0-100

def SA_label_density_weighting(df: pd.DataFrame,
                               gamma: float = 0.0,
                               res_deg: int = 4) -> pd.DataFrame:
    """
    Add column `SA_label_density_weighting` = |bias| * (1+γ·density).
    When γ==0 this collapses to |bias| 
    """
    if "sounding_fraction" not in df.columns:
        df["sounding_fraction"] = sounding_fraction(df, res_deg)
    df["SA_label_density_weighting"] = (
        np.abs(df["xco2raw_SA_bias"]) * (1.0 + gamma * df["sounding_fraction"])
    )
    return df


def tc_label(diff: pd.Series, pct: int) -> pd.Series:
    thr = np.nanpercentile(np.abs(diff), pct)
    return (np.abs(diff) > thr).astype(int)

def sa_binary_label(weighted_bias: pd.Series, pct: int) -> pd.Series:
    thr = np.nanpercentile(np.abs(weighted_bias), pct)
    return (np.abs(weighted_bias) > thr).astype(int)


def weight_tccon(df: pd.DataFrame) -> pd.DataFrame:
    cnt = df["tccon_name"].value_counts()
    df["weights"] = 1.0 / df["tccon_name"].map(cnt)
    df["weights"] /= df["weights"].sum()
    return df

# --------------------------------------------------------
def train_rf(df, y, feats, *, cw=(1.0, 1.0), w=None, rs=42):
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=12, max_samples=len(df)//2,
        min_samples_leaf=10, max_features="sqrt",
        class_weight={0: cw[0], 1: cw[1]},
        n_jobs=-1, random_state=rs, verbose=0,
    )
    rf.fit(df[feats], y, sample_weight=w)
    return rf


def build_ternary_filter(
        *,
        data_train          : pd.DataFrame,
        data_test           : pd.DataFrame,
        feats_small_area_all,          # (land_feats , ocean_feats)
        feats_tccon_all,               # (land_feats , ocean_feats)
        percentile_lnd      = 70,
        percentile_ocn      = 50,
        abstention_threshold_lnd = 1.25,
        abstention_threshold_ocn = 1.05,
        tccon_weighting     = True,    # bool or {"land":…, "ocean":…}
        class_weight        = None,
        gamma_sa            = 0.0,     # NEW  ← spatial weighting strength
        sa_res_deg          = 4,       # NEW  ← grid resolution for density
        save                = False,
):
    if isinstance(tccon_weighting, dict):
        wt_land  = bool(tccon_weighting.get("land" , False))
        wt_ocn   = bool(tccon_weighting.get("ocean", False))
    else:
        wt_land = wt_ocn = bool(tccon_weighting)

    if class_weight is None:
        class_weight = dict(tc_lnd=(1,1), tc_ocn=(1,1),
                            sa_lnd=(1,1), sa_ocn=(1,1))

    ft_tc_lnd, ft_tc_ocn = feats_tccon_all
    ft_sa_lnd, ft_sa_ocn = feats_small_area_all

    land_tr = data_train[data_train["land_fraction"] == 100].copy()
    ocn_tr  = data_train[data_train["land_fraction"] < 100].copy()

    land_tr = SA_label_density_weighting(land_tr, gamma_sa, sa_res_deg)
    ocn_tr  = SA_label_density_weighting(ocn_tr , gamma_sa, sa_res_deg)

    def _train_tc(df, feats, pct, cw, weight):
        df = df[df["xco2tccon"].notna()].copy()
        if weight: df = weight_tccon(df)
        y = tc_label(df["xco2MLcorr"] - df["xco2tccon"], pct)
        return train_rf(df, y, feats, cw=cw, w=df.get("weights"))

    M_tc_lnd = _train_tc(land_tr, ft_tc_lnd, percentile_lnd,
                         class_weight["tc_lnd"], wt_land)
    M_tc_ocn = _train_tc(ocn_tr , ft_tc_ocn, percentile_ocn,
                         class_weight["tc_ocn"], wt_ocn )

    def _train_sa(df, feats, pct, cw):
        df = df[np.isfinite(df["SA_label_density_weighting"])].copy()
        y  = sa_binary_label(df["SA_label_density_weighting"], pct)
        return train_rf(df, y, feats, cw=cw)

    M_sa_lnd = _train_sa(land_tr, ft_sa_lnd, percentile_lnd,
                         class_weight["sa_lnd"])
    M_sa_ocn = _train_sa(ocn_tr , ft_sa_ocn, percentile_ocn,
                         class_weight["sa_ocn"])

    def _apply(df):
        df = df.copy()
        is_land = df["land_fraction"] == 100

        df.loc[ is_land, "TCCON_flag"] = M_tc_lnd.predict(df.loc[ is_land, ft_tc_lnd])
        df.loc[~is_land, "TCCON_flag"] = M_tc_ocn.predict(df.loc[~is_land, ft_tc_ocn])

        df.loc[ is_land, "SA_flag"] = M_sa_lnd.predict(df.loc[ is_land, ft_sa_lnd])
        df.loc[~is_land, "SA_flag"] = M_sa_ocn.predict(df.loc[~is_land, ft_sa_ocn])

        σb = df["bias_correction_uncert"];  σr = df["xco2_uncertainty"]
        df["abst_metric"] = np.sqrt(np.maximum(σb**2 - σr**2, 0))
        df["abst_flag"]   = 1
        df.loc[ is_land & (df["abst_metric"] <= abstention_threshold_lnd), "abst_flag"] = 0
        df.loc[~is_land & (df["abst_metric"] <= abstention_threshold_ocn), "abst_flag"] = 0

        qf = np.full(len(df), 2, int)

        # QF = 0 
        cond_q0 = (df["TCCON_flag"]==0) & (df["SA_flag"]==0) & (df["abst_flag"]==0)
        qf[cond_q0] = 0

        # QF = 1
        cond_q1_land  =  is_land & (df["TCCON_flag"]==0) & ( (df["SA_flag"]==0) | (df["abst_flag"]==0) )
        cond_q1_ocn   = (~is_land) & ( (df["TCCON_flag"]==0) | (df["SA_flag"]==0) | (df["abst_flag"]==0) )
        qf[cond_q1_land | cond_q1_ocn] = np.minimum(qf[cond_q1_land | cond_q1_ocn], 1)

        df["xco2_quality_flag_b112"] = qf
        return df

    data_train = _apply(data_train)
    data_test  = _apply(data_test)

    if save:
        for tag, mdl in [("tc_lnd", M_tc_lnd), ("tc_ocn", M_tc_ocn),
                         ("sa_lnd", M_sa_lnd), ("sa_ocn", M_sa_ocn)]:
            with open(f"{tag}.p", "wb") as f:
                pickle.dump(mdl, f)

    return data_train, data_test, M_tc_lnd, M_tc_ocn, M_sa_lnd, M_sa_ocn




def load_one(fp: Path) -> pd.DataFrame:
    """Load pickle as *copy* so we never modify in-place."""
    return pd.read_pickle(fp).copy()


def split_tccon(df: pd.DataFrame, n_random: int = 1_000_000) -> pd.DataFrame:
    """All TCCON rows + random subset of non-TCCON."""
    tc   = df[df["xco2tccon"] > 0]
    rest = df[df["xco2tccon"].isna()].sample(n_random, random_state=42)
    return pd.concat([tc, rest], ignore_index=True)


def apply_bias(df: pd.DataFrame, *, land: bool) -> pd.DataFrame:
    """Run TCCON (uq=True) + SA corrections, add xco2MLcorr."""
    df = bias_correct(MODEL["TC_LND" if land else "TC_OCN"], df, ["xco2_raw"],
                      uq=True,  proxy_name="TCCON")
    df = bias_correct(MODEL["SA_LND" if land else "SA_OCN"], df, ["xco2_raw"],
                      uq=False, proxy_name="SA")
    df["xco2MLcorr"] = df["xco2_raw"]
    return df


def add_sa_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Median-remove XCO2 inside each small area (SA)."""
    df = df.sort_values("SA")
    df["xco2raw_SA_bias"] = calc_SA_bias(df["xco2MLcorr"].to_numpy(),
                                         df["SA"].to_numpy())
    return df



NEW_FLAG = "xco2_quality_flag_b112"      # ML ternary flag (0/1/2)
OLD_FLAG = "xco2_quality_flag"           # operational B11 flag
MODEL_COLS = slice("CT_2022+NRT2023-1", "MACC_v21r1")   # inclusive slice


def _rmse(a, b) -> float:
    return np.sqrt(mean_squared_error(a, b))

def _pct(num, den) -> float:
    return np.nan if den == 0 else 100.0 * num / den

def _round_dict(d: dict, ndig: int = 2) -> dict:
    """Round every float in *d* to *ndig* decimals (in place)."""
    for k, v in d.items():
        if isinstance(v, float):
            d[k] = np.round(v, ndig)
    return d

def collect_stats(df: pd.DataFrame, tag: str) -> dict:
    """Return a dict of metrics for *df* (land+ocean)."""
    stats = {"set": tag, "n": int(len(df))}          

    land = df["land_fraction"] == 100
    ocn  = ~land
    mlqf01 = df[NEW_FLAG].isin([0, 1])

    for v in (0, 1, 2):
        n = int((df[NEW_FLAG] == v).sum())
        stats[f"n_mlqf{v}"] = n
        stats[f"p_mlqf{v}"] = _pct(n, len(df))

    stats["n_mlqf01"] = int(mlqf01.sum())
    stats["p_mlqf01"] = _pct(stats["n_mlqf01"], len(df))

    n_qf0 = int((df[OLD_FLAG] == 0).sum())
    stats["n_qf0"] = n_qf0
    stats["p_qf0"]  = _pct(n_qf0, len(df))

    for surf, mask in [("lnd", land), ("ocn", ocn)]:
        denom = mask.sum()
        for v in (0, 1):
            stats[f"p_mlqf{v}_{surf}"]  = _pct(((df[NEW_FLAG] == v) & mask).sum(), denom)
        stats[f"p_mlqf01_{surf}"] = _pct((mlqf01 & mask).sum(),            denom)
        stats[f"p_qf0_{surf}"] = _pct(((df[OLD_FLAG]==0) & mask).sum(), denom)

    for surf, mask in [("", np.ones(len(df), bool)), ("_lnd", land), ("_ocn", ocn)]:
        msum = mask.sum()
        stats[f"p_tc_pass{surf}"] = _pct(((df["TCCON_flag"]==0)&mask).sum(), msum)
        stats[f"p_sa_pass{surf}"] = _pct(((df["SA_flag"]   ==0)&mask).sum(), msum)
        stats[f"p_uq_pass{surf}"] = _pct(((df["abst_flag"] ==0)&mask).sum(), msum)

    def _tc_rmse(mask):
        sub = df[mask & df["xco2tccon"].notna()]
        return np.nan if sub.empty else _rmse(sub["xco2MLcorr"], sub["xco2tccon"])

    stats["rmse_tccon_lnd_mlqf0"] = _tc_rmse((df[NEW_FLAG]==0)&land)
    stats["rmse_tccon_ocn_mlqf0"] = _tc_rmse((df[NEW_FLAG]==0)&ocn)
    stats["rmse_tccon_lnd_mlqf01"] = _tc_rmse(mlqf01 & land)
    stats["rmse_tccon_ocn_mlqf01"] = _tc_rmse(mlqf01 & ocn)
    stats["rmse_tccon_mlqf01"] = _tc_rmse(mlqf01)

    base = df[OLD_FLAG]==0
    stats["rmse_tccon_lnd_b111_mlbc"] = _tc_rmse(base & land)
    stats["rmse_tccon_ocn_b111_mlbc"] = _tc_rmse(base & ocn)

    sub = df[base & land & df["xco2tccon"].notna()]
    stats["rmse_tccon_lnd_b111_bc"]   = np.nan if sub.empty else _rmse(sub["xco2"], sub["xco2tccon"])
    sub = df[base & ocn  & df["xco2tccon"].notna()]
    stats["rmse_tccon_ocn_b111_bc"]   = np.nan if sub.empty else _rmse(sub["xco2"], sub["xco2tccon"])

    stats["stddev_sa_lnd"] = np.nanstd(df.loc[(df[NEW_FLAG]==0)&land,"xco2raw_SA_bias"])
    stats["stddev_sa_ocn"] = np.nanstd(df.loc[(df[NEW_FLAG]==0)&ocn ,"xco2raw_SA_bias"])
    stats["stddev_sa_lnd_b111"] = np.nanstd(df.loc[(df[OLD_FLAG]==0)&land,"xco2raw_SA_bias"])
    stats["stddev_sa_ocn_b111"] = np.nanstd(df.loc[(df[OLD_FLAG]==0)&ocn ,"xco2raw_SA_bias"])
    stats["stddev_sa_lnd_mlqf01"] = np.nanstd(df.loc[mlqf01 & land,"xco2raw_SA_bias"])
    stats["stddev_sa_ocn_mlqf01"] = np.nanstd(df.loc[mlqf01 & ocn ,"xco2raw_SA_bias"])

    model_mean = df.loc[:, MODEL_COLS].mean(axis=1)
    def _model_rmse(mask, use_ml=True):
        sub = df[mask & (model_mean > 0)]
        if sub.empty: return np.nan
        return _rmse(model_mean[sub.index],
                     sub["xco2MLcorr"] if use_ml else sub["xco2"])

    stats["rmse_models_lnd_mlqf0"] = _model_rmse((df[NEW_FLAG]==0)&land)
    stats["rmse_models_ocn_mlqf0"] = _model_rmse((df[NEW_FLAG]==0)&ocn )
    stats["rmse_models_lnd_qf0"] = _model_rmse((df[OLD_FLAG]==0)&land, use_ml=False)
    stats["rmse_models_ocn_qf0"] = _model_rmse((df[OLD_FLAG]==0)&ocn , use_ml=False)
    stats["rmse_models_lnd_qf0_mlbc"] = _model_rmse((df[OLD_FLAG]==0)&land)
    stats["rmse_models_ocn_qf0_mlbc"] = _model_rmse((df[OLD_FLAG]==0)&ocn )
    stats["rmse_models_lnd_mlqf01"] = _model_rmse(mlqf01 & land)
    stats["rmse_models_ocn_mlqf01"] = _model_rmse(mlqf01 & ocn )
    stats["rmse_models_mlqf01"] = _model_rmse(mlqf01)

    return _round_dict(stats, 2)     # <<–– round here!

def summary(train_df=None, test_df=None, *, pretty=True):
    """
    Compute & display (or return) the statistics table.

    """
    if train_df is None: train_df = globals()["data_train"]
    if test_df  is None: test_df  = globals()["data_test"]

    table = (pd.DataFrame([
                collect_stats(train_df, "train"),
                collect_stats(test_df,  "test")
             ])
             .set_index("set")
             .round(2)                
             .T)

    if pretty:
        display(table.style.format(precision=2))
        return None
    return table

""" CONSTRUCT TERNARY FLAG WITH PRE TRAINED MODELS """
def apply_ternary_flag(
        df                  : pd.DataFrame,
        *,
        model_dir           : str | Path = ".",
        feats_tc_lnd        : list[str],
        feats_tc_ocn        : list[str],
        feats_sa_lnd        : list[str],
        feats_sa_ocn        : list[str],
        abs_thresh_lnd      : float = 1.25,
        abs_thresh_ocn      : float = 1.05,
        tc_lnd_name         : str = "tc_lnd.p",
        tc_ocn_name         : str = "tc_ocn.p",
        sa_lnd_name         : str = "sa_lnd.p",
        sa_ocn_name         : str = "sa_ocn.p",
        out_qf_col          : str = "xco2_quality_flag_b112",
        keep_bits           : bool = True,
    ) -> pd.DataFrame:
    """
    Parameters:
    
    df : DataFrame
        Must contain *all* feature columns + ``bias_correction_uncert`` and
        ``xco2_uncertainty``.
    model_dir : str | Path
        Folder that holds the four ``*.p`` pickles made earlier.
    feats_*   : list[str]
        The exact feature lists used during training.
    abs_thresh_* : float
        Land / ocean abstention thresholds.

    Returns:
    
    df_out : DataFrame
        A **copy** of *df* with a new ternary flag column (and the three
        sub-filter bit columns if *keep_bits* is True).
    """
    from pathlib import Path
    model_dir = Path(model_dir)

    with open(model_dir / tc_lnd_name, "rb") as f:
        M_tc_lnd = pickle.load(f)
    with open(model_dir / tc_ocn_name, "rb") as f:
        M_tc_ocn = pickle.load(f)
    with open(model_dir / sa_lnd_name, "rb") as f:
        M_sa_lnd = pickle.load(f)
    with open(model_dir / sa_ocn_name, "rb") as f:
        M_sa_ocn = pickle.load(f)

    out      = df.copy()
    is_land  = out["land_fraction"] == 100
    out.loc[ is_land, "TCCON_flag"] = M_tc_lnd.predict(out.loc[ is_land, feats_tc_lnd])
    out.loc[~is_land, "TCCON_flag"] = M_tc_ocn.predict(out.loc[~is_land, feats_tc_ocn])

    out.loc[ is_land, "SA_flag"]    = M_sa_lnd.predict(out.loc[ is_land, feats_sa_lnd])
    out.loc[~is_land, "SA_flag"]    = M_sa_ocn.predict(out.loc[~is_land, feats_sa_ocn])

    σb = out["bias_correction_uncert"]
    σr = out["xco2_uncertainty"]
    out["abst_metric"] = np.sqrt(np.maximum(σb**2 - σr**2, 0.))
    out["abst_flag"]   = 1
    out.loc[ is_land & (out["abst_metric"] <= abs_thresh_lnd), "abst_flag"] = 0
    out.loc[~is_land & (out["abst_metric"] <= abs_thresh_ocn), "abst_flag"] = 0

    qf = np.full(len(out), 2, dtype=int)

    cond_q0 = (out["TCCON_flag"]==0) & (out["SA_flag"]==0) & (out["abst_flag"]==0)
    qf[cond_q0] = 0


    cond_q1_land =  is_land & (out["TCCON_flag"]==0) & ( (out["SA_flag"]==0) | (out["abst_flag"]==0) )
    cond_q1_ocn  = (~is_land) & ( (out["TCCON_flag"]==0) | (out["SA_flag"]==0) | (out["abst_flag"]==0) )
    qf[cond_q1_land | cond_q1_ocn] = 1   # any previously 0 stay 0

    out[out_qf_col] = qf

    if not keep_bits:
        out = out.drop(columns=["TCCON_flag", "SA_flag", "abst_metric", "abst_flag"])

    return out
