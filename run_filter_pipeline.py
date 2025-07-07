#!/usr/bin/env python

"""
run_quality_filter_pipeline.py
──────────────────────────────
1. (optional) run the bias-correction pre-processing pipeline
2. (optional) run optimise_filter.py (multi-objective search)
3. assemble train / test sets, train the ternary filter, apply to
   the full record and save results to parquet

USAGE
─────
python run_quality_filter_pipeline.py            # full run
python run_quality_filter_pipeline.py --skip-bias
python run_quality_filter_pipeline.py --no-opt
python run_quality_filter_pipeline.py --clean-status


author: william.r.keely@jpl.nasa.gov
"""


import subprocess, sys, os, argparse, json
from pathlib import Path

import filter_utils as fu                            # noqa

BIAS_SCRIPT = Path("run_bias_correction_pipeline.py")
OPT_SCRIPT  = Path("optimize_filter.py")
STATUS_FILE = Path("quality_pipeline_status.txt")

# pickled / parquet artefacts written by optimise_filter.py
BEST_PARAM_FILE = Path("_cache/opt_best.json")        # simple JSON dump

TRAIN_YEARS = [2015, 2016, 2017, 2018, 2019, 2020]
TEST_YEAR   = 2021
OUT_PARQUET = Path("quality_flagged_record.parquet")

def call_script(path, *extra):
    """Run another python script; raise on failure."""
    print(f"{path} {' '.join(extra)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, str(path), *extra], check=True, env=env)
    print(f"finished {path.name}")

def load_best_params():
    if BEST_PARAM_FILE.exists():
        with BEST_PARAM_FILE.open() as fh:
            return json.load(fh)
    return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-bias", action="store_true", help="skip bias-correction pipeline")
    ap.add_argument("--no-opt",   action="store_true", help="skip optimise_filter.py")
    ap.add_argument("--clean-status", action="store_true", help="delete checkpoint and exit")
    args = ap.parse_args()

    if args.clean_status:
        STATUS_FILE.unlink(missing_ok=True)
        print("checkpoint cleared")
        return

    if not args.skip_bias:
        if not BIAS_SCRIPT.exists():
            sys.exit(f"ERROR: {BIAS_SCRIPT} not found")
        call_script(BIAS_SCRIPT)

    if not args.no_opt:
        if OPT_SCRIPT.exists():
            call_script(OPT_SCRIPT)
        else:
            print("WARNING: optimise_filter.py not found – continuing without optimisation")

    print("building train / test data sets …")
    data_train = fu.pd.concat([fu.load_year("land", y).pipe(fu.apply_bias, land=True)
                               .pipe(fu.add_sa_bias)
                               .append(fu.load_year("ocean", y).pipe(fu.apply_bias, land=False)
                               .pipe(fu.add_sa_bias))
                               for y in TRAIN_YEARS], ignore_index=True)

    land_test  = fu.load_year("land", TEST_YEAR).pipe(fu.apply_bias, land=True)
    ocean_test = fu.load_year("ocean", TEST_YEAR).pipe(fu.apply_bias, land=False)
    data_test  = fu.pd.concat([land_test, ocean_test], ignore_index=True).pipe(fu.add_sa_bias)

    for df in (data_train, data_test):
        df["aod_diff"] = df["aod_total"] - df["aod_total_apriori"]

    hp = dict(
        percentile_lnd           = 70,
        percentile_ocn           = 50,
        abstention_threshold_lnd = 1.30,
        abstention_threshold_ocn = 1.05,
        class_weight             = {
            "tc_lnd": (1.0, 1.0),
            "tc_ocn": (1.0, 1.3),
            "sa_lnd": (1.0, 1.0),
            "sa_ocn": (1.0, 1.0),
        },
        tccon_weighting = True,
        save            = False,
    )
    hp.update(load_best_params())          # overwrite with optimiser results (if any)

    print("training ternary filter")
    data_train, data_test, *_ = fu.build_ternary_filter(
        data_train           = data_train,
        data_test            = data_test,
        feats_small_area_all = (fu.feats_sa_lnd, fu.feats_sa_ocn),
        feats_tccon_all      = (fu.feats_tc_lnd, fu.feats_tc_ocn),
        **hp
    )

    full_record = fu.pd.concat([data_train, data_test], ignore_index=True)
    full_record.to_parquet(OUT_PARQUET, index=False)
    print(f"saved output:{OUT_PARQUET}")

if __name__ == "__main__":
    main()
