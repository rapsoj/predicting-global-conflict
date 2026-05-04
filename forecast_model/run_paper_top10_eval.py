"""
Re-run LGBM-Tweedie V2 ablation + RF Baseline on the joint paper's top-10 regions.

The paper's top-10 are defined by total event count during the holdout period
(last 6 months), as used in pre_report.pdf §6.3 and §6.4. This produces numbers
directly comparable to CAST+ (MAE 52.290) and the CAST RF Baseline (MAE 67.968).

Output: outputs/ablation_paper_top10.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from config import settings
from utils.evaluators import evaluate_model
from utils.ablation import build_feature_sets

HOLDOUT = 6
TARGETS = settings.targets

# Paper's top-10 by holdout-period activity (pre_report.pdf §5.2 / §6.3)
PAPER_TOP10 = [
    "UKR - Donetsk",
    "UKR - Sumy",
    "RUS - Belgorod",
    "UKR - Kherson",
    "PSX - Gaza",
    "UKR - Kharkiv",
    "UKR - Zaporizhzhia",
    "RUS - Kursk",
    "UKR - Chernihiv",
    "PSX - West Bank",
]

LGBM_FACTORY = lambda: lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.5,
    n_estimators=200, learning_rate=0.05, num_leaves=31,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=1,
)
RF_FACTORY = lambda: RandomForestRegressor(n_estimators=100, random_state=42)

print("Loading v2 enriched data...")
df = pd.read_csv("data/processed/model_data_v2_enriched.csv")
print(f"Data: {df.shape[0]:,} rows, {df['matched_admin1_id'].nunique()} regions")

for col in ["majority_religion", "minority1_religion", "minority2_religion"]:
    if col in df.columns and df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))

feature_sets = build_feature_sets(df)
print(f"Feature sets: {list(feature_sets.keys())}")

# Verify all paper regions exist
missing = [r for r in PAPER_TOP10 if r not in df["matched_admin1_id"].values]
if missing:
    print(f"WARNING: missing regions in CSV: {missing}")
else:
    print("All 10 paper regions confirmed in CSV.")

records = []
t0 = time.time()

for region in PAPER_TOP10:
    for target in TARGETS:
        for fs_name, fs_preds in feature_sets.items():
            r = evaluate_model(df, region, target, fs_preds, LGBM_FACTORY, holdout=HOLDOUT)
            if r:
                records.append({
                    "model": "LGBM-Tweedie", "feature_set": fs_name,
                    "region": region, "target": target, **r,
                })
        # RF Baseline (ACLED-only, equivalent to CAST RF baseline)
        r_rf = evaluate_model(df, region, target, feature_sets["Baseline"], RF_FACTORY, holdout=HOLDOUT)
        if r_rf:
            records.append({
                "model": "RF-Baseline", "feature_set": "Baseline",
                "region": region, "target": target, **r_rf,
            })

print(f"Done in {time.time()-t0:.0f}s. {len(records)} results.")

results_df = pd.DataFrame(records)
os.makedirs("outputs", exist_ok=True)
results_df.to_csv("outputs/ablation_paper_top10.csv", index=False)
print("Saved: outputs/ablation_paper_top10.csv")

# --- Summary ---
print("\n" + "=" * 70)
print("PAPER TOP-10 EVALUATION SUMMARY")
print("Compare LGBM-Tweedie best vs RF-Baseline vs paper's CAST+ (52.290)")
print("=" * 70)

lgbm_df = results_df[results_df["model"] == "LGBM-Tweedie"]
rf_df   = results_df[results_df["model"] == "RF-Baseline"]

print(f"\nRF Baseline (ACLED-only) — analogous to CAST RF baseline (paper: 67.968):")
print(f"  Overall MAE: {rf_df['mae'].mean():.3f}")
for t in TARGETS:
    print(f"  {t}: MAE={rf_df[rf_df['target']==t]['mae'].mean():.3f}")

print(f"\nLGBM-Tweedie by feature set (overall MAE across all targets + regions):")
pivot = lgbm_df.groupby("feature_set")["mae"].mean().reindex(list(feature_sets.keys()))
for fs, mae in pivot.items():
    print(f"  {fs:15s}: {mae:.3f}")

best_fs = pivot.idxmin()
best_mae = pivot.min()
rf_mae = rf_df["mae"].mean()
pct = (best_mae - rf_mae) / rf_mae * 100
print(f"\nBest LGBM-Tweedie: {best_fs} = MAE {best_mae:.3f}  ({pct:+.1f}% vs RF Baseline)")
print(f"Paper CAST+ on same regions: 52.290 (from pre_report.pdf Table 4)")

print(f"\nPer-target LGBM-Tweedie best:")
for t in TARGETS:
    t_df = lgbm_df[lgbm_df["target"] == t]
    best_t = t_df.groupby("feature_set")["mae"].mean().min()
    best_t_fs = t_df.groupby("feature_set")["mae"].mean().idxmin()
    print(f"  {t}: MAE={best_t:.3f} ({best_t_fs})")

print(f"\nDA (all months) — LGBM-Tweedie best feature set ({best_fs}):")
best_lgbm = lgbm_df[lgbm_df["feature_set"] == best_fs]
print(f"  Overall DA:         {best_lgbm['directional_accuracy'].mean():.4f}")
print(f"  Overall DA (nonzero changing months only): {best_lgbm['directional_accuracy_nonzero'].mean():.4f}")
