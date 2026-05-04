"""
Run LGBM-Tweedie +Country evaluation on ALL valid regions.
Produces outputs/full_region_eval.csv and prints summary.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import lightgbm as lgb
from itertools import product as iproduct

from config import settings
from utils.evaluators import evaluate_model
from utils.ablation import build_feature_sets

HOLDOUT = 6
TARGETS = settings.targets
MODEL_FACTORY = lambda: lgb.LGBMRegressor(
    objective="tweedie", tweedie_variance_power=1.5,
    n_estimators=200, learning_rate=0.05,
    num_leaves=31, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=1,
)

print("Loading v2 enriched data...")
df = pd.read_csv("data/processed/model_data_v2_enriched.csv")
print(f"Data: {df.shape[0]:,} rows, {df['matched_admin1_id'].nunique()} regions")

# Label-encode string religion columns (LightGBM requires numeric dtypes)
from sklearn.preprocessing import LabelEncoder
for col in ["majority_religion", "minority1_religion", "minority2_religion"]:
    if col in df.columns and df[col].dtype == object:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].fillna("unknown").astype(str))

feature_sets = build_feature_sets(df)
BEST_FS_NAME = "+Country"
BEST_FS = feature_sets[BEST_FS_NAME]
print(f"Feature set '{BEST_FS_NAME}': {len(BEST_FS)} features")

# --- Find all valid regions (same criteria as Rosa's notebook) ---
all_regions = df["matched_admin1_id"].unique()
valid_regions = []
for region in all_regions:
    rdf = df[df["matched_admin1_id"] == region].sort_values("month_year")
    if len(rdf) < HOLDOUT + 6:
        continue
    train = rdf.iloc[:-HOLDOUT]
    if train[TARGETS].nunique().max() <= 1:
        continue
    valid_regions.append(region)

print(f"Valid regions: {len(valid_regions)}")

# --- Persistence baseline ---
print("Computing persistence baseline...")
persist_records = []
for region, target in iproduct(valid_regions, TARGETS):
    rdf = df[df["matched_admin1_id"] == region].sort_values("month_year")
    test = rdf.iloc[-HOLDOUT:]
    lag_col = f"{target} (t-1)"
    if lag_col not in test.columns:
        continue
    y_test = test[target].fillna(0).values
    y_pred = test[lag_col].fillna(0).values
    mae = float(np.mean(np.abs(y_test - y_pred)))
    y_prev = test[lag_col].fillna(0).values
    true_dir = np.sign(y_test - y_prev)
    pred_dir = np.sign(y_pred - y_prev)
    da = float(np.mean(true_dir == pred_dir))
    persist_records.append({"model": "Persistence", "feature_set": "—",
                             "target": target, "region": region,
                             "mae": round(mae, 4), "directional_accuracy": round(da, 4)})

persist_df = pd.DataFrame(persist_records)
print("Persistence done.")

# --- LGBM-Tweedie evaluation ---
combos = list(iproduct(valid_regions, TARGETS))
print(f"Running LGBM-Tweedie on {len(combos)} (region, target) pairs...")
t0 = time.time()
model_records = []
for i, (region, target) in enumerate(combos):
    r = evaluate_model(df, region, target, BEST_FS, MODEL_FACTORY, holdout=HOLDOUT)
    if r:
        model_records.append({"model": "LGBM-Tweedie", "feature_set": BEST_FS_NAME,
                               "target": target, "region": region, **r})
    if (i + 1) % 500 == 0:
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        eta = (len(combos) - i - 1) / rate
        print(f"  {i+1}/{len(combos)}  ({rate:.0f}/s, ETA {eta:.0f}s)")

model_df = pd.DataFrame(model_records)
print(f"Done in {time.time()-t0:.0f}s. {len(model_records)} results.")

# --- Combine and save ---
all_df = pd.concat([persist_df, model_df], ignore_index=True)
os.makedirs("outputs", exist_ok=True)
all_df.to_csv("outputs/full_region_eval.csv", index=False)
print("Saved: outputs/full_region_eval.csv")

# --- Summary ---
print("\n" + "="*70)
print("FULL-REGION EVALUATION SUMMARY")
print("Metric: mean over all valid regions, 6-month holdout")
print("="*70)

for target in TARGETS:
    p = persist_df[persist_df["target"] == target]
    m = model_df[model_df["target"] == target]
    p_mae = p["mae"].mean()
    m_mae = m["mae"].mean()
    p_da  = p["directional_accuracy"].mean()
    m_da  = m["directional_accuracy"].mean()
    pct   = (m_mae - p_mae) / p_mae * 100
    print(f"\n{target}:")
    print(f"  Persistence   MAE={p_mae:.4f}  DA={p_da:.4f}")
    print(f"  LGBM-Tweedie  MAE={m_mae:.4f}  DA={m_da:.4f}  ({pct:+.1f}% vs persist)")

p_all = persist_df["mae"].mean()
m_all = model_df["mae"].mean()
p_da_all = persist_df["directional_accuracy"].mean()
m_da_all = model_df["directional_accuracy"].mean()
print(f"\nOVERALL:")
print(f"  Persistence   MAE={p_all:.4f}  DA={p_da_all:.4f}")
print(f"  LGBM-Tweedie  MAE={m_all:.4f}  DA={m_da_all:.4f}  ({(m_all-p_all)/p_all*100:+.1f}% vs persist)")
