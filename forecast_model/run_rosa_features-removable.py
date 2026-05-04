"""
run_rosa_features.py
--------------------
Adapts Rosa-Branch features into the giray pipeline and benchmarks them
against the V2 +Engineered baseline.

Four additions from Rosa:
  1. All 8 religion-specific holiday counts
  2. Religion identity (label-encoded) + composition percentages
  3. 10 WBD domain-engineered macro features
  4. 7 holiday x religion interaction features

Loads existing model_data_v2_enriched.csv, computes new features on top,
and runs LGBM-Tweedie + RF comparison with train AND test MAE.

Run from: forecast_model/
"""

import sys, os
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor

from utils.preprocessing_v2 import prepare_enriched_pipeline_v2, build_feature_sets_v2
from utils.evaluators import find_top_regions
from utils.features.WBD_features import add_worldbank_engineered_features, WBD_FEATURE_NAMES
from utils.features.hol_rel import add_holiday_religion_features, HOL_REL_FEATURE_NAMES
from utils.features.improved_features import (
    PREDICTORS_V2, COUNTRY_FEATURES, HOLIDAY_FEATURES_V2,
    HOLIDAY_FEATURES_SPECIFIC, HOLIDAY_INTERACTION_FEATURES,
    RELIGION_FEATURES, WBD_FEATURES,
)
from config import settings

HOLDOUT = 6
TARGETS = settings.targets
SEP     = '=' * 72

MODELS = {
    'LGBM-Tweedie': lambda: lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        n_estimators=200, verbose=-1),
    'RF': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}

LABEL_COLS = ['majority_religion', 'minority1_religion', 'minority2_religion']


# ── Feature preparation ───────────────────────────────────────────────────────

def build_enriched_df(df_base):
    """
    Compute all Rosa features on top of the V2 enriched DataFrame.
    Order: holiday interactions first (need string religion), then label encode.
    """
    df = df_base.copy()

    # 1. WBD engineered features
    print('  Computing WBD engineered features...')
    df = add_worldbank_engineered_features(df)

    # 2. Holiday x religion interactions (needs string religion cols)
    print('  Computing holiday x religion interactions...')
    df = add_holiday_religion_features(df)

    # 3. Label-encode religion identity columns
    print('  Label-encoding religion columns...')
    for col in LABEL_COLS:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        vals = df[col].fillna('__MISSING__').astype(str)
        df[col] = le.fit_transform(vals)

    return df


def build_feature_sets(df, base_feats, risk_cols, engineered_cols):
    """Build cumulative feature set tiers including all Rosa additions."""
    avail = set(df.columns)

    hol_specific  = [c for c in HOLIDAY_FEATURES_SPECIFIC  if c in avail]
    hol_interact  = [c for c in HOLIDAY_INTERACTION_FEATURES if c in avail]
    religion_feats = [c for c in RELIGION_FEATURES          if c in avail]
    wbd_feats      = [c for c in WBD_FEATURES               if c in avail]
    country_feats  = [c for c in COUNTRY_FEATURES           if c in avail]
    eng_feats      = [c for c in engineered_cols            if c in avail]

    return {
        'V2 Baseline':       base_feats,
        '+ Religion':        base_feats + religion_feats,
        '+ WBD':             base_feats + religion_feats + wbd_feats,
        '+ Holidays (all)':  base_feats + religion_feats + wbd_feats + HOLIDAY_FEATURES_V2 + hol_specific,
        '+ Hol Interact':    base_feats + religion_feats + wbd_feats + HOLIDAY_FEATURES_V2 + hol_specific + hol_interact,
        '+ Country':         base_feats + religion_feats + wbd_feats + HOLIDAY_FEATURES_V2 + hol_specific + hol_interact + country_feats,
        '+ Engineered':      base_feats + religion_feats + wbd_feats + HOLIDAY_FEATURES_V2 + hol_specific + hol_interact + country_feats + eng_feats,
    }


# ── Evaluator ─────────────────────────────────────────────────────────────────

def evaluate(df, region, target, predictors, model_factory):
    rdf   = df[df['matched_admin1_id'] == region].sort_values('month_year').copy()
    if len(rdf) < HOLDOUT + 6:
        return None
    avail = [p for p in predictors if p in rdf.columns]
    train = rdf.iloc[:-HOLDOUT]
    test  = rdf.iloc[-HOLDOUT:]
    X_tr  = train[avail].fillna(0)
    X_te  = test[avail].fillna(0)
    y_tr  = train[target].fillna(0)
    y_te  = test[target].fillna(0)
    w     = train.get('importance_weight',
                      pd.Series(1.0, index=train.index)).fillna(1)
    if y_tr.sum() == 0:
        return None
    m = model_factory()
    try:
        m.fit(X_tr, y_tr, sample_weight=w)
    except TypeError:
        try:
            m.fit(X_tr, y_tr)
        except Exception:
            return None
    except Exception:
        return None
    tr_pred = np.maximum(0, m.predict(X_tr))
    te_pred = np.maximum(0, m.predict(X_te))
    return {
        'train_mae': round(mean_absolute_error(y_tr, tr_pred), 2),
        'test_mae':  round(mean_absolute_error(y_te, te_pred),  2),
    }


def persistence_mae(df, region, target):
    rdf = df[df['matched_admin1_id'] == region].sort_values('month_year')
    if len(rdf) < HOLDOUT + 6:
        return None
    test = rdf.iloc[-HOLDOUT:]
    lag  = f'{target} (t-1)'
    if lag not in test.columns:
        return None
    return mean_absolute_error(test[target].fillna(0), test[lag].fillna(0))


def run_eval(df, regions, feature_sets):
    records = []
    total   = len(MODELS) * len(feature_sets) * len(TARGETS) * len(regions)
    done    = 0
    for mname, mfac in MODELS.items():
        for fsname, feats in feature_sets.items():
            for target in TARGETS:
                for region in regions:
                    r = evaluate(df, region, target, feats, mfac)
                    if r:
                        records.append({'model': mname, 'feature_set': fsname,
                                        'target': target, 'region': region, **r})
                    done += 1
                    if done % 100 == 0:
                        print(f'  {done}/{total}...')
    return pd.DataFrame(records)


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_results(res, regions, df, label):
    persist = {t: np.nanmean([persistence_mae(df, r, t) for r in regions])
               for t in TARGETS}
    persist['Overall'] = np.mean(list(persist.values()))

    print(f'\n{label}')
    print('-' * 90)
    header = f'  {"Model / Feature set":38s}  {"Battles":>8}  {"Expl":>8}  {"VaC":>6}  {"Overall":>8}  {"Train":>8}  {"Gap":>6}'
    print(header)
    print('  ' + '-' * 85)
    print(f'  {"Persistence":38s}  '
          f'{persist["Battles"]:>8.2f}  '
          f'{persist["Explosions/Remote violence"]:>8.2f}  '
          f'{persist["Violence against civilians"]:>6.2f}  '
          f'{persist["Overall"]:>8.2f}')

    for (mname, fsname), grp in res.groupby(['model', 'feature_set']):
        tt = grp.groupby('target')['test_mae'].mean()
        tr = grp['train_mae'].mean()
        ov = grp['test_mae'].mean()
        gap = ov - tr
        row = f'{mname} / {fsname}'
        print(f'  {row:38s}  '
              f'{tt.get("Battles", float("nan")):>8.2f}  '
              f'{tt.get("Explosions/Remote violence", float("nan")):>8.2f}  '
              f'{tt.get("Violence against civilians", float("nan")):>6.2f}  '
              f'{ov:>8.2f}  {tr:>8.2f}  {gap:>+6.1f}')


def print_delta(res, label):
    print(f'\nDelta vs V2 Baseline — {label}  (negative = improvement)')
    print('-' * 72)
    for mname in MODELS:
        base = res[(res['model'] == mname) & (res['feature_set'] == 'V2 Baseline')]
        base_mae = base['test_mae'].mean()
        for fsname in res['feature_set'].unique():
            if fsname == 'V2 Baseline':
                continue
            sub = res[(res['model'] == mname) & (res['feature_set'] == fsname)]
            if sub.empty:
                continue
            new_mae = sub['test_mae'].mean()
            d = new_mae - base_mae
            pct = d / base_mae * 100
            flag = '+' if d < 0 else '-'
            print(f'  {mname:15s} / {fsname:25s}: '
                  f'{base_mae:.2f} -> {new_mae:.2f}  ({d:+.2f}, {pct:+.1f}%) {flag}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading V2 enriched dataset...')
    df_base = prepare_enriched_pipeline_v2(clean_data=False)
    fs_v2   = build_feature_sets_v2(df_base)

    print('Adding Rosa features...')
    df = build_enriched_df(df_base)

    # Detect existing risk + engineered column names
    risk_cols = [c for c in df.columns if c.startswith('risk_') and c.endswith('(t-1)')]
    eng_cols  = [f for f in [
        'Battles (t-2)', 'Explosions/Remote violence (t-2)',
        'Violence against civilians (t-2)', 'organized_violence (t-1)',
        'is_active (t-1)', 'battles_x_remote (t-1)',
        'Battles_3mo_avg (t-1)', 'Remote_3mo_avg (t-1)', 'VaC_3mo_avg (t-1)',
    ] if f in df.columns]

    base_feats  = PREDICTORS_V2 + risk_cols + settings.macro_features
    feature_sets = build_feature_sets(df, base_feats, risk_cols, eng_cols)

    print('\nFeature set sizes:')
    for name, feats in feature_sets.items():
        avail = [f for f in feats if f in df.columns]
        print(f'  {name:30s}: {len(avail)} features')

    top10 = find_top_regions(df_base, TARGETS, n=10)

    # Broad sample: 80 regions across activity tiers
    all_reg = [r for r in df['matched_admin1_id'].unique()
               if len(df[df['matched_admin1_id'] == r]) >= HOLDOUT + 6]
    rng     = np.random.default_rng(42)
    broad80 = rng.choice(all_reg, size=min(80, len(all_reg)), replace=False).tolist()

    print(f'\nRunning top-10 evaluation ({len(top10)} regions, '
          f'{len(MODELS) * len(feature_sets) * len(TARGETS) * len(top10)} evals)...')
    res_top10 = run_eval(df, top10, feature_sets)

    print(f'\nRunning broad evaluation ({len(broad80)} regions)...')
    res_broad = run_eval(df, broad80, feature_sets)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('RESULTS — Rosa features integrated into giray pipeline')
    print(SEP)

    print_results(res_top10, top10,   df, 'TOP-10 most active regions')
    print_results(res_broad, broad80, df, 'BROAD sample (80 regions)')

    print(f'\n{SEP}')
    print('DELTA vs V2 Baseline')
    print(SEP)
    print_delta(res_top10, 'Top-10')
    print_delta(res_broad, 'Broad-80')

    # Save
    os.makedirs('outputs', exist_ok=True)
    res_top10.to_csv('outputs/rosa_features_top10.csv',  index=False)
    res_broad.to_csv('outputs/rosa_features_broad80.csv', index=False)
    print('\nSaved: outputs/rosa_features_top10.csv')
    print('Saved: outputs/rosa_features_broad80.csv')


if __name__ == '__main__':
    main()
