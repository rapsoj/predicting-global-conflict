"""
run_label_encoding.py
---------------------
Tests label encoding of country_code and religion columns as additional
features on top of the V2 +Engineered feature set.

Matches Rosa's approach:
  - LabelEncoder fitted on training data, applied to test
  - Unknown test values get -1
  - Missing values get -1

Reports both TRAIN and TEST MAE to check for overfitting.
Evaluates on:
  (a) Top-10 most active regions  -- consistent with existing benchmarks
  (b) All regions (broader)       -- consistent with Rosa's evaluation setup

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
from config import settings

HOLDOUT = 6
TARGETS = settings.targets
SEP = '=' * 72

LABEL_ENCODE_COLS = [
    'country_code',
    'majority_religion',
    'minority1_religion',
    'minority2_religion',
]

RELIGION_NUMERIC_COLS = [
    'majority_pct',
    'minority1_pct',
    'minority2_pct',
    'nonreligpct',
]

MODELS = {
    'LGBM-Tweedie': lambda: lgb.LGBMRegressor(
        objective='tweedie', tweedie_variance_power=1.5,
        n_estimators=200, verbose=-1),
    'RF': lambda: RandomForestRegressor(n_estimators=100, random_state=42),
}


# ── Label encoding (Rosa-style) ───────────────────────────────────────────────

def encode_categoricals(X_train, X_test, cols):
    """
    Fit LabelEncoder on training data, apply to test.
    Missing/unknown values -> -1. Matches Rosa's encode_categoricals().
    """
    X_train = X_train.copy()
    X_test  = X_test.copy()

    for col in cols:
        if col not in X_train.columns:
            continue
        le = LabelEncoder()
        train_vals = X_train[col].fillna('__MISSING__').astype(str)
        test_vals  = X_test[col].fillna('__MISSING__').astype(str)

        le.fit(train_vals)
        known = set(le.classes_)

        X_train[col] = le.transform(train_vals)
        X_test[col]  = test_vals.map(
            lambda x: le.transform([x])[0] if x in known else -1
        )

    return X_train, X_test


def apply_global_label_encoding(df, cols):
    """
    Pre-encode label columns globally (safe for static attributes like
    country_code and religion which do not vary over time within a region).
    Returns df with integer-encoded columns replacing the original strings.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        mask = df[col].notna()
        vals = df[col].fillna('__MISSING__').astype(str)
        df[col] = le.fit_transform(vals)
        df[col] = df[col].astype(int)
    return df


# ── Evaluator with train + test MAE ──────────────────────────────────────────

def evaluate(df, region, target, predictors, model_factory, encode_cols=None):
    """
    Returns {train_mae, test_mae} for one (region, target, feature set).
    If encode_cols is given, fits LabelEncoder on train and applies to test
    for those columns (Rosa-style, no global leakage).
    """
    rdf   = df[df['matched_admin1_id'] == region].sort_values('month_year').copy()
    if len(rdf) < HOLDOUT + 6:
        return None

    avail = [p for p in predictors if p in rdf.columns]
    train = rdf.iloc[:-HOLDOUT]
    test  = rdf.iloc[-HOLDOUT:]

    X_tr = train[avail].fillna(0)
    X_te = test[avail].fillna(0)
    y_tr = train[target].fillna(0)
    y_te = test[target].fillna(0)
    w    = train.get('importance_weight',
                     pd.Series(1.0, index=train.index)).fillna(1)

    # Apply Rosa-style per-split label encoding if requested
    if encode_cols:
        cols_present = [c for c in encode_cols if c in avail]
        if cols_present:
            X_tr, X_te = encode_categoricals(X_tr, X_te, cols_present)

    if y_tr.sum() == 0:
        return None

    m = model_factory()
    try:
        m.fit(X_tr, y_tr, sample_weight=w)
    except (TypeError, Exception):
        try:
            m.fit(X_tr, y_tr)
        except Exception:
            return None

    train_pred = np.maximum(0, m.predict(X_tr))
    test_pred  = np.maximum(0, m.predict(X_te))

    return {
        'train_mae': round(mean_absolute_error(y_tr, train_pred), 2),
        'test_mae':  round(mean_absolute_error(y_te, test_pred),  2),
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


def run_comparison(df, regions, feature_sets, label_encode_cols):
    """
    For each model x feature_set x target x region combination, collect
    train and test MAE.
    """
    results = []
    total   = len(MODELS) * len(feature_sets) * len(TARGETS) * len(regions)
    done    = 0

    for model_name, model_factory in MODELS.items():
        for fs_name, (feats, encode_cols) in feature_sets.items():
            for target in TARGETS:
                for region in regions:
                    r = evaluate(df, region, target, feats, model_factory,
                                 encode_cols=encode_cols)
                    if r:
                        results.append({
                            'model':      model_name,
                            'feature_set': fs_name,
                            'target':     target,
                            'region':     region,
                            **r,
                        })
                    done += 1
                    if done % 50 == 0:
                        print(f'  {done}/{total}...')
    return pd.DataFrame(results)


def print_table(df_res, regions, label):
    print(f'\n{label}  (n={len(regions)} regions)')
    print('-' * 72)

    persist = {t: np.mean([r for r in
                            [persistence_mae(raw_df, reg, t) for reg in regions]
                            if r is not None])
               for t in TARGETS}
    persist['Overall'] = np.mean(list(persist.values()))

    print(f'  {"":30s}  {"Battles":>8}  {"Explosions":>10}  {"VaC":>6}  '
          f'{"Overall":>8}  {"Train":>8}  {"Overfit?":>9}')
    print('  ' + '-' * 85)

    # Persistence row
    print(f'  {"Persistence":30s}  '
          f'{persist["Battles"]:>8.2f}  '
          f'{persist["Explosions/Remote violence"]:>10.2f}  '
          f'{persist["Violence against civilians"]:>6.2f}  '
          f'{persist["Overall"]:>8.2f}  '
          f'{"—":>8}  {"—":>9}')

    for (model_name, fs_name), grp in df_res.groupby(['model', 'feature_set']):
        test_by_t  = grp.groupby('target')['test_mae'].mean()
        train_by_t = grp.groupby('target')['train_mae'].mean()
        test_ov    = grp['test_mae'].mean()
        train_ov   = grp['train_mae'].mean()
        overfit    = 'YES' if (test_ov - train_ov) / (train_ov + 1e-9) > 0.3 else 'no'

        row_label = f'{model_name} / {fs_name}'
        print(f'  {row_label:30s}  '
              f'{test_by_t.get("Battles", float("nan")):>8.2f}  '
              f'{test_by_t.get("Explosions/Remote violence", float("nan")):>10.2f}  '
              f'{test_by_t.get("Violence against civilians", float("nan")):>6.2f}  '
              f'{test_ov:>8.2f}  '
              f'{train_ov:>8.2f}  '
              f'{overfit:>9}')


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    global raw_df

    print('Loading V2 enriched dataset...')
    df   = prepare_enriched_pipeline_v2(clean_data=False)
    fs   = build_feature_sets_v2(df)
    raw_df = df
    BEST = fs['+Engineered']

    # New features to add
    new_label_cols    = [c for c in LABEL_ENCODE_COLS if c in df.columns]
    new_religion_nums = [c for c in RELIGION_NUMERIC_COLS if c in df.columns]
    new_cols          = new_label_cols + new_religion_nums

    BEST_WITH_LABELS = BEST + new_cols

    print(f'\nV2 +Engineered:         {len(BEST)} features')
    print(f'+ Label-encoded cols:   {len(BEST_WITH_LABELS)} features')
    print(f'  New cols added: {new_cols}')

    # Feature sets to compare:
    # (feature_list, encode_cols_for_per_split_encoding)
    feature_sets = {
        'V2 +Eng (baseline)':   (BEST,             None),
        '+ Labels (global enc)': (BEST_WITH_LABELS, None),         # pre-encoded globally
        '+ Labels (split enc)':  (BEST_WITH_LABELS, new_label_cols), # Rosa-style per-split
    }

    top10   = find_top_regions(df, TARGETS, n=10)
    all_reg = [r for r in df['matched_admin1_id'].unique()
               if len(df[df['matched_admin1_id'] == r]) >= HOLDOUT + 6]

    # For all-regions run use a sample of 100 to keep runtime reasonable
    rng       = np.random.default_rng(42)
    sample100 = rng.choice(all_reg, size=min(100, len(all_reg)),
                            replace=False).tolist()

    # ── Pre-encode globally for global-enc variant ────────────────────────────
    print('\nApplying global label encoding...')
    df_global = apply_global_label_encoding(df, new_label_cols)

    # Build a parallel feature_sets dict pointing to df_global where needed
    # We run two separate passes: one on df (for baseline + split-enc),
    # one on df_global (for global-enc variant).

    print(f'\nRunning top-10 evaluation ({len(top10)} regions)...')
    combos_top10 = len(MODELS) * 3 * len(TARGETS) * len(top10)
    print(f'Total evaluations: {combos_top10}')

    # Baseline and split-enc on original df
    fs_on_orig = {
        'V2 +Eng (baseline)':   (BEST,             None),
        '+ Labels (split enc)':  (BEST_WITH_LABELS, new_label_cols),
    }
    res_top10_orig = run_comparison(df, top10, fs_on_orig, new_label_cols)

    # Global-enc on globally encoded df
    fs_on_global = {
        '+ Labels (global enc)': (BEST_WITH_LABELS, None),
    }
    res_top10_global = run_comparison(df_global, top10, fs_on_global, new_label_cols)

    res_top10 = pd.concat([res_top10_orig, res_top10_global], ignore_index=True)

    print(f'\nRunning broad evaluation ({len(sample100)} sampled regions)...')
    res_broad_orig   = run_comparison(df,        sample100, fs_on_orig,   new_label_cols)
    res_broad_global = run_comparison(df_global, sample100, fs_on_global, new_label_cols)
    res_broad = pd.concat([res_broad_orig, res_broad_global], ignore_index=True)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('RESULTS — Label encoding of country_code + religion columns')
    print(SEP)

    print_table(res_top10, top10, 'TOP-10 most active regions')
    print_table(res_broad, sample100, 'BROAD sample (100 regions, includes sporadic + dormant)')

    # ── Delta summary ─────────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('DELTA vs V2 baseline — test MAE improvement (negative = better)')
    print(SEP)

    for eval_label, res in [('Top-10', res_top10), ('Broad-100', res_broad)]:
        print(f'\n  {eval_label}:')
        baseline = res[res['feature_set'] == 'V2 +Eng (baseline)']
        for fs_name in ['+ Labels (global enc)', '+ Labels (split enc)']:
            sub = res[res['feature_set'] == fs_name]
            if sub.empty:
                continue
            for model_name in MODELS:
                base_mae = baseline[baseline['model'] == model_name]['test_mae'].mean()
                new_mae  = sub[sub['model'] == model_name]['test_mae'].mean()
                delta    = new_mae - base_mae
                pct      = delta / base_mae * 100 if base_mae else 0
                flag     = '+' if delta < 0 else '-'
                print(f'    {model_name:15s} / {fs_name:25s}: '
                      f'{base_mae:.2f} -> {new_mae:.2f}  '
                      f'({delta:+.2f}, {pct:+.1f}%) {flag}')

    # Save
    os.makedirs('outputs', exist_ok=True)
    res_top10.to_csv('outputs/label_encoding_top10.csv', index=False)
    res_broad.to_csv('outputs/label_encoding_broad.csv', index=False)
    print('\nSaved: outputs/label_encoding_top10.csv')
    print('Saved: outputs/label_encoding_broad.csv')


if __name__ == '__main__':
    main()
