"""
improvements_7.py
-----------------
Six targeted improvements over V2 best (LGBM-Tweedie +Engineered, MAE=53.05).

  1. Two-stage model   ~ binary onset classifier + Tweedie intensity regressor
  2. Stratified eval   ~ always-active / sporadic / dormant region tiers
  3. Per-horizon MAE   ~ accuracy at each of the 6 holdout months
  4. Trajectory feats  ~ slope and acceleration of target time series (t-1)
  7. Log-target        ~ predict log1p(Y), evaluate in original scale
  9. Neighbour traj    ~ slope of neighbour event counts

Run from: forecast_model/
"""

import sys, os
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

from utils.preprocessing_v2 import prepare_enriched_pipeline_v2, build_feature_sets_v2
from utils.evaluators import find_top_regions
from config import settings

HOLDOUT  = 6
TARGETS  = settings.targets
TWEEDIE  = lambda: lgb.LGBMRegressor(
    objective='tweedie', tweedie_variance_power=1.5, n_estimators=200, verbose=-1)
MSE_LGB  = lambda: lgb.LGBMRegressor(
    objective='regression', n_estimators=200, verbose=-1)

SEP = '=' * 72


# ── helpers ───────────────────────────────────────────────────────────────────

def _split(df, region):
    rdf = df[df['matched_admin1_id'] == region].sort_values('month_year').copy()
    if len(rdf) < HOLDOUT + 6:
        return None, None
    return rdf.iloc[:-HOLDOUT], rdf.iloc[-HOLDOUT:]


def _fit_predict(train, test, avail, target, factory, log_y=False):
    X_tr = train[avail].fillna(0)
    X_te = test[avail].fillna(0)
    y_tr = train[target].fillna(0)
    w    = train.get('importance_weight',
                     pd.Series(1.0, index=train.index)).fillna(1)
    if log_y:
        y_tr = np.log1p(y_tr)
    # Tweedie requires positive sum of labels; fall back to zeros if all-zero target
    if y_tr.sum() == 0:
        return np.zeros(len(X_te))
    m = factory()
    try:
        m.fit(X_tr, y_tr, sample_weight=w)
    except (TypeError, Exception):
        try:
            m.fit(X_tr, y_tr)
        except Exception:
            return np.zeros(len(X_te))
    raw = m.predict(X_te)
    return np.maximum(0, np.expm1(raw) if log_y else raw)


def avail_feats(df, predictors):
    return [p for p in predictors if p in df.columns]


def mean_mae(records):
    return round(np.mean([r for r in records if r is not None]), 2)


def fmt(val, ref, label=''):
    delta = val - ref
    pct   = delta / ref * 100 if ref else 0
    flag  = '+' if delta < 0 else '-'
    return f'{val:.2f}  ({delta:+.2f}, {pct:+.1f}% vs {label}) {flag}'


# ── FEATURE ENGINEERING ~ items 4 & 9 ────────────────────────────────────────

def add_trajectory_features(df):
    df = df.copy().sort_values(['matched_admin1_id', 'month_year'])
    grp = df.groupby('matched_admin1_id')
    new_cols = []

    # Item 4: slope and acceleration for each target
    for t in TARGETS:
        t1, t2 = f'{t} (t-1)', f'{t} (t-2)'
        if t1 not in df.columns or t2 not in df.columns:
            continue
        sc = f'{t}_slope(t-1)'
        df[sc] = df[t1] - df[t2]
        new_cols.append(sc)

        t3    = grp[t2].shift(1)                        # computed t-3 per region
        ac    = f'{t}_accel(t-1)'
        df[ac] = (df[t1] - df[t2]) - (df[t2] - t3)
        new_cols.append(ac)

    # Item 9: slope of neighbour counts
    nbr_map = {
        'Battles':                    'Battles_neighbours (t-1)',
        'Explosions/Remote violence': 'Explosions/Remote violence_neighbours (t-1)',
        'Violence against civilians': 'Violence against civilians_neighbours (t-1)',
    }
    for t, nc in nbr_map.items():
        if nc not in df.columns:
            continue
        sc    = f'{t}_nbr_slope(t-1)'
        df[sc] = df[nc] - grp[nc].shift(1)
        new_cols.append(sc)

    return df.fillna(0), new_cols


# ── STRATIFICATION ~ item 2 ───────────────────────────────────────────────────

def stratify_regions(df, n=8):
    """
    Classify all regions by training-period activity and return top-n per tier.
    Tiers: always-active (frac>0.7, mean>5) / dormant (frac<0.15) / sporadic.
    """
    rows = []
    for region in df['matched_admin1_id'].unique():
        rdf = df[df['matched_admin1_id'] == region]
        if len(rdf) < HOLDOUT + 6:
            continue
        train = rdf.iloc[:-HOLDOUT]
        vals  = pd.concat([train[t].fillna(0) for t in TARGETS if t in train])
        rows.append({
            'region':       region,
            'frac_nonzero': (vals > 0).mean(),
            'mean_y':       vals.mean(),
        })
    st = pd.DataFrame(rows)

    always  = st[(st['frac_nonzero'] > 0.70) & (st['mean_y'] > 5)]
    dormant = st[st['frac_nonzero'] < 0.15]
    sporadic = st[~st.index.isin(always.index) & ~st.index.isin(dormant.index)]

    pick = lambda sub, col, asc: (sub.sort_values(col, ascending=asc)
                                     .head(n)['region'].tolist())
    return {
        'always_active': pick(always,  'mean_y',       False),
        'sporadic':      pick(sporadic, 'frac_nonzero', False),
        'dormant':       pick(dormant,  'frac_nonzero', False),
    }


# ── EVALUATORS ────────────────────────────────────────────────────────────────

def eval_persistence(df, region, target):
    _, test = _split(df, region)
    if test is None:
        return None
    lag = f'{target} (t-1)'
    if lag not in test.columns:
        return None
    return mean_absolute_error(test[target].fillna(0), test[lag].fillna(0))


def eval_standard(df, region, target, predictors, factory=TWEEDIE):
    train, test = _split(df, region)
    if train is None:
        return None
    av   = avail_feats(df, predictors)
    pred = _fit_predict(train, test, av, target, factory)
    return mean_absolute_error(test[target].fillna(0), pred)


def eval_log_target(df, region, target, predictors):
    """Item 7: train on log1p(Y), back-transform predictions."""
    train, test = _split(df, region)
    if train is None:
        return None
    av   = avail_feats(df, predictors)
    pred = _fit_predict(train, test, av, target, MSE_LGB, log_y=True)
    return mean_absolute_error(test[target].fillna(0), pred)


class TwoStageModel:
    """
    Item 1: binary onset classifier + Tweedie intensity regressor.

    Edge cases handled:
      - Always-active region (>95% non-zero): skip classifier, use regressor directly.
      - Always-dormant region (<5% non-zero): predict zeros.
      - Mixed: soft combination P(active) x E[Y|active].
    """

    def fit(self, X, y, w=None):
        y_bin = (y > 0).astype(int)
        active_frac = y_bin.mean()
        self._always_active  = active_frac > 0.95
        self._always_dormant = active_frac < 0.05

        # Intensity regressor (trained on active months only)
        mask = y > 0
        self._has_reg = mask.sum() >= 10 and y[mask].sum() > 0
        if self._has_reg:
            self.reg_ = lgb.LGBMRegressor(
                objective='tweedie', tweedie_variance_power=1.5,
                n_estimators=200, verbose=-1)
            try:
                self.reg_.fit(X[mask], y[mask],
                              sample_weight=w[mask] if w is not None else None)
            except Exception:
                self._has_reg = False

        # Classifier (only needed for mixed regions)
        if not self._always_active and not self._always_dormant:
            self.clf_ = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
            try:
                self.clf_.fit(X, y_bin, sample_weight=w)
                self._has_clf = True
            except Exception:
                self._has_clf = False
        else:
            self._has_clf = False
        return self

    def predict(self, X):
        if self._always_dormant or not self._has_reg:
            return np.zeros(len(X))
        intens = np.maximum(0, self.reg_.predict(X))
        if self._always_active or not self._has_clf:
            return intens
        prob = self.clf_.predict_proba(X)[:, 1]
        return prob * intens


def eval_two_stage(df, region, target, predictors):
    """Item 1."""
    train, test = _split(df, region)
    if train is None:
        return None
    av  = avail_feats(df, predictors)
    Xtr = train[av].fillna(0).values
    Xte = test[av].fillna(0).values
    ytr = train[target].fillna(0).values
    yte = test[target].fillna(0).values
    w   = train.get('importance_weight',
                    pd.Series(1.0, index=train.index)).fillna(1).values
    m   = TwoStageModel().fit(Xtr, ytr, w)
    return mean_absolute_error(yte, np.maximum(0, m.predict(Xte)))


def eval_per_horizon(df, region, target, predictors, factory=TWEEDIE):
    """Item 3: return per-step absolute errors (shape: holdout,)."""
    train, test = _split(df, region)
    if train is None:
        return None
    av   = avail_feats(df, predictors)
    pred = _fit_predict(train, test, av, target, factory)
    return np.abs(test[target].fillna(0).values - pred)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print('Loading V2 enriched dataset...')
    df = prepare_enriched_pipeline_v2(clean_data=False)
    fs = build_feature_sets_v2(df)
    BEST = fs['+Engineered']

    print('Adding trajectory features (items 4 & 9)...')
    df_traj, traj_cols = add_trajectory_features(df)
    BEST_TRAJ = BEST + traj_cols

    top10 = find_top_regions(df, TARGETS, n=10)
    print(f'Top-10 regions: {top10}\n')

    # ── PERSISTENCE BASELINE ──────────────────────────────────────────────────
    print(SEP)
    print('PERSISTENCE BASELINE  (reference for all comparisons)')
    print(SEP)
    persist = {t: mean_mae([eval_persistence(df, r, t) for r in top10])
               for t in TARGETS}
    persist['Overall'] = round(np.mean(list(persist.values())), 2)
    for k, v in persist.items():
        print(f'  {k:35s}: {v:.2f}')

    # V2 Tweedie reference (re-run to have same region set)
    print('\nComputing V2 LGBM-Tweedie +Engineered reference...')
    v2ref = {t: mean_mae([eval_standard(df, r, t, BEST) for r in top10])
             for t in TARGETS}
    v2ref['Overall'] = round(np.mean(list(v2ref.values())), 2)

    # ── ITEM 4+9: TRAJECTORY FEATURES ────────────────────────────────────────
    print(f'\n{SEP}')
    print('ITEMS 4 & 9 ~ Conflict trajectory + neighbour trajectory features')
    print(f'New features added: {traj_cols}')
    print(SEP)
    traj = {t: mean_mae([eval_standard(df_traj, r, t, BEST_TRAJ) for r in top10])
            for t in TARGETS}
    traj['Overall'] = round(np.mean(list(traj.values())), 2)

    print(f'\n{"Target":35s}  {"V2 Tweedie":>10}  {"+ Trajectory":>12}  {"d":>8}')
    print('-' * 72)
    for t in TARGETS + ['Overall']:
        v = v2ref[t]; n = traj[t]; d = n - v
        flag = '+' if d < 0 else '-'
        print(f'  {t:33s}  {v:>10.2f}  {n:>12.2f}  {d:>+8.2f} {flag}')

    # ── ITEM 7: LOG-TARGET ────────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('ITEM 7 ~ Log-transform target  (train on log1p Y, evaluate in raw scale)')
    print(SEP)
    log_r = {t: mean_mae([eval_log_target(df, r, t, BEST) for r in top10])
             for t in TARGETS}
    log_r['Overall'] = round(np.mean(list(log_r.values())), 2)

    print(f'\n{"Target":35s}  {"V2 Tweedie":>10}  {"Log-target":>10}  {"d":>8}')
    print('-' * 72)
    for t in TARGETS + ['Overall']:
        v = v2ref[t]; n = log_r[t]; d = n - v
        flag = '+' if d < 0 else '-'
        print(f'  {t:33s}  {v:>10.2f}  {n:>10.2f}  {d:>+8.2f} {flag}')

    # ── ITEM 1: TWO-STAGE MODEL ───────────────────────────────────────────────
    print(f'\n{SEP}')
    print('ITEM 1 ~ Two-stage model  (classifier × intensity regressor)')
    print(SEP)
    ts = {t: mean_mae([eval_two_stage(df, r, t, BEST) for r in top10])
          for t in TARGETS}
    ts['Overall'] = round(np.mean(list(ts.values())), 2)

    print(f'\n{"Target":35s}  {"V2 Tweedie":>10}  {"Two-stage":>10}  {"d":>8}')
    print('-' * 72)
    for t in TARGETS + ['Overall']:
        v = v2ref[t]; n = ts[t]; d = n - v
        flag = '+' if d < 0 else '-'
        print(f'  {t:33s}  {v:>10.2f}  {n:>10.2f}  {d:>+8.2f} {flag}')

    # ── COMBINED: TRAJ + LOG + TWO-STAGE ─────────────────────────────────────
    print(f'\n{SEP}')
    print('COMBINED ~ Two-stage + trajectory features + log-target for Stage 2')
    print(SEP)

    def eval_combined(df_t, region, target, predictors):
        """Two-stage where Stage 2 trains on log1p(Y) and uses trajectory feats."""
        train, test = _split(df_t, region)
        if train is None:
            return None
        av  = avail_feats(df_t, predictors)
        Xtr = train[av].fillna(0).values
        Xte = test[av].fillna(0).values
        ytr = train[target].fillna(0).values
        yte = test[target].fillna(0).values
        w   = train.get('importance_weight',
                        pd.Series(1.0, index=train.index)).fillna(1).values

        # Stage 1: classifier
        clf = lgb.LGBMClassifier(n_estimators=200, verbose=-1)
        clf.fit(Xtr, (ytr > 0).astype(int), sample_weight=w)
        prob = clf.predict_proba(Xte)[:, 1]

        # Stage 2: log-target regressor on active months
        mask = ytr > 0
        if mask.sum() >= 10:
            reg = lgb.LGBMRegressor(objective='regression', n_estimators=200, verbose=-1)
            reg.fit(Xtr[mask], np.log1p(ytr[mask]),
                    sample_weight=w[mask])
            intens = np.maximum(0, np.expm1(reg.predict(Xte)))
        else:
            intens = np.zeros(len(Xte))

        pred = np.maximum(0, prob * intens)
        return mean_absolute_error(yte, pred)

    comb = {t: mean_mae([eval_combined(df_traj, r, t, BEST_TRAJ) for r in top10])
            for t in TARGETS}
    comb['Overall'] = round(np.mean(list(comb.values())), 2)

    print(f'\n{"Target":35s}  {"V2 Tweedie":>10}  {"Combined":>10}  {"d":>8}')
    print('-' * 72)
    for t in TARGETS + ['Overall']:
        v = v2ref[t]; n = comb[t]; d = n - v
        flag = '+' if d < 0 else '-'
        print(f'  {t:33s}  {v:>10.2f}  {n:>10.2f}  {d:>+8.2f} {flag}')

    # ── ITEM 2: STRATIFIED EVALUATION ────────────────────────────────────────
    print(f'\n{SEP}')
    print('ITEM 2 ~ Stratified evaluation by region activity level')
    print(SEP)
    tiers = stratify_regions(df, n=8)
    for tier_name, regions in tiers.items():
        if not regions:
            continue
        print(f'\n  Tier: {tier_name.upper()}  ({len(regions)} regions)')
        print(f'  Regions: {regions[:4]}{"..." if len(regions)>4 else ""}')
        for label, fn in [
            ('Persistence',  lambda r, t: eval_persistence(df, r, t)),
            ('V2 Tweedie',   lambda r, t: eval_standard(df, r, t, BEST)),
            ('Two-stage',    lambda r, t: eval_two_stage(df, r, t, BEST)),
            ('Log-target',   lambda r, t: eval_log_target(df, r, t, BEST)),
            ('+ Trajectory', lambda r, t: eval_standard(df_traj, r, t, BEST_TRAJ)),
        ]:
            maes = [fn(r, t) for r in regions for t in TARGETS]
            maes = [x for x in maes if x is not None]
            if maes:
                print(f'    {label:15s}: {np.mean(maes):.2f}')

    # ── ITEM 3: PER-HORIZON MAE ───────────────────────────────────────────────
    print(f'\n{SEP}')
    print('ITEM 3 ~ Per-horizon MAE  (how accuracy degrades over 6-month holdout)')
    print(SEP)

    horizon_persist = np.zeros(HOLDOUT)
    horizon_v2      = np.zeros(HOLDOUT)
    horizon_comb    = np.zeros(HOLDOUT)
    counts          = 0

    for region in top10:
        for target in TARGETS:
            ep = eval_per_horizon(df, region, target, BEST, TWEEDIE)
            hp = _split(df, region)
            if ep is None or hp[1] is None:
                continue
            _, test = hp
            lag = f'{target} (t-1)'
            if lag not in test.columns:
                continue
            persist_err = np.abs(test[target].fillna(0).values -
                                 test[lag].fillna(0).values)
            ec = eval_per_horizon(df_traj, region, target, BEST_TRAJ, TWEEDIE)
            if ec is None:
                continue
            horizon_persist += persist_err
            horizon_v2      += ep
            horizon_comb    += ec
            counts          += 1

    if counts:
        horizon_persist /= counts
        horizon_v2      /= counts
        horizon_comb    /= counts
        print(f'\n  {"Horizon":>8}  {"Persistence":>12}  {"V2 Tweedie":>12}  {"+ Trajectory":>14}')
        print('  ' + '-' * 52)
        for h in range(HOLDOUT):
            print(f'  {h+1:>8}  {horizon_persist[h]:>12.2f}  '
                  f'{horizon_v2[h]:>12.2f}  {horizon_comb[h]:>14.2f}')

    # ── FINAL SUMMARY TABLE ───────────────────────────────────────────────────
    print(f'\n{SEP}')
    print('FINAL SUMMARY ~ top-10 most-active regions, mean MAE')
    print(SEP)
    p = persist['Overall']
    rows = [
        ('Persistence (floor)',        persist),
        ('V2 LGBM-Tweedie (baseline)', v2ref),
        ('+ Trajectory feats (4+9)',   traj),
        ('Log-target (7)',             log_r),
        ('Two-stage (1)',              ts),
        ('Combined (1+4+7+9)',         comb),
    ]
    print(f'\n  {"Method":30s}  {"Battles":>8}  {"Explosions":>10}  {"VaC":>6}  {"Overall":>8}  {"vs V2":>8}')
    print('  ' + '-' * 78)
    for name, d in rows:
        ov = d['Overall']
        dv = ov - v2ref['Overall']
        flag = '+' if dv < 0 else ('~' if name.startswith('V2') else '-')
        print(f'  {name:30s}  {d["Battles"]:>8.2f}  '
              f'{d["Explosions/Remote violence"]:>10.2f}  '
              f'{d["Violence against civilians"]:>6.2f}  '
              f'{ov:>8.2f}  {dv:>+7.2f} {flag}')

    # Save results
    result_rows = []
    for name, d in rows:
        result_rows.append({'method': name, **d})
    pd.DataFrame(result_rows).to_csv('outputs/improvements_7_results.csv', index=False)
    print('\nSaved: outputs/improvements_7_results.csv')


if __name__ == '__main__':
    main()
