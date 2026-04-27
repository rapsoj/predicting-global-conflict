"""
utils/features/improved_features.py
-------------------------------------
Three feature engineering improvements over the baseline pipeline:

  1. Cyclic time encoding    — sin/cos for month and quarter instead of 11+3 binary dummies.
                               Correctly places December adjacent to January on the seasonal
                               cycle; reduces dimensionality from 14 columns to 4.

  2. Current-month holidays  — holiday_count_month at time t (no t-1 lag).
                               Holidays are deterministic calendar facts known in advance,
                               so using current-month values introduces zero leakage while
                               giving the model more precise information.

  3. Country-level hierarchy — leave-one-out country aggregate (t-1) as a national context
                               signal. Approximates a hierarchical model's country-level prior
                               in a form that tree-based models can exploit directly.
"""

import numpy as np
import pandas as pd


# ── Predictor lists for the v2 pipeline ──────────────────────────────────────

# Replaces 11 month dummies (month_2..month_12) + 3 quarter dummies (quarter_2..quarter_4)
# with 4 cyclic features. All other baseline predictors are unchanged.
PREDICTORS_V2 = [
    'Battles (t-1)',
    'Explosions/Remote violence (t-1)',
    'Protests (t-1)',
    'Riots (t-1)',
    'Strategic developments (t-1)',
    'Violence against civilians (t-1)',
    'Excessive force against protesters (t-1)',
    'Agreement (t-1)',
    'Explosions/Remote violence_neighbours (t-1)',
    'Strategic developments_neighbours (t-1)',
    'Protests_neighbours (t-1)',
    'Violence against civilians_neighbours (t-1)',
    'Battles_neighbours (t-1)',
    'Riots_neighbours (t-1)',
    'linear_month_trend',
    'year',
    'month_sin',
    'month_cos',
    'quarter_sin',
    'quarter_cos',
]

# Current-month holiday features — no t-1 lag (holidays are known in advance)
HOLIDAY_FEATURES_V2 = ['holiday_count_month', 'is_holiday_month']

# Country-level leave-one-out aggregate features (all lagged t-1)
COUNTRY_FEATURES = [
    'country_battles_excl (t-1)',
    'country_remote_excl (t-1)',
    'country_vac_excl (t-1)',
    'country_total_excl (t-1)',
]


# ── 1. Cyclic time encoding ───────────────────────────────────────────────────

def add_cyclic_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop-in replacement for data_cleaning.add_time_trend_features().

    Replaces one-hot month/quarter dummies with sin/cos projections onto
    the unit circle. Key properties:
      - December (12) and January (1) are adjacent, not maximally distant
      - Reduces from 14 binary columns to 4 continuous features
      - Gradient-based models benefit more than trees, but trees still gain
        from fewer splits needed to represent seasonal patterns

    Retained unchanged: linear_month_trend, year.
    """
    df = df.reset_index()
    df['month_year'] = pd.to_datetime(df['month_year'])

    start_date = pd.Timestamp('2018-01-01')
    df['linear_month_trend'] = (
        (df['month_year'].dt.year - start_date.year) * 12
        + (df['month_year'].dt.month - start_date.month)
    )
    df['year'] = df['month_year'].dt.year

    month   = df['month_year'].dt.month
    quarter = df['month_year'].dt.quarter

    df['month_sin']   = np.sin(2 * np.pi * month   / 12)
    df['month_cos']   = np.cos(2 * np.pi * month   / 12)
    df['quarter_sin'] = np.sin(2 * np.pi * quarter / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * quarter / 4)

    df['month_year'] = df['month_year'].dt.strftime("%Y-%m")
    df.set_index(['matched_admin1_id', 'month_year'], inplace=True)
    return df


# ── 3. Country-level hierarchy ────────────────────────────────────────────────

def add_country_level_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 4 country-level leave-one-out aggregate features, each lagged t-1.

    For each (admin-1, month_year), sums the three target event types across
    all admin-1 regions in the same country EXCEPT the target region itself.
    This gives the model a signal for national-level conflict dynamics while
    avoiding direct leakage of the target region's own counts.

    Motivation: a hierarchical Bayesian model would learn a country-level
    prior that admin-1 regions partially pool toward. These features are a
    tree-compatible approximation of that prior — cheaper to compute and
    interpretable as "how active is the rest of this country?".

    New columns (all lagged t-1 to prevent leakage):
        country_battles_excl (t-1)
        country_remote_excl (t-1)
        country_vac_excl (t-1)
        country_total_excl (t-1)

    Requires: 'Battles', 'Explosions/Remote violence',
              'Violence against civilians' columns in df (the target columns).
    """
    was_indexed = isinstance(df.index, pd.MultiIndex)
    df = df.copy()
    if was_indexed:
        df = df.reset_index()

    EVENT_COLS = [
        'Battles',
        'Explosions/Remote violence',
        'Violence against civilians',
    ]
    avail = [c for c in EVENT_COLS if c in df.columns]
    if not avail:
        raise ValueError(
            "[add_country_level_features] Target event columns not found. "
            "Ensure Battles / Explosions/Remote violence / Violence against civilians "
            "are present (they are saved as settings.targets in the baseline CSV)."
        )

    # Extract country ISO3 from "ISO3 - region name" pattern
    df['_country'] = df['matched_admin1_id'].str.split(' - ').str[0]

    # Country total per (country, month) across all admin-1 regions
    country_totals = (
        df.groupby(['_country', 'month_year'])[avail]
        .sum()
        .reset_index()
        .rename(columns={c: f'_ctry_{c}' for c in avail})
    )
    df = df.merge(country_totals, on=['_country', 'month_year'], how='left')

    # Leave-one-out: subtract the region's own contribution, clip at 0
    col_map = {
        'Battles':                     'country_battles_excl',
        'Explosions/Remote violence':  'country_remote_excl',
        'Violence against civilians':  'country_vac_excl',
    }
    for orig, new_col in col_map.items():
        if orig in df.columns:
            df[new_col] = (df[f'_ctry_{orig}'] - df[orig]).clip(lower=0)

    excl_cols = [v for v in col_map.values() if v in df.columns]
    df['country_total_excl'] = df[excl_cols].sum(axis=1)

    # Lag 1 month within each region — prevents leakage
    lag_src = excl_cols + ['country_total_excl']
    df = df.sort_values(['matched_admin1_id', 'month_year'])
    for col in lag_src:
        df[f'{col} (t-1)'] = (
            df.groupby('matched_admin1_id')[col].shift(1).fillna(0)
        )

    # Drop intermediate columns
    drop = lag_src + [f'_ctry_{c}' for c in avail] + ['_country']
    df = df.drop(columns=[c for c in drop if c in df.columns])

    if was_indexed:
        df = df.set_index(['matched_admin1_id', 'month_year'])
    return df
