"""
utils/features/encoding.py
--------------------------
Feature encoding improvements used in the V3 (improvements_6) benchmark:

  1. encode_religion_columns  — WRP string codes (e.g. 'islmsunpct') → integers
  2. apply_log_transforms     — log1p on heavy-tailed count predictors
  3. add_religion_weighted_holidays — single relevance score weighting each
                                      religion's holiday count by the country's
                                      fraction following that religion
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── V3: Per-split and global label encoding (Rosa-style) ─────────────────────

def encode_categoricals(X_train: pd.DataFrame, X_test: pd.DataFrame,
                        cols: list) -> tuple:
    """
    Fit LabelEncoder on training data only, then apply to test.

    Unknown test values (and missing values) → -1.
    Matches Rosa's per-split encode_categoricals() approach — no leakage.

    Parameters
    ----------
    X_train, X_test : feature DataFrames (copies are made internally).
    cols            : column names to encode (skips columns not present).

    Returns
    -------
    (X_train_enc, X_test_enc) with integer-encoded columns.
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
            lambda x, k=known, e=le: e.transform([x])[0] if x in k else -1
        )

    return X_train, X_test


def apply_global_label_encoding(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Encode label columns globally (fit_transform on the whole DataFrame).

    Safe for static attributes like country_code and religion that do not
    vary over time within a region. Returns a copy with integer columns.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        le = LabelEncoder()
        vals = df[col].fillna('__MISSING__').astype(str)
        df[col] = le.fit_transform(vals).astype(int)
    return df


# ── Count columns that benefit from log1p (zero-inflated, heavy right tail) ──

_COUNT_COLS = [
    'Battles (t-1)', 'Explosions/Remote violence (t-1)', 'Protests (t-1)',
    'Riots (t-1)', 'Strategic developments (t-1)', 'Violence against civilians (t-1)',
    'Excessive force against protesters (t-1)', 'Agreement (t-1)',
    'Explosions/Remote violence_neighbours (t-1)',
    'Strategic developments_neighbours (t-1)',
    'Protests_neighbours (t-1)', 'Violence against civilians_neighbours (t-1)',
    'Battles_neighbours (t-1)', 'Riots_neighbours (t-1)',
    'country_battles_excl (t-1)', 'country_remote_excl (t-1)',
    'country_vac_excl (t-1)', 'country_total_excl (t-1)',
    'Battles (t-2)', 'Explosions/Remote violence (t-2)',
    'Violence against civilians (t-2)',
    'organized_violence (t-1)', 'battles_x_remote (t-1)',
    'Battles_3mo_avg (t-1)', 'Remote_3mo_avg (t-1)', 'VaC_3mo_avg (t-1)',
]

# ── WRP pct-column name → corresponding holiday column ───────────────────────

_RELIGION_TO_HOLIDAY = {
    # Christian variants
    'chrstprotpct': 'christian_holiday_count',
    'chrstcatpct':  'christian_holiday_count',
    'chrstorthpct': 'christian_holiday_count',
    'chrstgenpct':  'christian_holiday_count',
    'chrstangpct':  'christian_holiday_count',
    'chrstothrpct': 'christian_holiday_count',
    # Islam (Sunni + generic)
    'islmsunpct':   'islam_holiday_count',
    'islmibapct':   'islam_holiday_count',
    'islmnatpct':   'islam_holiday_count',
    'islmothpct':   'islam_holiday_count',
    # Islam (Shia)
    'islmshipct':   'shia_holiday_count',
    # Hindu
    'hindgenpct':   'hindu_holiday_count',
    'hindvaishnpct':'hindu_holiday_count',
    'hindothpct':   'hindu_holiday_count',
    # Buddhist
    'budmaypct':    'buddhist_holiday_count',
    'budtherpct':   'buddhist_holiday_count',
    'budothrpct':   'buddhist_holiday_count',
    # Jewish
    'judorthodpct': 'jewish_holiday_count',
    'judconserpct': 'jewish_holiday_count',
    'judrefpct':    'jewish_holiday_count',
    'judothpct':    'jewish_holiday_count',
    # Non-religious / secular
    'norelpct':     'nonreligious_holiday_count',
    'nonreligpct':  'nonreligious_holiday_count',
}

_RELIGION_COLS = ['majority_religion', 'minority1_religion', 'minority2_religion']
_RELIGION_PCTS = ['majority_pct',      'minority1_pct',      'minority2_pct']


# ── Public API ────────────────────────────────────────────────────────────────

def encode_religion_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert WRP religion string codes to integer labels.

    All three religion rank columns (majority, minority1, minority2) share a
    single fitted LabelEncoder so the integer space is consistent.  NaN → -1.
    Returns a copy; does not modify the caller's frame.
    """
    df = df.copy()
    present = [c for c in _RELIGION_COLS if c in df.columns]
    if not present:
        return df

    all_vals = pd.concat([df[c].dropna() for c in present]).astype(str).unique()
    le = LabelEncoder().fit(all_vals)

    for col in present:
        mask = df[col].notna()
        df.loc[mask, col] = le.transform(df.loc[mask, col].astype(str))
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)

    return df


def apply_log_transforms(df: pd.DataFrame) -> tuple:
    """
    Add log1p-transformed columns for all count-based predictors.

    Risk indicator columns (pattern: risk_* (t-1)) are auto-detected and
    included in addition to the static list in _COUNT_COLS.

    Returns
    -------
    df_new : pd.DataFrame
        Copy of df with additional 'log1p_{col}' columns appended.
    col_map : dict
        {original_col: log1p_col} for every transformed column.
    """
    df = df.copy()
    risk_cols = [c for c in df.columns if c.startswith('risk_') and c.endswith('(t-1)')]
    to_transform = [c for c in _COUNT_COLS + risk_cols if c in df.columns]

    col_map = {}
    for col in to_transform:
        new_col = f'log1p_{col}'
        df[new_col] = np.log1p(df[col].fillna(0).clip(lower=0))
        col_map[col] = new_col

    return df, col_map


def add_religion_weighted_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'holiday_relevance_score': a single per-row value equal to

        Σ_i  religion_i_pct  ×  holiday_count_for_religion_i

    summed over majority + two minority religion tiers.  Captures how many
    *relevant* holidays occur this month given the country's religious makeup.
    Uses vectorised operations; safe for DataFrames of any size.
    """
    df = df.copy()
    score = pd.Series(0.0, index=df.index)

    for rel_col, pct_col in zip(_RELIGION_COLS, _RELIGION_PCTS):
        if rel_col not in df.columns or pct_col not in df.columns:
            continue
        pct = df[pct_col].fillna(0)
        for wrp_name, hcol in _RELIGION_TO_HOLIDAY.items():
            if hcol not in df.columns:
                continue
            mask = df[rel_col] == wrp_name
            if mask.any():
                score += mask.astype(float) * pct * df[hcol].fillna(0)

    df['holiday_relevance_score'] = score
    return df
