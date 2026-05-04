"""
utils/features/hol_rel.py
--------------------------
Holiday × religion interaction features — adapted from Rosa-Branch.

Builds three layers:
  1. Relevance-weighted holidays  — holiday count × religion's population share
  2. Minority tension signal      — minority religion has holiday but isn't dominant
  3. Total mobilisation pressure  — composite of all weighted signals

Religion columns must still be in string form (WRP pct-column names) when
this function is called. Label encoding should happen after.

Works on a flat DataFrame (matched_admin1_id as column, not index level).
"""

import pandas as pd


# WRP denomination column → holiday count column
RELIGION_TO_HOLIDAY = {
    # Islam
    'islmsunpct':  'islam_holiday_count',
    'islmshipct':  'shia_holiday_count',
    'islmibdpct':  'islam_holiday_count',
    'islmnatpct':  'islam_holiday_count',
    'islmalwpct':  'islam_holiday_count',
    'islmahmpct':  'islam_holiday_count',
    'islmothrpct': 'islam_holiday_count',
    'islmgenpct':  'islam_holiday_count',
    # Christianity
    'chrstprotpct': 'christian_holiday_count',
    'chrstcatpct':  'christian_holiday_count',
    'chrstorthpct': 'christian_holiday_count',
    'chrstangpct':  'christian_holiday_count',
    'chrstothrpct': 'christian_holiday_count',
    'chrstgenpct':  'christian_holiday_count',
    # Judaism
    'judorthpct':  'jewish_holiday_count',
    'judconspct':  'jewish_holiday_count',
    'judrefpct':   'jewish_holiday_count',
    'judothrpct':  'jewish_holiday_count',
    'judgenpct':   'jewish_holiday_count',
    # Hinduism
    'hindgenpct':  'hindu_holiday_count',
    'jaingenpct':  'hindu_holiday_count',
    # Buddhism
    'budmahpct':   'buddhist_holiday_count',
    'budthrpct':   'buddhist_holiday_count',
    'budothrpct':  'buddhist_holiday_count',
    'budgenpct':   'buddhist_holiday_count',
    # Cultural / other
    'syncgenpct':  'cultural_holiday_count',
    'anmgenpct':   'cultural_holiday_count',
    'zorogenpct':  'cultural_holiday_count',
    'bahgenpct':   'cultural_holiday_count',
    'taogenpct':   'cultural_holiday_count',
    'confgenpct':  'cultural_holiday_count',
    'shntgenpct':  'cultural_holiday_count',
    'othrgenpct':  'cultural_holiday_count',
    'nonreligpct': 'nonreligious_holiday_count',
}

HOL_REL_FEATURE_NAMES = [
    'weighted_holidays_majority',
    'weighted_holidays_minority1',
    'weighted_holidays_minority2',
    'minority_tension_minority1',
    'minority_tension_minority2',
    'minority_tension_total',
    'total_religious_mobilization',
]


def _weighted_holidays(df, religion_col, pct_col, out_col):
    """Vectorised: for each row, find the matching holiday column and weight by pct."""
    rel_series = df[religion_col].str.lower().str.strip() if df[religion_col].dtype == object \
        else df[religion_col].astype(str).str.lower().str.strip()
    result = pd.Series(0.0, index=df.index)
    for rel_name, hol_col in RELIGION_TO_HOLIDAY.items():
        if hol_col not in df.columns:
            continue
        mask = rel_series == rel_name
        if mask.any():
            result[mask] = df.loc[mask, hol_col] * df.loc[mask, pct_col].fillna(0)
    df[out_col] = result


def _minority_tension(df, minority_rel_col, minority_pct_col, out_col):
    """Holiday occurs for a minority religion that is NOT the majority."""
    min_series = df[minority_rel_col].str.lower().str.strip() if df[minority_rel_col].dtype == object \
        else df[minority_rel_col].astype(str).str.lower().str.strip()
    maj_series = df['majority_religion'].str.lower().str.strip() if df['majority_religion'].dtype == object \
        else df['majority_religion'].astype(str).str.lower().str.strip()
    result = pd.Series(0.0, index=df.index)
    for rel_name, hol_col in RELIGION_TO_HOLIDAY.items():
        if hol_col not in df.columns:
            continue
        mask = (min_series == rel_name) & (maj_series != rel_name)
        if mask.any():
            result[mask] = df.loc[mask, hol_col] * df.loc[mask, minority_pct_col].fillna(0)
    df[out_col] = result


def add_holiday_religion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add religion-weighted holiday interaction features.

    Requires religion columns to still be in string (WRP) form.
    Returns df with 7 new columns (current month only — holidays are
    calendar facts, no lag needed per V2 pipeline philosophy).
    """
    df = df.copy()

    required = ['majority_religion', 'minority1_religion', 'minority2_religion',
                'majority_pct', 'minority1_pct', 'minority2_pct']
    if not all(c in df.columns for c in required):
        return df

    _weighted_holidays(df, 'majority_religion',  'majority_pct',  'weighted_holidays_majority')
    _weighted_holidays(df, 'minority1_religion', 'minority1_pct', 'weighted_holidays_minority1')
    _weighted_holidays(df, 'minority2_religion', 'minority2_pct', 'weighted_holidays_minority2')

    _minority_tension(df, 'minority1_religion', 'minority1_pct', 'minority_tension_minority1')
    _minority_tension(df, 'minority2_religion', 'minority2_pct', 'minority_tension_minority2')

    df['minority_tension_total']        = (df['minority_tension_minority1'] +
                                           df['minority_tension_minority2'])
    df['total_religious_mobilization']  = (df['weighted_holidays_majority'] +
                                           df['weighted_holidays_minority1'] +
                                           df['weighted_holidays_minority2'])
    return df
