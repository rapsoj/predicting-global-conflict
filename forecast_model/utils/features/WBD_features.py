"""
utils/features/WBD_features.py
-------------------------------
World Bank engineered features — adapted from Rosa-Branch.

Conceptual model:
    Structural pressure:    income_inequality
    Mobilization capacity:  youth_unemployment
    Shock trigger:          inflation
    Institutional buffer:   income_level_code

All inputs use prior-year (_py) versions to avoid leakage.
income_level_code is structural (no lag needed): LIC=1, LMC=2, UMC=3, HIC=4.
"""

import numpy as np
import pandas as pd


WBD_FEATURE_NAMES = [
    'structural_vulnerability',
    'inflation_shock',
    'shock_vulnerability',
    'capacity_adjusted_risk',
    'inflation_change',
    'high_inequality_flag',
    'inflation_shock_poor',
    'conflict_momentum',
    'violence_escalation',
    'macro_conflict_pressure',
]


def add_worldbank_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 10 engineered World Bank features on top of the enriched DataFrame.

    Expects columns produced by add_worldbank_features() + the _py leakage-safe
    variants computed in prepare_enriched_pipeline_v2():
        inflation_py, youth_unemployment_py, income_inequality_py,
        income_level_code

    Works on a flat DataFrame (matched_admin1_id as a column, not index level).
    """
    df = df.copy()

    ineq   = df['income_inequality_py'].fillna(0)
    youth  = df['youth_unemployment_py'].fillna(0)
    infl   = df['inflation_py'].fillna(0)
    income = df['income_level_code'].fillna(2)   # default LMC if missing

    # Structural pressure × mobilisation capacity
    df['structural_vulnerability'] = ineq * youth

    # Binary shock flag: top-10% inflation months
    threshold = infl.quantile(0.9)
    df['inflation_shock'] = (infl > threshold).astype(int)

    # Shock amplified by structural vulnerability
    df['shock_vulnerability'] = df['inflation_shock'] * df['structural_vulnerability']

    # Institutional buffer: higher income → lower effective risk
    # income_level_code is 1-4; +3 keeps denominator safely above 0 (range 4-7)
    df['capacity_adjusted_risk'] = df['structural_vulnerability'] / (income + 3)

    # Inflation trend: month-to-month change in prior-year inflation per region
    df_sorted = df.sort_values(['matched_admin1_id', 'month_year'])
    df['inflation_change'] = (
        df_sorted.groupby('matched_admin1_id')['inflation_py']
        .diff()
        .reindex(df.index)
    )

    # Threshold effect: top-25% inequality countries
    df['high_inequality_flag'] = (ineq > ineq.quantile(0.75)).astype(int)

    # Inflation shock concentrated in low/lower-middle income countries
    # LIC=1, LMC=2 → poor; UMC=3, HIC=4 → not poor
    df['inflation_shock_poor'] = (
        df['inflation_shock'] * (income <= 2).astype(int)
    )

    # Conflict momentum: recent violence in region + neighbours
    df['conflict_momentum'] = (
        df.get('Battles (t-1)', 0) +
        df.get('Violence against civilians (t-1)', 0) +
        df.get('Battles_neighbours (t-1)', 0) +
        df.get('Violence against civilians_neighbours (t-1)', 0)
    )

    # Violent vs non-violent activity ratio
    violent     = df.get('Violence against civilians (t-1)', 0) + df.get('Battles (t-1)', 0)
    non_violent = df.get('Protests (t-1)', 0) + df.get('Riots (t-1)', 0) + 1
    df['violence_escalation'] = violent / non_violent

    # Combined macro × conflict pressure
    df['macro_conflict_pressure'] = (
        df['structural_vulnerability'] * df['conflict_momentum']
    )

    return df.fillna(0)
