predictors = [
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
    'majority_religion',
    'majority_pct',
    'minority1_religion',
    'minority1_pct',
    'minority2_religion',
    'minority2_pct',
    'nonreligpct',
    ]

# importance_weight is a recency-based sample weight for training, NOT a model feature.
# It is read directly from the dataframe in evaluators.py (sample_weight parameter).
# Including it as a predictor would leak temporal ordering into the model.

targets = [
    'Battles',
    'Explosions/Remote violence',
    'Violence against civilians'
    ]

# ── Enrichment feature groups ──────────────────────────────────────────────────
# Raw column names produced by add_holiday_features() — used directly as features
# in the v2 pipeline (no t-1 lag; holidays are deterministic calendar facts).
holiday_raw_cols = [
    'holiday_count_month', 'is_holiday_month',
    'christian_holiday_count', 'islam_holiday_count', 'shia_holiday_count',
    'hindu_holiday_count', 'buddhist_holiday_count', 'jewish_holiday_count',
    'cultural_holiday_count', 'nonreligious_holiday_count',
]
holiday_features = holiday_raw_cols  # v2: no lag, raw == feature

macro_raw_cols = ['inflation', 'youth_unemployment', 'income_inequality', 'income_level_code']

# Leakage-safe macro features: prior-year values (_py). income_level_code is a
# stable structural attribute and does not need a prior-year shift.
macro_features = [
    'inflation_py', 'youth_unemployment_py', 'income_inequality_py',
    'income_level_code',
]

# Risk indicator feature names are detected at runtime from column patterns
# (startswith 'risk_', endswith '(t-1)') since they depend on master_raw.csv content.
# Engineered feature names are defined in utils/data_cleaning.build_enhanced_features.
