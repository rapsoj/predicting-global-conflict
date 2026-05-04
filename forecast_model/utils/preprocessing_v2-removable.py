"""
utils/preprocessing_v2.py
--------------------------
V2 preprocessing pipeline incorporating three improvements over the baseline:

  1. Cyclic time encoding   — month_sin/cos, quarter_sin/cos instead of 14 dummies
  2. Current-month holidays — holiday_count_month at t (no t-1 lag)
  3. Country-level hierarchy — leave-one-out country aggregate, lagged t-1

Entry point:
    prepare_enriched_pipeline_v2(clean_data=False) -> pd.DataFrame

Saves to:  data/processed/model_data_v2_enriched.csv

Feature sets produced by build_feature_sets_v2():
    Baseline    — 20 ACLED predictors (cyclic time replaces 14 dummies)
    +Risk       — + 7 CAST risk indicators (t-1)
    +Macro      — + inflation, youth_unemployment, income_inequality (prior-year), income_level_code
    +Holidays   — + all holiday features (current month, no lag)
    +Country    — + 4 country-level leave-one-out aggregates (t-1)
    +Engineered — + lag-2, rolling averages, co-escalation interaction
"""

import os
from pathlib import Path
import pandas as pd
import geopandas as gpd

from utils import data_cleaning, map_admin_regions
from utils.risk_merge import RiskIndicatorMerger
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from utils.features.religion import add_religion_features
from utils.fetch_world_bank_data import WorldBankDataFetcher
from utils.features.improved_features import (
    add_cyclic_time_features,
    add_country_level_features,
    PREDICTORS_V2,
    COUNTRY_FEATURES,
    HOLIDAY_FEATURES_V2,
)
from config import settings

_ROOT       = Path(__file__).parent.parent
_V2_OUT     = _ROOT / "data/processed/model_data_v2_enriched.csv"
_RAW_CSV    = _ROOT / "data/raw/1997-01-01-2025-07-03.csv"
_BOUNDARIES = _ROOT / "data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"
_TMP_BASE   = _ROOT / "data/processed/_tmp_v2_baseline.csv"


def _build_combined_v2():
    """
    Shared base step — identical to preprocessing._build_combined() except
    cyclic time features replace month/quarter dummies.
    """
    df = pd.read_csv(_RAW_CSV)
    df = df[df['year'] >= 2018].copy()
    df['date'] = pd.to_datetime(df['event_date'], format='%d %B %Y')
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    # admin1 column naming fix
    if 'country_code' not in df.columns:
        df['country_code'] = df['country']
    if 'admin1' not in df.columns:
        if 'admin1_name' in df.columns:
            df['admin1'] = df['admin1_name']
        elif 'admin1_region' in df.columns:
            df['admin1'] = df['admin1_region']

    gdf = gpd.read_file(_BOUNDARIES)
    df_neighbours = map_admin_regions.add_admin1_neighbors(df, gdf)
    df_neighbours = df_neighbours.dropna(subset=['matched_admin1_id']).copy()

    neighbour_data = data_cleaning.summarise_neighbour_events(df_neighbours)
    event_data     = data_cleaning.get_monthly_events(df_neighbours)
    subevent_data  = data_cleaning.get_monthly_subevents(
        df_neighbours, ['Excessive force against protesters', 'Agreement']
    )

    wb_dir = "data/raw/world_bank"
    indicators_path = os.path.join(wb_dir, "combined_indicators.csv")
    metadata_path   = os.path.join(wb_dir, "country_metadata.csv")
    if not os.path.exists(indicators_path) or not os.path.exists(metadata_path):
        wb = WorldBankDataFetcher()
        countries = wb.get_countries()
        data = wb.get_all_indicators()
        wb.save_data(data, countries, wb_dir)
        countries.to_csv(metadata_path, index=False)

    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    combined = data_cleaning.add_lagged_columns(combined)
    combined = add_cyclic_time_features(combined)       # v2: cyclic instead of dummies
    combined = data_cleaning.add_importance_weights(combined)
    return combined, gdf


def prepare_enriched_pipeline_v2(
    clean_data: bool = False,
    master_raw_csv: str = "data/raw/master_raw.csv",
    indicators_csv: str = "data/raw/world_bank/combined_indicators.csv",
    metadata_csv:   str = "data/raw/world_bank/country_metadata.csv",
    holidays_csv:   str = "data/raw/holidays_raw.csv",
) -> pd.DataFrame:
    """
    Full v2 enrichment pipeline.

    Enrichment sequence:
      1. Risk indicators (CAST signals, t-1)
      2. World Bank macro indicators (prior-year)
      3. Holidays at time t (no lag — deterministic calendar knowledge)
      4. Religion features (structural country-level context)
      5. Country-level leave-one-out aggregate (t-1)
      6. Engineered features (lag-2, rolling averages, interactions)

    Returns a flat DataFrame (not multi-indexed) for compatibility with
    ablation / evaluator utilities.
    """
    if not clean_data and _V2_OUT.exists():
        print("Loading v2 enriched data from disk...")
        return pd.read_csv(_V2_OUT)

    # Resolve relative paths against the project root
    master_raw_csv = str(_ROOT / master_raw_csv) if not Path(master_raw_csv).is_absolute() else master_raw_csv
    indicators_csv = str(_ROOT / indicators_csv) if not Path(indicators_csv).is_absolute() else indicators_csv
    metadata_csv   = str(_ROOT / metadata_csv)   if not Path(metadata_csv).is_absolute()   else metadata_csv
    holidays_csv   = str(_ROOT / holidays_csv)   if not Path(holidays_csv).is_absolute()   else holidays_csv

    print("Building v2 enriched dataset...")
    combined, gdf = _build_combined_v2()

    # ── Save temporary baseline for RiskIndicatorMerger ──────────────────────
    keep = PREDICTORS_V2 + settings.targets + ['importance_weight']
    baseline_tmp = combined[[c for c in keep if c in combined.columns]]
    _TMP_BASE.parent.mkdir(parents=True, exist_ok=True)
    baseline_tmp.to_csv(_TMP_BASE)

    # ── 1. Risk indicators ────────────────────────────────────────────────────
    print("  [v2] Merging risk indicators...")
    merger = RiskIndicatorMerger(lag=1)
    df = merger.merge(str(_TMP_BASE), master_raw_csv)
    _TMP_BASE.unlink(missing_ok=True)
    df = df.set_index(["matched_admin1_id", "month_year"])

    # ── 2. World Bank macro indicators ───────────────────────────────────────
    print("  [v2] Adding macro indicators...")
    df = add_worldbank_features(df, gdf,
                                indicators_path=indicators_csv,
                                metadata_path=metadata_csv)
    df = df.sort_index()
    raw_to_py = {
        'inflation':          'inflation_py',
        'youth_unemployment': 'youth_unemployment_py',
        'income_inequality':  'income_inequality_py',
    }
    for raw_col, py_col in raw_to_py.items():
        if raw_col in df.columns:
            shifted      = df.groupby(level='matched_admin1_id')[raw_col].shift(12)
            year_key     = df.index.get_level_values('month_year').str[:4].astype(int)
            year_medians = shifted.groupby(year_key).transform('median')
            df[py_col]   = shifted.fillna(year_medians).fillna(shifted.median())
    if 'income_level_code' in df.columns:
        df['income_level_code'] = df['income_level_code'].fillna(df['income_level_code'].median())

    # ── 3. Current-month holidays (NO lag) ───────────────────────────────────
    # Holidays are a known calendar fact — use the actual count at time t.
    # add_holiday_features() returns all holiday columns at time t;
    # we deliberately skip the shift(1) used in the v1 pipeline.
    print("  [v2] Adding current-month holiday features (no lag)...")
    df = add_holiday_features(df, gdf)
    df = df.sort_index()

    # ── 4. Religion features ─────────────────────────────────────────────────
    print("  [v2] Adding religion features...")
    df = add_religion_features(df)

    # ── 5. Country-level hierarchy features ──────────────────────────────────
    print("  [v2] Adding country-level hierarchy features...")
    df = add_country_level_features(df)

    # ── 6. Engineered features ────────────────────────────────────────────────
    print("  [v2] Building engineered features...")
    df = data_cleaning.build_enhanced_features(df.reset_index())

    _V2_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_V2_OUT, index=False)
    print(f"  Saved v2 enriched: {_V2_OUT}")
    return df


def build_feature_sets_v2(df: pd.DataFrame) -> dict:
    """
    Cumulative feature sets for the v2 ablation benchmark.

    Mirrors ablation.build_feature_sets() but uses v2 predictor lists and
    adds a +Country tier between +Holidays and +Engineered.
    """
    risk_features = [
        c for c in df.columns
        if c.startswith("risk_") and c.endswith("(t-1)")
    ]
    if not risk_features:
        print("WARNING: No risk_ (t-1) columns found. "
              "'+Risk' and higher sets will equal 'Baseline'.")

    engineered_features = [
        f for f in [
            "Battles (t-2)", "Explosions/Remote violence (t-2)",
            "Violence against civilians (t-2)",
            "organized_violence (t-1)", "is_active (t-1)", "battles_x_remote (t-1)",
            "Battles_3mo_avg (t-1)", "Remote_3mo_avg (t-1)", "VaC_3mo_avg (t-1)",
        ]
        if f in df.columns
    ]

    country_feats = [f for f in COUNTRY_FEATURES if f in df.columns]

    return {
        "Baseline":    PREDICTORS_V2,
        "+Risk":       PREDICTORS_V2 + risk_features,
        "+Macro":      PREDICTORS_V2 + risk_features + settings.macro_features,
        "+Holidays":   PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2,
        "+Country":    PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2 + country_feats,
        "+Engineered": PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2 + country_feats + engineered_features,
    }
