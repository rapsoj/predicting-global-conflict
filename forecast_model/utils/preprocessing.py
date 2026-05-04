import os
from pathlib import Path
import pandas as pd
import geopandas as gpd
from utils import data_cleaning, map_admin_regions
from utils.risk_merge import RiskIndicatorMerger
from utils.features.holidays import add_holiday_features
from utils.features.worldbank import add_worldbank_features
from utils.features.religion import add_religion_features
from utils.features.improved_features import (
    add_cyclic_time_features, add_country_level_features,
    PREDICTORS_V2, COUNTRY_FEATURES, HOLIDAY_FEATURES_V2,
)
from utils.fetch_world_bank_data import WorldBankDataFetcher
from config import settings

_ROOT         = Path(__file__).parent.parent
_BASELINE_OUT = _ROOT / "data/processed/model_data.csv"
_ENRICHED_OUT = _ROOT / "data/processed/model_data_risk_macro_holidays_engineered.csv"
_V2_OUT       = _ROOT / "data/processed/model_data_v2_enriched.csv"
_TMP_BASE     = _ROOT / "data/processed/_tmp_v2_baseline.csv"
_RAW_CSV      = _ROOT / "data/raw/1997-01-01-2025-07-03.csv"
_BOUNDARIES   = _ROOT / "data/raw/boundaries/ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp"


def _build_combined():
    """
    Shared internal step: load raw ACLED, build event/neighbour tables,
    add lagged columns, time features, and importance weights.

    Returns (combined, gdf) where combined has MultiIndex (matched_admin1_id, month_year).
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

    # Get World Bank data
    wb_dir = "data/raw/world_bank"
    indicators_path = os.path.join(wb_dir, "combined_indicators.csv")
    metadata_path = os.path.join(wb_dir, "country_metadata.csv")

    if not os.path.exists(indicators_path) or not os.path.exists(metadata_path):
        wb = WorldBankDataFetcher()
        countries = wb.get_countries()
        data = wb.get_all_indicators()

        wb.save_data(data, countries, wb_dir)
        countries.to_csv(metadata_path, index=False)

    combined = pd.concat([event_data, subevent_data], axis=1).join(neighbour_data, how='left')
    combined = data_cleaning.add_lagged_columns(combined)
    combined = add_cyclic_time_features(combined)
    combined = data_cleaning.add_importance_weights(combined)
    return combined, gdf


def prepare_data_pipeline(clean_data: bool = False) -> pd.DataFrame:
    """
    Baseline pipeline: 30 ACLED predictors + targets + importance_weight.
    Saves to data/processed/model_data.csv.
    """
    if not clean_data and _BASELINE_OUT.exists():
        print("Loading baseline data from disk...")
        return pd.read_csv(_BASELINE_OUT, index_col=[0, 1])

    print("Building baseline dataset...")
    combined, _ = _build_combined()

    keep = settings.predictors + settings.targets + ['importance_weight']
    model_data = combined[[c for c in keep if c in combined.columns]]

    _BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_csv(_BASELINE_OUT)
    print(f"Saved: {_BASELINE_OUT}")
    return model_data


def prepare_enriched_pipeline(
    clean_data: bool = False,
    master_raw_csv: str = "data/raw/master_raw.csv",
    indicators_csv: str = "data/raw/world_bank/combined_indicators.csv",
    metadata_csv:   str = "data/raw/world_bank/country_metadata.csv",
    holidays_csv:   str = "data/raw/holidays_raw.csv",
) -> pd.DataFrame:
    """
    Full enrichment pipeline — adds on top of the baseline in sequence:
      1. Risk indicators (CAST signals, lagged t-1)
      2. World Bank macro indicators (prior-year, anti-leakage)
      3. Holiday features (lagged t-1)
      4. Engineered features (lag-2, rolling averages, interactions)

    Also saves the baseline (model_data.csv) as a side effect so both
    datasets are always available on disk.

    Saves to data/processed/model_data_risk_macro_holidays_engineered.csv.
    """
    if not clean_data and _ENRICHED_OUT.exists():
        print("Loading enriched data from disk...")
        return pd.read_csv(_ENRICHED_OUT)

    print("Building enriched dataset...")
    combined, gdf = _build_combined()

    # ── Save baseline as a side effect ────────────────────────────────────
    keep = settings.predictors + settings.targets + ['importance_weight']
    baseline = combined[[c for c in keep if c in combined.columns]]
    _BASELINE_OUT.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(_BASELINE_OUT)
    print(f"  Saved baseline: {_BASELINE_OUT}")

    # ── 1. Risk indicators (RiskIndicatorMerger reads from disk) ──────────
    print("  Merging risk indicators...")
    merger = RiskIndicatorMerger(lag=1)
    df = merger.merge(_BASELINE_OUT, master_raw_csv)
    # df is now a flat DataFrame; set MultiIndex for the feature functions
    df = df.set_index(["matched_admin1_id", "month_year"])

    # ── 2. World Bank macro indicators ────────────────────────────────────
    print("  Adding macro indicators...")
    df = add_worldbank_features(df, gdf,
                                indicators_path=indicators_csv,
                                metadata_path=metadata_csv)
    df = df.sort_index()
    raw_to_py = {r: p for r, p in
                 zip(['inflation', 'youth_unemployment', 'income_inequality'],
                     ['inflation_py', 'youth_unemployment_py', 'income_inequality_py'])}
    for raw_col, py_col in raw_to_py.items():
        if raw_col in df.columns:
            shifted = df.groupby(level='matched_admin1_id')[raw_col].shift(12)
            year_key = df.index.get_level_values('month_year').str[:4].astype(int)
            year_medians = shifted.groupby(year_key).transform('median')
            df[py_col] = shifted.fillna(year_medians).fillna(shifted.median())

    if 'income_level_code' in df.columns:
        median_level = df['income_level_code'].median()
        df['income_level_code'] = df['income_level_code'].fillna(median_level)

    # ── 3. Holiday features (current month, no lag) ────────────────────────
    # Holidays are deterministic calendar facts known in advance — no lag needed.
    print("  Adding holiday features...")
    df = add_holiday_features(df, gdf)
    df = df.sort_index()

    print("  Adding religion features...")
    df = add_religion_features(df)

    # ── 4. Country-level hierarchy features ───────────────────────────────
    print("  Adding country-level hierarchy features...")
    df = add_country_level_features(df)

    # ── 5. Engineered features ─────────────────────────────────────────────
    print("  Building engineered features...")
    df = data_cleaning.build_enhanced_features(df.reset_index())

    _ENRICHED_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_ENRICHED_OUT, index=False)
    print(f"  Saved enriched: {_ENRICHED_OUT}")
    return df


def filter_admin1_data(df, admin1_region):
    return df.loc[admin1_region]


# ── V2 pipeline ───────────────────────────────────────────────────────────────
# Merged from preprocessing_v2.py. _build_combined() is shared — both pipelines
# now use cyclic time encoding, so no separate _build_combined_v2 is needed.

def prepare_enriched_pipeline_v2(
    clean_data: bool = False,
    master_raw_csv: str = "data/raw/master_raw.csv",
    indicators_csv: str = "data/raw/world_bank/combined_indicators.csv",
    metadata_csv:   str = "data/raw/world_bank/country_metadata.csv",
    holidays_csv:   str = "data/raw/holidays_raw.csv",
) -> pd.DataFrame:
    """
    V2 enrichment pipeline. Identical enrichment sequence to prepare_enriched_pipeline
    but returns a flat DataFrame and caches to model_data_v2_enriched.csv.

    Enrichment sequence:
      1. Risk indicators (CAST signals, t-1)
      2. World Bank macro indicators (prior-year, anti-leakage)
      3. Current-month holidays (no lag — deterministic calendar facts)
      4. Religion features (structural country-level context)
      5. Country-level leave-one-out aggregate (t-1)
      6. Engineered features (lag-2, rolling averages, interactions)
    """
    if not clean_data and _V2_OUT.exists():
        print("Loading v2 enriched data from disk...")
        return pd.read_csv(_V2_OUT)

    master_raw_csv = str(_ROOT / master_raw_csv) if not Path(master_raw_csv).is_absolute() else master_raw_csv
    indicators_csv = str(_ROOT / indicators_csv) if not Path(indicators_csv).is_absolute() else indicators_csv
    metadata_csv   = str(_ROOT / metadata_csv)   if not Path(metadata_csv).is_absolute()   else metadata_csv

    print("Building v2 enriched dataset...")
    combined, gdf = _build_combined()

    keep = PREDICTORS_V2 + settings.targets + ['importance_weight']
    baseline_tmp = combined[[c for c in keep if c in combined.columns]]
    _TMP_BASE.parent.mkdir(parents=True, exist_ok=True)
    baseline_tmp.to_csv(_TMP_BASE)

    print("  [v2] Merging risk indicators...")
    merger = RiskIndicatorMerger(lag=1)
    df = merger.merge(str(_TMP_BASE), master_raw_csv)
    _TMP_BASE.unlink(missing_ok=True)
    df = df.set_index(["matched_admin1_id", "month_year"])

    print("  [v2] Adding macro indicators...")
    df = add_worldbank_features(df, gdf,
                                indicators_path=indicators_csv,
                                metadata_path=metadata_csv)
    df = df.sort_index()
    for raw_col, py_col in [('inflation', 'inflation_py'),
                             ('youth_unemployment', 'youth_unemployment_py'),
                             ('income_inequality', 'income_inequality_py')]:
        if raw_col in df.columns:
            shifted      = df.groupby(level='matched_admin1_id')[raw_col].shift(12)
            year_key     = df.index.get_level_values('month_year').str[:4].astype(int)
            year_medians = shifted.groupby(year_key).transform('median')
            df[py_col]   = shifted.fillna(year_medians).fillna(shifted.median())
    if 'income_level_code' in df.columns:
        df['income_level_code'] = df['income_level_code'].fillna(df['income_level_code'].median())

    print("  [v2] Adding current-month holiday features (no lag)...")
    df = add_holiday_features(df, gdf)
    df = df.sort_index()

    print("  [v2] Adding religion features...")
    df = add_religion_features(df)

    print("  [v2] Adding country-level hierarchy features...")
    df = add_country_level_features(df)

    print("  [v2] Building engineered features...")
    df = data_cleaning.build_enhanced_features(df.reset_index())

    _V2_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(_V2_OUT, index=False)
    print(f"  Saved v2 enriched: {_V2_OUT}")
    return df


def build_feature_sets_v2(df: pd.DataFrame) -> dict:
    """
    Cumulative feature sets for the v2 ablation benchmark.
    Tiers: Baseline → +Risk → +Macro → +Holidays → +Country → +Engineered.
    """
    risk_features = [c for c in df.columns
                     if c.startswith("risk_") and c.endswith("(t-1)")]
    engineered_features = [f for f in [
        "Battles (t-2)", "Explosions/Remote violence (t-2)",
        "Violence against civilians (t-2)",
        "organized_violence (t-1)", "is_active (t-1)", "battles_x_remote (t-1)",
        "Battles_3mo_avg (t-1)", "Remote_3mo_avg (t-1)", "VaC_3mo_avg (t-1)",
    ] if f in df.columns]
    country_feats = [f for f in COUNTRY_FEATURES if f in df.columns]

    return {
        "Baseline":    PREDICTORS_V2,
        "+Risk":       PREDICTORS_V2 + risk_features,
        "+Macro":      PREDICTORS_V2 + risk_features + settings.macro_features,
        "+Holidays":   PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2,
        "+Country":    PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2 + country_feats,
        "+Engineered": PREDICTORS_V2 + risk_features + settings.macro_features + HOLIDAY_FEATURES_V2 + country_feats + engineered_features,
    }