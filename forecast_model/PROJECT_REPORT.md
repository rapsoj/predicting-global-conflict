# CAST+: Giray's Alternative Model Stream — Internal Technical Log
**Joint paper:** "CAST+: Augmenting the ACLED Conflict Alert System" (Rosa Daneshmandnia & Giray Ünlü)  
**This document:** Internal log of Giray's LGBM-Tweedie ablation study (§5.4, §6.4, §7.5 of the joint paper)  
**Period:** February – April 2026  
**Role of this work in the paper:** Alternative model stream. The primary CAST+ model is Rosa's Two-Stage CatBoost.  
  Giray's contribution is (1) the ablation study identifying which feature layers matter, and  
  (2) the country-level hierarchy feature engineering, tested across 5 model families.  
**Primary baseline for paper comparisons:** CAST RF (Random Forest, ACLED-only, 29 features) — MAE 67.97 on paper's top-10, MAE 2.344 globally  
**Secondary baseline:** Persistence (ŷ = y(t−1)) — MAE 45.92 on historical top-10, MAE 0.91 globally  
**Holdout window:** 6 months (chronological)  
**Primary metrics:** Mean Absolute Error (MAE); Directional Accuracy — DA-nonzero (changing months only, comparable to CatBoost's 79–82%)  
**Note on MAPE:** Not reported in paper-facing sections — paper §7.2 explicitly states MAPE is inappropriate for this data (86–92% zero-delta months cause division-by-zero instability)

---

## 1. Problem and Baseline

### 1.1 Task

Forecast monthly event counts at admin-1 (province/state) level, 1–6 months ahead.  
Three targets are treated as independent regression problems:

| Target | Nature |
|---|---|
| Battles | Armed clashes between organised groups |
| Explosions / Remote violence | Airstrikes, IEDs, shelling |
| Violence against civilians | Deliberate attacks on non-combatants |

Dataset: ACLED raw file covers **1 January 1997 – 20 June 2025** (2,619,146 raw events total).  
The pipeline filters to **2018–June 2025** (2,241,762 events, 85.6% of the raw file) before
aggregation. Pre-2018 data is discarded due to inconsistent global coverage in ACLED's
historical collection (1997–2015 records average ~10k events/year globally vs 290k+ from 2021 onwards).  
After monthly aggregation: **~278,821 region-month rows across ~3,097 global admin-1 regions.**

### 1.2 Evaluation Design

Three evaluation scopes are used. The paper's top-10 is the primary comparison tier for §6.4.

#### Paper's top-10 (primary comparison for §6.4)

- **Definition:** 10 most active regions by total event count **during the holdout period** (last 6 months). Matches Rosa's CatBoost evaluation exactly — makes §6.4 comparison valid.
- **Regions:** UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia, RUS-Kursk, UKR-Chernihiv, PSX-West Bank
- **Source:** `outputs/ablation_paper_top10.csv`

#### Historical top-10 (internal ablation benchmark, V1/V2 experiments)

- **Definition:** 10 most active regions by total event count **over the full date range**. Used for all V1/V2 ablation experiments (§4.1–§4.7). More diverse than the paper's top-10 — includes Syria, Brazil.
- **Regions:** UKR-Donetsk, UKR-Kharkiv, UKR-Sumy, UKR-Kherson, UKR-Zaporizhzhia, UKR-Luhansk, SYR-Idlib, SYR-Aleppo, PSX-Gaza, BRA-Rio de Janeiro
- **Note:** Results on this set are not directly comparable to Rosa's CatBoost numbers.

#### Full-region evaluation (global representativeness)

- **Evaluation set:** All 2,262 valid admin-1 regions (≥ 12 months data, non-zero training target). Comparable to Rosa's 2,205-region evaluation (57-region difference due to different validity filters — both correct for their respective model objectives).
- **Source:** `outputs/full_region_eval.csv`

#### Common settings

- **Split:** Chronological. Train = all months up to T−6. Test = last 6 months. No shuffling.
- **Metrics:**
  - **MAE:** Mean Absolute Error (events / region-month), averaged over regions × targets × 6 holdout months.
  - **DA-nonzero (paper-comparable):** Directional accuracy restricted to months where conflict actually changes (true delta ≠ 0). Formula: `mean(sign(ŷ − y_{t-1}) == sign(y − y_{t-1}))` over changing months only. This matches the basis of CatBoost's reported 79–82% DA and makes the two models directly comparable.
  - **DA (all months):** Same formula averaged over all holdout months including zero-change months. Reported for completeness but not used for cross-model comparison (definitions differ).
  - **MAPE:** Not reported in paper-facing sections — paper §7.2 states it is inappropriate for zero-inflated data.

### 1.3 Baselines

Two baselines are used, matching the joint paper's §6.1 reference points.

#### CAST RF Baseline (primary comparison)

The CAST baseline is a Random Forest trained on ACLED-only features (29 predictors, no enrichment). This is the starting point all CAST+ improvements are measured against.

| Evaluation scope | Overall MAE |
|---|---|
| Paper's top-10 regions | **67.97** (paper) / **69.47** (our re-run — within 2%) |
| All valid regions (2,205) | **2.344** (paper) |

> Our RF Baseline re-run on the paper's top-10 (69.47) is structurally equivalent to the
> paper's CAST RF (67.97). The small difference reflects implementation details, not a
> different model. For §6.4 comparisons, we use our re-run figure (69.47) for consistency.

#### Persistence Baseline (secondary)

The naive benchmark: predict next month's count equal to the current month's count (ŷ(t) = y(t−1)). In the delta-target framework (Rosa's model), this is equivalent to predicting ΔY = 0 every month — which achieves DA = 0 on all changing months by definition.

**Paper's top-10 regions (holdout-period):**

| Target | Persistence MAE |
|---|---|
| Battles | 45.83 |
| Explosions / Remote violence | 88.33 |
| Violence against civilians | 3.60 |
| **Overall** | **45.92** |

**All valid regions (2,262):**

| Target | Persistence MAE | DA (all months) |
|---|---|---|
| Battles | 0.88 | 0.83 |
| Explosions / Remote violence | 1.09 | 0.89 |
| Violence against civilians | 0.76 | 0.75 |
| **Overall** | **0.91** | **0.83** |

> Persistence DA is 0.83 globally only because 83% of region-months have zero conflict
> change — predicting "no change" is trivially correct on those months. On changing months,
> persistence DA is 0 by construction. This is why DA-nonzero is the meaningful metric for
> comparing trained models.

---

## 2. Feature Engineering

Features are organised into cumulative layers, each corresponding to one tier of the ablation design.

### Layer 1: Autoregressive (Lag-1)

One-month lagged counts for all 8 ACLED event types:

```
Battles (t-1), Explosions/Remote violence (t-1), Protests (t-1),
Riots (t-1), Strategic developments (t-1), Violence against civilians (t-1),
Excessive force against protesters (t-1), Agreement (t-1)
```

Plus lag-1 of event counts summed over all **bordering admin-1 regions** (spatial spill-over):

```
Battles_neighbours (t-1), Explosions/Remote violence_neighbours (t-1),
Protests_neighbours (t-1), Riots_neighbours (t-1),
Strategic developments_neighbours (t-1), Violence against civilians_neighbours (t-1)
```

Neighbour adjacency is derived from the Natural Earth admin-1 shapefile by polygon intersection.  
All lags are computed per-region to prevent cross-region leakage.

### Layer 2: Temporal

**V1 (binary dummies):** 11 month dummies + 3 quarter dummies + year + linear_month_trend = 16 features  
**V2 (cyclic encoding):** Replace 14 binary columns with 4 continuous sin/cos projections:

```
month_sin  = sin(2π × month / 12)      month_cos  = cos(2π × month / 12)
quarter_sin = sin(2π × quarter / 4)    quarter_cos = cos(2π × quarter / 4)
```

Retained: `year`, `linear_month_trend` (months elapsed since Jan 2018).

**Why cyclic encoding:** One-hot encoding places December and January at opposite ends
of the feature space. The sin/cos encoding correctly represents their adjacency on the
seasonal cycle and reduces 14 columns to 4, cutting model complexity.

**Baseline predictor set = Layers 1 + 2 + religion features (Layer 5). Total: 27 features.**

### Layer 3: Risk Indicators

Columns from `master_raw.csv` whose names match `risk_* (t-1)`. These are auto-detected
at runtime; their count depends on the external risk data source.

### Layer 4: Macroeconomic (World Bank)

| Feature | Source | Leakage handling |
|---|---|---|
| `inflation_py` | CPI annual % | Prior-year shift (WB data published ~12–18 months late) |
| `youth_unemployment_py` | Youth unemployment rate | Prior-year shift |
| `income_inequality_py` | Gini coefficient | Prior-year; sparse → within-country bfill/ffill |
| `income_level_code` | Income group (LIC=1…HIC=4) | Structural — no lag needed |

### Layer 5: Religious Composition (World Religion Project, 2010 snapshot)

For each country, the top-3 religion denominations by population share:

```
majority_religion, majority_pct, minority1_religion, minority1_pct,
minority2_religion, minority2_pct, nonreligpct
```

WRP string denomination codes (e.g. `islmsunpct`) are label-encoded to integers.

### Layer 6: Holiday Features

89,217 national holiday records are cleaned (34 spelling variants normalised),
religion-tagged, and aggregated to monthly country-level counts:

```
christian_holiday_count, islam_holiday_count, shia_holiday_count,
hindu_holiday_count, buddhist_holiday_count, jewish_holiday_count,
cultural_holiday_count, nonreligious_holiday_count,
holiday_count_month, is_holiday_month
```

**Lag policy:** V1 used t-1 lag. V2 uses no lag — holidays are calendar facts known in
advance, so the current-month value introduces zero leakage while being more informative.

### Layer 7: Country-Level Hierarchy (V2 Addition)

Four leave-one-out country aggregates, all lagged t-1:

```
country_battles_excl (t-1)  = Σ Battles(t-1) over all admin-1 in same country, EXCLUDING focal region
country_remote_excl (t-1)   = same for Explosions/Remote violence
country_vac_excl (t-1)      = same for Violence against civilians
country_total_excl (t-1)    = row sum of the three above
```

**Why leave-one-out:** Avoids leaking the focal region's own count. Provides a
national-context signal analogous to a hierarchical model's country-level prior,
in a form tree-based models can exploit directly.

### Layer 8: Engineered Cross-Variable Features

| Feature | Formula | Rationale |
|---|---|---|
| `Battles (t-2)` | shift t-1 by 1 additional month | Enables slope computation |
| `Explosions/Remote violence (t-2)` | same | — |
| `Violence against civilians (t-2)` | same | — |
| `organized_violence (t-1)` | Battles + Explosions + VaC (t-1) | Aggregate threat index |
| `is_active (t-1)` | 1 if organized_violence > 0 | Binary zero-inflation flag |
| `battles_x_remote (t-1)` | Battles(t-1) × Explosions(t-1) | Co-escalation interaction |
| `Battles_3mo_avg (t-1)` | 3-month rolling mean | Trend smoother for sporadic regions |
| `Remote_3mo_avg (t-1)` | — | — |
| `VaC_3mo_avg (t-1)` | — | — |

### Layer 9 (V3 Experiments): Trajectory Features

| Feature | Formula |
|---|---|
| `{target}_slope(t-1)` | target(t-1) − target(t-2) — rate of change |
| `{target}_accel(t-1)` | [target(t-1)−target(t-2)] − [target(t-2)−target(t-3)] — curvature |
| `{target}_nbr_slope(t-1)` | neighbours(t-1) − neighbours(t-2) — neighbour momentum |

Computed for all three targets + three neighbour series = 9 additional columns.

---

## 3. Models

| ID | Algorithm | Objective / Loss | Key Hyperparameters |
|---|---|---|---|
| RF | Random Forest | MSE | n_estimators=100 |
| LGBM-Poisson | LightGBM | Poisson log-likelihood | n_estimators=200, lr=0.05, num_leaves=31 |
| LGBM-Tweedie | LightGBM | Tweedie (p=1.5) | n_estimators=200, lr=0.05, num_leaves=31, α=0.1, λ=1.0 |
| XGBoost | XGBoost | Squared error | n_estimators=200, lr=0.05, max_depth=6 |
| GBR | sklearn GBR | Squared error | n_estimators=100, lr=0.1, max_depth=4 |

**Why Tweedie (p=1.5):** Conflict counts are non-negative, zero-inflated, and
overdispersed (variance >> mean). The Tweedie family at p=1.5 models a compound
Poisson-Gamma process: the zero mass at dormant months and the heavy-tailed counts
in active months are captured within a single objective. Poisson (p=1) assumes
mean=variance, which is violated in every active region. Squared-error objectives
penalise large predictions symmetrically, which biases the model toward the mean.

**Sample weighting:** All models use exponential recency weights:  
`w = exp(−0.05 × months_since_latest)`. An observation 20 months old has weight ≈0.37,
40 months old ≈0.14.

---

## 4. Results

### 4.1 V1 Ablation — 5 Models × 5 Feature Sets

750 evaluations (5 models × 5 feature sets × 3 targets × 10 regions).  
Feature sets: Baseline → +Risk → +Macro → +Holidays → +Engineered.

**Overall MAE (mean across all targets + all 10 regions):**

| Model | Baseline | +Risk | +Macro | +Holidays | +Engineered |
|---|---|---|---|---|---|
| GBR | 60.09 | 58.85 | 58.13 | 58.62 | 61.42 |
| LGBM-Poisson | 53.63 | 53.73 | 53.65 | 53.69 | **53.20** |
| LGBM-Tweedie | 57.48 | 57.67 | 57.27 | 57.36 | 57.73 |
| RF | 57.25 | 57.95 | 56.64 | 57.12 | 57.37 |
| XGBoost | 62.45 | 62.14 | 62.20 | 62.38 | 60.41 |

**V1 best: LGBM-Poisson + Engineered = MAE 53.20** (−7.1% vs RF Baseline of 57.25)

**Per-target breakdown (V1):**

*Battles MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Engineered |
|---|---|---|---|---|---|
| GBR | 58.78 | 55.49 | 53.60 | 56.04 | 57.13 |
| LGBM-Poisson | 52.01 | 52.22 | 52.13 | 52.27 | **51.28** |
| LGBM-Tweedie | 52.51 | 52.99 | 53.34 | 53.56 | **50.52** |
| RF | 56.10 | 56.45 | 55.63 | 56.58 | 54.34 |
| XGBoost | 55.57 | 55.14 | 54.64 | 54.96 | **49.62** |

*Explosions/Remote violence MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Engineered |
|---|---|---|---|---|---|
| GBR | 115.80 | 115.38 | 115.45 | 114.43 | 121.76 |
| LGBM-Poisson | 103.95 | 104.02 | 103.86 | 103.86 | **103.44** |
| LGBM-Tweedie | 115.00 | 115.08 | 113.60 | 113.62 | 117.80 |
| RF | 110.28 | 112.13 | 109.04 | 109.62 | 112.74 |
| XGBoost | 125.91 | 125.52 | 126.20 | 126.48 | 126.24 |

*Violence against civilians MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Engineered |
|---|---|---|---|---|---|
| GBR | 5.69 | 5.67 | 5.35 | 5.41 | 5.37 |
| LGBM-Poisson | 4.94 | 4.94 | 4.95 | 4.95 | **4.89** |
| LGBM-Tweedie | 4.93 | 4.93 | 4.87 | 4.88 | **4.88** |
| RF | 5.36 | 5.28 | 5.25 | 5.16 | 5.03 |
| XGBoost | 5.86 | 5.76 | 5.76 | 5.70 | **5.37** |

**Marginal gain per feature layer (RF, V1):**

| Transition | MAE | Delta |
|---|---|---|
| Baseline | 57.25 | — |
| → +Risk | 57.95 | +0.71 (slightly worse) |
| → +Macro | 56.64 | −1.31 |
| → +Holidays | 57.12 | +0.48 (slightly worse) |
| → +Engineered | 57.37 | +0.25 (slightly worse) |

**Observation:** In V1, feature enrichment provides marginal or no gains for most models.
Only LGBM-Poisson consistently benefits from the +Engineered tier. The lack of the
country-hierarchy layer is the critical gap.

---

### 4.2 V2 Ablation — 5 Models × 6 Feature Sets (+Country tier added)

900 evaluations (5 models × 6 feature sets × 3 targets × 10 regions).  
New tier: +Country (leave-one-out country hierarchy) inserted between +Holidays and +Engineered.

**Overall MAE (mean across all targets + all 10 regions):**

| Model | Baseline | +Risk | +Macro | +Holidays | **+Country** | +Engineered |
|---|---|---|---|---|---|---|
| GBR | 64.25 | 65.17 | 67.00 | 65.90 | 69.47 | 63.73 |
| LGBM-Poisson | 55.64 | 55.84 | 55.78 | 55.85 | 56.83 | 56.71 |
| **LGBM-Tweedie** | 56.84 | 56.94 | 57.62 | 57.59 | **53.05** | 53.60 |
| RF | 58.98 | 59.10 | 60.48 | 59.43 | 60.11 | 60.08 |
| XGBoost | 63.84 | 63.22 | 63.17 | 63.40 | 64.10 | 60.56 |

**V2 best: LGBM-Tweedie + Country = MAE 53.05**

**Marginal gain per feature layer (LGBM-Tweedie, V2):**

| Transition | MAE | Delta | % Change |
|---|---|---|---|
| Baseline | 56.84 | — | — |
| → +Risk | 56.94 | +0.10 | +0.2% |
| → +Macro | 57.62 | +0.68 | +1.2% |
| → +Holidays | 57.59 | −0.03 | −0.0% |
| → **+Country** | **53.05** | **−4.54** | **−7.9%** |
| → +Engineered | 53.60 | +0.56 | +1.1% |

> **The +Country (leave-one-out hierarchy) layer is the single largest improvement: −4.54 MAE (−7.9%).
> No other feature group comes close. This effect is strong for LGBM-Tweedie but absent or negative
> for other models — suggesting the Tweedie objective's handling of zero-inflation interacts
> favourably with the country-level context signal.**

**Per-target breakdown (V2):**

*Battles MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Country | +Engineered |
|---|---|---|---|---|---|---|
| GBR | 65.52 | 67.15 | 66.08 | 65.64 | 71.24 | 56.34 |
| LGBM-Poisson | 48.56 | 48.82 | 48.74 | 48.83 | 49.73 | 50.67 |
| **LGBM-Tweedie** | 49.84 | 50.14 | 50.27 | 50.39 | **47.50** | 50.94 |
| RF | 58.08 | 57.47 | 60.04 | 58.53 | 58.76 | 57.33 |
| XGBoost | 56.04 | 55.22 | 55.00 | 54.98 | 56.86 | **51.11** |

*Explosions/Remote violence MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Country | +Engineered |
|---|---|---|---|---|---|---|
| GBR | 121.62 | 122.96 | 129.56 | 126.96 | 132.28 | 129.36 |
| LGBM-Poisson | 113.33 | 113.65 | 113.61 | 113.71 | 115.83 | 114.54 |
| **LGBM-Tweedie** | 115.87 | 115.87 | 117.75 | 117.57 | 106.69 | **105.05** |
| RF | 113.53 | 114.48 | 116.20 | 114.52 | 116.50 | 117.82 |
| XGBoost | 129.61 | 128.62 | 128.78 | 129.49 | 130.11 | **125.60** |

*Violence against civilians MAE:*
| Model | Baseline | +Risk | +Macro | +Holidays | +Country | +Engineered |
|---|---|---|---|---|---|---|
| GBR | 5.62 | 5.40 | 5.37 | 5.10 | 4.90 | 5.48 |
| LGBM-Poisson | 5.05 | 5.04 | 4.99 | 5.02 | 4.92 | 4.91 |
| **LGBM-Tweedie** | 4.80 | **4.80** | 4.82 | 4.80 | 4.94 | 4.82 |
| RF | 5.34 | 5.35 | 5.19 | 5.24 | 5.06 | 5.08 |
| XGBoost | 5.86 | 5.82 | 5.74 | 5.73 | 5.34 | **4.97** |

**Best model per target (V2):**

| Target | Best model | Best feature set | MAE |
|---|---|---|---|
| Battles | LGBM-Tweedie | +Country | 47.50 |
| Explosions/Remote violence | LGBM-Tweedie | +Engineered | 105.05 |
| Violence against civilians | LGBM-Tweedie | Baseline (+Risk tied) | 4.80 |

> **Scope caveat:** All V2 ablation numbers above are computed on the **10 most
> conflict-active regions globally** (historical top-10 by total event count). These regions
> have abundant, high-magnitude, variable conflict signal — the ideal setting for
> LGBM-Tweedie. Results do not generalise to the full 2,262-region dataset, where
> zero-dominated months cause a complete reversal: persistence outperforms all trained models
> on MAE because predicting "no change" is correct in 83% of global region-months. See §4.8
> for the full-region evaluation and the explanation of why this reversal occurs.

---

### 4.3 Per-Region Performance (LGBM-Tweedie, Best Feature Set per Region)

| Region | Battles MAE | Explos./Remote MAE | VaC MAE | Best FS |
|---|---|---|---|---|
| UKR - Donetsk | 239.70 | 65.37 | 2.65 | +Country / +Macro / Baseline |
| UKR - Kharkiv | 28.28 | 59.76 | 0.51 | +Country / +Country / +Engineered |
| UKR - Sumy | 43.57 | 168.23 | 0.08 | +Engineered / +Country / +Engineered |
| UKR - Kherson | 11.73 | 252.81 | 1.01 | +Engineered / +Engineered / +Country |
| UKR - Zaporizhzhia | 32.85 | 69.92 | 0.52 | +Country / Baseline / +Engineered |
| UKR - Luhansk | 28.95 | 7.40 | 0.20 | +Engineered / +Engineered / +Country |
| SYR - Idlib | 11.71 | 83.55 | 3.63 | +Macro / +Country / Baseline |
| SYR - Aleppo | 11.21 | 64.90 | 17.39 | +Engineered / +Engineered / +Holidays |
| PSX - Gaza | 30.90 | 209.66 | 16.66 | +Engineered / +Engineered / +Engineered |
| BRA - Rio de Janeiro | 19.73 | 0.31 | 3.96 | +Macro / +Macro / Baseline |

**Observation on UKR - Donetsk (Battles MAE = 239.70):** This region is an extreme outlier.
It is the most active conflict zone in the dataset (Ukraine eastern front, peak fighting).
Even the persistence baseline struggles here because the scale of activity shifted dramatically
during the holdout period. All models systematically underestimate peak conflict months.

---

### 4.4 V1 vs V2 Baseline Comparison

The V2 pipeline changes the baseline encoding (cyclic time features) and the dataset
(adds country hierarchy columns). Comparing Baseline-tier MAE between ablations:

| Model | V1 Baseline MAE | V2 Baseline MAE | Change |
|---|---|---|---|
| LGBM-Tweedie | 57.48 | 56.84 | −0.64 |
| RF | 57.25 | 58.98 | +1.73 |
| LGBM-Poisson | 53.63 | 55.64 | +2.01 |
| GBR | 60.09 | 64.25 | +4.16 |
| XGBoost | 62.45 | 63.84 | +1.39 |

LGBM-Tweedie benefits from the cyclic time encoding even at the baseline tier; other models
show minor regression likely due to the change in training data composition.

---

### 4.5 Rosa Branch: Extended Feature Evaluation (LGBM-Tweedie, 80-region broad sample)

Additional feature groups tested on LGBM-Tweedie V2, evaluated on both top-10
and a broader 80-region sample:

**Top-10 regions (test MAE):**

| Feature Set | Train MAE | Test MAE | vs V2 Baseline |
|---|---|---|---|
| V2 Baseline | 9.31 | **57.35** | — |
| + Religion (WRP encoding) | 9.33 | 57.63 | +0.28 |
| + WBD (derived macro features) | 9.17 | 58.64 | +1.29 |
| + Holidays (all 8 religion types) | 9.13 | 57.81 | +0.46 |
| + Holiday interactions | 9.15 | 58.70 | +1.35 |
| + Country (LOO) | 8.92 | 58.89 | +1.54 |
| + Engineered | 8.81 | **56.91** | −0.44 |

**Observation:** On this evaluation run, none of the additional feature groups improved
over the V2 Baseline except +Engineered (−0.44). The WBD derived features, holiday
interactions, and holiday-religion weighted scores all increase MAE slightly. The gap
between train MAE (8–9) and test MAE (57–59) across all configurations suggests
substantial overfitting — models memorise training dynamics but struggle to generalise
to the 6-month holdout.

**Label encoding (top-10 regions, LGBM-Tweedie):**

| Feature Set | Test MAE | vs Baseline |
|---|---|---|
| V2 +Engineered (baseline) | **56.75** | — |
| + Labels (split encoding) | 57.39 | +0.64 |
| + Labels (global encoding) | 57.39 | +0.64 |

**Observation:** Label-encoding religion columns (WRP denomination codes → integers)
does not help LGBM-Tweedie. The model handles string categories natively in LightGBM;
forced integer encoding slightly degrades performance.

---

### 4.6 Improvements Round 6: Structural Architecture Experiments

Six architectural variants of the core pipeline, each benchmarked against persistence:

| Method | Battles | Explosions/Remote | VaC | Overall |
|---|---|---|---|---|
| Persistence (floor) | 45.84 | 88.33 | 3.60 | **45.92** |
| Step 1 — RegressorChain (Tweedie) | 50.98 | 111.53 | 4.53 | 55.68 |
| Step 2 — Encoded features (Tweedie) | 50.42 | 108.21 | 4.67 | 54.43 |
| Step 4 — Religion-weighted holidays (RF) | 56.16 | 114.85 | 5.06 | 58.69 |
| Step 5 — Delta target (LGBM-Poisson) | 50.99 | 108.07 | 4.19 | 54.42 |
| Step 6 — Combined Chain+RF (enc+delta) | 48.83 | 112.90 | 4.70 | 55.48 |

**RegressorChain:** Models the three targets sequentially — Battles first, then
Explosions (conditioning on predicted Battles), then VaC (conditioning on both).
This captures inter-target dependencies: remote violence typically follows battle
escalation. Result: 55.68 — slightly better than LGBM-Poisson alone but still
above persistence.

**Delta target (Step 5):** Models the change in event counts (y(t) − y(t-1)) rather
than the level, using LGBM-Poisson. Result: 54.42 — marginal improvement for VaC
(4.19 vs 4.53), comparable on Battles and Explosions.

**Religion-weighted holidays (Step 4):** Weights each month's holiday count by the
country's share following that religion. Tested on RF. Result: 58.69 — notably worse
than baseline, suggesting RF cannot efficiently exploit this interaction signal.

**Observation:** No Round 6 method beats persistence on the overall metric.
The best approach (Step 2: encoded features) reaches 54.43, still ~18% above persistence (45.92).

---

### 4.7 Improvements Round 7: V3 Experiments

Benchmarked against V2 LGBM-Tweedie (+Country tier) as the reference:

| Method | Battles | Explosions/Remote | VaC | Overall | vs V2 |
|---|---|---|---|---|---|
| **Persistence (floor)** | **45.83** | **88.33** | **3.60** | **45.92** | — |
| V2 LGBM-Tweedie (baseline) | 50.97 | 108.91 | 4.63 | 54.84 | — |
| + Trajectory features (Items 4+9) | 51.54 | 107.86 | 4.38 | 54.59 | −0.25 |
| Log-target training (Item 7) | 49.99 | 140.23 | 4.54 | 64.92 | +10.08 |
| Two-stage model (Item 1) | 57.32 | 151.81 | 5.20 | 71.44 | +16.60 |
| Combined (Items 1+4+7+9) | 192.29 | 221.08 | 7.60 | 140.32 | +85.48 |

**Trajectory features (Items 4+9):** Slope and acceleration of lagged targets +
slope of neighbour counts. Modest improvement (−0.25 overall), driven by
Explosions/Remote violence (108.91 → 107.86). Battles slightly worse (+0.57).
The trajectory signal helps differentiate escalating vs de-escalating situations
but does not dramatically shift the performance level.

**Log-target training (Item 7):** Train on log1p(y), evaluate on original scale.
**Significantly degrades performance: +10.08 overall.** The Explosions/Remote violence
target explodes from 108.91 to 140.23 (+29%). Log-transform compresses the heavy tail
that the Tweedie objective was already designed to handle. The back-transformation
via expm1 amplifies errors in high-count months.

**Two-stage model (Item 1):** Binary onset classifier (P(y>0)) × Tweedie intensity
regressor on active months only. **Significantly degrades: +16.60 overall.** The
classifier stage introduces prediction error that compounds with the regressor's
error. In the top-10 most active regions (evaluation subset), conflict is near-
continuous — the onset classifier rarely helps and occasionally suppresses true
positives.

**Combined (Items 1+4+7+9):** All four V3 improvements together. **Catastrophic
regression: +85.48 overall.** Battles climbs to 192.29 (from 50.97). The combined
model is highly unstable, particularly for the Battles target.

> **Key finding from Round 7:** V3 experiments uniformly fail to improve over the
> V2 baseline. Only the trajectory feature addition shows any positive signal (−0.25),
> and only marginally. The two-stage architecture and log-target transformation are
> harmful in this evaluation setting. The combined model is numerically unstable.
> All trained models remain above the persistence baseline of 45.92.

---

### 4.8 Full-Region Evaluation: LGBM-Tweedie vs Persistence (All 2,262 Valid Regions)

**Why performance reverses when moving from top-10 to the full dataset:**

All V1 and V2 ablation results (§4.1–§4.4) were computed exclusively on the **10 most
conflict-active regions globally** — places like UKR-Donetsk, SYR-Idlib, PSX-Gaza, where
violence events are frequent, large in magnitude, and highly variable month-to-month. In those
regions, LGBM-Tweedie +Country beats the RF Baseline (53.05 vs 57.25, −7.3%) because there is
enough conflict signal for the model to learn meaningful patterns.

The full dataset of 2,262 valid Admin-1 regions is a fundamentally different distribution:
roughly **83% of all region-months have zero change in conflict levels** from one month to the
next. Most of the world, most of the time, has no armed conflict. In this zero-dominated setting:

- **Persistence trivially wins on MAE** by predicting ŷ(t) = y(t−1). When 83% of months have
  no change, predicting "nothing changes" is correct 83% of the time and incurs near-zero error
  across the majority of the dataset.
- **LGBM-Tweedie is structurally penalised.** The Tweedie objective (designed for non-negative
  count data) always outputs a strictly positive prediction. It cannot predict zero or "no
  change." Every quiet region-month where the true value is 0 or unchanged receives a wrong,
  positive prediction — accumulating large MAE across thousands of inactive regions.
- **The model was never designed for this task on quiet regions.** Its strength is estimating
  the *magnitude* of conflict in active zones, not deciding whether conflict is present at all.
  That binary decision (change vs. no change) is exactly what the two-stage CatBoost
  architecture (Rosa branch) was built to solve.

In summary: the top-10 ablation showed what LGBM-Tweedie can do when conflict signal is
abundant. The full-region evaluation exposed what it cannot do when the dataset is dominated by
inactivity. These are two different questions, and both results are correct within their scope.

To establish a fair comparison base with the CatBoost delta-target approach (Rosa branch,
evaluated on all valid regions), LGBM-Tweedie with the best feature set (+Country, V2) was
re-evaluated across all 2,262 valid admin-1 regions. Two directional accuracy variants are
reported: DA (all months) and DA-nonzero (changing months only, matching the CatBoost paper
definition of §6 in the joint paper).

**LGBM-Tweedie (+Country) vs Persistence — all valid regions:**

| Model | Battles MAE | Explos./Remote MAE | VaC MAE | Overall MAE | DA (all months) | DA (changing months) |
|---|---|---|---|---|---|---|
| Persistence | 0.88 | 1.09 | 0.76 | **0.91** | 0.83 | — |
| LGBM-Tweedie +Country | 1.63 | 2.97 | 0.85 | **1.54** | 0.21 | **0.84** |

**Per-target DA (changing months only) — LGBM-Tweedie, 2,262 regions:**

| Target | DA-nonzero |
|---|---|
| Battles | 0.8137 |
| Explosions/Remote violence | 0.8385 |
| Violence against civilians | 0.8510 |
| **Overall** | **0.8373** |

> Note: LGBM-Tweedie is evaluated on regions where the training target is non-zero
> (Tweedie requires a positive label sum). This covers 1,343 / 953 / 2,126 regions
> per target respectively (vs 2,262 for persistence).
>
> **Critical finding on DA:** The previously-reported DA of 0.21 was computed over all
> months including zero-change months, where Tweedie always predicts positive → wrong direction.
> When restricted to months where conflict actually changed (DA-nonzero), LGBM-Tweedie
> achieves 0.84 globally — directly comparable to CatBoost's reported 79–82% DA.
> This makes the two models' directional performance comparable at the global scale.

**Region count note (Conflict D resolution):**
The global LGBM-Tweedie evaluation covers 2,262 regions (regions with at least 12 months of
data and non-zero training target activity). The joint paper's CatBoost global evaluation
covers 2,205 regions (additionally requiring variation in monthly delta targets). The 57-region
difference reflects the different validity criteria appropriate to each model's objective function.

**Historical top-10 high-conflict regions (our ablation benchmark, for reference):**

| Model | Battles MAE | Explos./Remote MAE | VaC MAE | Overall MAE | DA (all) |
|---|---|---|---|---|---|
| Persistence | 45.83 | 88.33 | 3.60 | **45.92** | 0.17 |
| LGBM-Tweedie +Country | 47.75 | 106.55 | 4.85 | **53.05** | 0.53 |

**Interpretation — directional accuracy by evaluation scope:**

The DA metric reveals starkly different model behaviour depending on which regions are evaluated:

| Evaluation scope | Persistence DA | LGBM-Tweedie DA (all) | LGBM-Tweedie DA (nonzero) | Observation |
|---|---|---|---|---|
| All regions (2,262) | **0.83** | 0.21 | **0.84** | Nonzero-DA is comparable to CatBoost's 79–82% |
| Paper's top-10 active | — | 0.49 | **0.65** | Strong directional signal in peak conflict zones |

In active conflict regions, DA-nonzero is 0.65 on the paper's top-10 (see §4.10), vs
CatBoost's 79–82% globally — a meaningful gap driven by the Tweedie objective's inability
to predict "no change" in quiet months.

---

### 4.10 Paper's Top-10 Evaluation: LGBM-Tweedie on Joint Paper's Region Set (Option A)

To enable a direct comparison with §6.4 of the joint paper (`pre_report.pdf`), the full
V2 ablation (LGBM-Tweedie × 6 feature sets) and the RF Baseline were re-run on the paper's
top-10 regions. The paper's top-10 are defined by **total event count during the holdout
period** (last 6 months), unlike our historical top-10 which uses the full date range.

**Paper's top-10 regions (holdout-period definition):**
UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia,
RUS-Kursk, UKR-Chernihiv, PSX-West Bank

**RF Baseline (ACLED-only, analogous to CAST RF baseline):**

| Target | RF Baseline MAE | Paper's CAST RF MAE |
|---|---|---|
| Battles | 64.34 | 63.94 |
| Explosions/Remote violence | 132.56 | 127.93 |
| Violence against civilians | 11.51 | 12.04 |
| **Overall** | **69.47** | **67.97** |

> The RF Baseline MAE (69.47) is within ~2% of the paper's CAST RF (67.97), confirming
> the implementations are structurally equivalent.

**LGBM-Tweedie ablation on paper's top-10 (overall MAE, all targets):**

| Feature Set | MAE | vs RF Baseline |
|---|---|---|
| Baseline | 76.15 | +9.6% |
| +Risk | 76.23 | +9.7% |
| +Macro | 76.86 | +10.6% |
| +Holidays | 76.88 | +10.6% |
| **+Country** | **73.06** | **+5.2%** |
| +Engineered | 73.88 | +6.4% |

**LGBM-Tweedie +Country (best) vs paper's CAST+ (52.29) on same 10 regions:**

| Model | Battles MAE | Explos./Remote MAE | VaC MAE | Overall MAE | DA (all) | DA (nonzero) |
|---|---|---|---|---|---|---|
| RF Baseline | 64.34 | 132.56 | 11.51 | **69.47** | — | — |
| LGBM-Tweedie +Country | 66.68 | 140.21 | 12.29 | **73.06** | 0.49 | **0.65** |
| CAST+ (CatBoost) | 29.62 | 120.41 | 6.84 | **52.29** | — | **0.79–0.82** |

**Per-target winner on paper's top-10:**

| Target | LGBM-Tweedie best | CAST+ MAE | Winner |
|---|---|---|---|
| Battles | 66.68 (+Country) | **29.62** | CAST+ by 55% |
| Explosions/Remote violence | 138.37 (+Engineered) | **120.41** | CAST+ by 13% |
| Violence against civilians | 12.29 (+Country) | **6.84** | CAST+ by 44% |

**Key findings from Option A re-run:**

1. **LGBM-Tweedie does not beat the RF Baseline on the paper's top-10 (+5.2% worse).** On
   our historical top-10 it was −7.3% better, because that set includes Syria and Brazil
   (lower-scale, historically active) where the country-hierarchy feature helps more. On the
   paper's holdout-dominated set (7 of 10 regions are Ukraine frontlines), the extreme
   scale and volatility of the Russia-Ukraine war overwhelms the country-hierarchy signal.

2. **CAST+ substantially outperforms LGBM-Tweedie on these regions** — the two-stage
   CatBoost architecture with delta formulation is better suited to the extreme volatility
   of Ukraine frontline regions.

3. **DA-nonzero (changing months) for LGBM-Tweedie on paper's top-10: 0.65**, compared to
   CatBoost's 0.79–0.82 globally. LGBM-Tweedie has meaningful directional signal but
   CatBoost's two-stage design captures change direction more reliably.

*Source: `outputs/ablation_paper_top10.csv` (computed May 2026, `run_paper_top10_eval.py`).*

---

### 4.9 Rosa Branch: CatBoost Delta-Target Model — Full-Region Comparison

Rosa's CatBoost approach models the *change* in conflict counts (delta targets: Δy = y(t) − y(t−1))
using a two-stage pipeline: Stage 1 classifies whether a change occurs; Stage 2 estimates
the signed magnitude. This directly addresses the directional accuracy weakness of the
absolute-count Tweedie approach.

**CatBoost TwoStage (delta targets) vs Persistence — 100-region sample:**

| Model | delta_battles MAE | delta_remote MAE | delta_vac MAE | DA Battles | DA Remote | DA VaC |
|---|---|---|---|---|---|---|
| Persistence (no change) | 0.93 | 0.91¹ | — | 0.95 | 0.94 | 0.88 |
| CatBoost TwoStage | — | — | — | **0.70** | **0.83** | **0.80** |

¹ Persistence predicts delta = 0; MAE = mean(|actual delta|). DA for persistence
  is high in quiet regions (most deltas are zero) and lower in active regions.

**CatBoost tiered evaluation by conflict intensity (100 regions):**

| Target | Tier | Persistence MAE | CatBoost MAE | Persistence DA | CatBoost DA | MAE improvement |
|---|---|---|---|---|---|---|
| Battles | quiet (<1 mean) | 0.103 | **0.054** | 0.949 | **0.987** | **+47%** |
| Battles | moderate (1–10) | 1.333 | 1.447 | 0.599 | 0.599 | −8% |
| Battles | active (>10) | 0.933 | 1.085 | 0.556 | **0.611** | −16% |
| Remote violence | quiet | 0.074 | **0.058** | 0.938 | **0.951** | **+22%** |
| Remote violence | moderate | 0.553 | 0.577 | 0.693 | **0.713** | −4% |
| Remote violence | active | 0.125 | 0.240 | **0.875** | 0.750 | −92% |
| VaC | quiet | 0.143 | 0.175 | 0.879 | 0.882 | −23% |
| VaC | moderate | 1.267 | 1.376 | 0.400 | **0.475** | −9% |

*Source: `change_target_track.ipynb`, Rosa-Branch, cell 80.*

**Key findings from the CatBoost delta model:**

1. **Quiet regions (dominant in global data):** CatBoost beats persistence on MAE
   (+22–47% improvement) and also improves DA slightly. This is the main benefit.
2. **Moderate and active regions:** Persistence MAE is often lower, but CatBoost DA is
   competitive or better (correctly predicts direction more often even when the magnitude
   is off).
3. **Overall directional accuracy** (CatBoost TwoStage, 100 regions): 0.70 / 0.83 / 0.80
   per target — substantially higher than the LGBM absolute-count model's 0.21 DA
   across all regions, because the two-stage design explicitly models "no change" cases.

**Comparison summary — both approaches, comparable evaluation scope:**

| Approach | Evaluation | Overall MAE | Overall DA | vs Persistence (MAE) |
|---|---|---|---|---|
| Persistence | All regions | 0.91 | 0.83 | reference |
| LGBM-Tweedie +Country (absolute) | All regions | 1.54 | 0.21 | +69% worse |
| Persistence (no change) | Delta, 100 regions | 0.93 | ~0.93 | reference |
| CatBoost TwoStage (delta) | Delta, 100 regions | ~1.84 | ~0.75 | +97% worse |

> Both trained models fail to beat persistence on MAE across the full region set.
> The critical difference is **directional accuracy**: the CatBoost delta approach achieves
> 0.70–0.83 DA vs 0.21 for LGBM-Tweedie, because it explicitly models whether
> conflict increases, decreases, or stays the same rather than always predicting
> a positive absolute count.

---

## 5. Summary of Performance Across All Experiments

> **Paper framing:** LGBM-Tweedie is the alternative model stream in CAST+. All improvements
> are expressed relative to the **CAST RF baseline** (not persistence). MAPE is excluded.
> DA comparisons use **DA-nonzero** (changing months only) to be comparable to CatBoost's 79–82%.

### 5.1 Primary Comparison — Paper's Top-10 (§6.4 of joint paper)

| Model | Overall MAE | DA-nonzero | vs CAST RF Baseline |
|---|---|---|---|
| CAST RF Baseline (ACLED-only) | **69.47** | — | reference |
| LGBM-Tweedie +Country | 73.06 | 0.65 | +5.2% worse |
| **CAST+ (Two-Stage CatBoost)** | **52.29** | **0.79–0.82** | **−24.7%** |

Per-target on paper's top-10:

| Target | CAST RF | LGBM-Tweedie best | CAST+ | LGBM vs RF | CAST+ vs RF |
|---|---|---|---|---|---|
| Battles | 64.34 | 66.68 (+Country) | **29.62** | +3.6% | **−54.0%** |
| Remote violence | 132.56 | 138.37 (+Engineered) | **120.41** | +4.4% | **−9.2%** |
| VaC | 11.51 | 12.29 (+Country) | **6.84** | +6.8% | **−40.6%** |

> CAST+ wins on every target. LGBM-Tweedie does not beat the RF Baseline on the paper's
> holdout-period top-10 — the Ukraine-frontline-dominated set rewards the delta+two-stage
> architecture. LGBM-Tweedie's ablation value lies in the feature analysis, not the MAE.

### 5.2 Internal Ablation Benchmark — Historical Top-10

| Experiment | Model | Feature Set | Overall MAE | vs RF Baseline (57.25) |
|---|---|---|---|---|
| RF Baseline | RF | Baseline | 57.25 | reference |
| V1 best | LGBM-Poisson | +Engineered | 53.20 | −7.1% |
| **V2 best** | **LGBM-Tweedie** | **+Country** | **53.05** | **−7.3%** |
| Round 6 best | LGBM-Tweedie | Encoded features | 54.43 | −4.9% |
| Round 7 best | LGBM-Tweedie | +Traj features | 54.59 | −4.6% |

> On the more diverse historical top-10 (includes Syria, Brazil), LGBM-Tweedie +Country
> beats the RF Baseline by 7.3%. This shows the country-hierarchy feature adds genuine lift
> in heterogeneous, lower-scale conflict settings — the complementary finding for §7.5.

### 5.3 Global Evaluation — All Valid Regions

| Model | Overall MAE | DA (all months) | DA-nonzero | vs Persistence |
|---|---|---|---|---|
| Persistence | **0.91** | 0.83 | — | reference |
| LGBM-Tweedie +Country | 1.54 | 0.21 | **0.84** | +69.6% worse |
| CAST+ (CatBoost) | **1.59** | **0.79–0.82** | — | +74.7% worse |

> Neither trained model beats persistence on global MAE — zero-dominated data means
> predicting "no change" is trivially near-optimal on MAE. The meaningful metric globally
> is DA-nonzero: LGBM-Tweedie (0.84) is directly comparable to CatBoost (0.79–0.82),
> showing both models learn directional patterns equally well on a fair basis.

### 5.4 Key Findings for the Joint Paper

- **CAST+ is the primary model.** LGBM-Tweedie is the alternative stream providing ablation evidence and feature insight.
- **Primary baseline is CAST RF** (69.47 on paper's top-10, 2.344 globally), not persistence.
- **LGBM-Tweedie does not beat the CAST RF on the paper's top-10** (+5.2% worse). Its −7.3% improvement over RF on the historical top-10 reflects the country-hierarchy feature's strength in diverse conflict settings.
- **DA-nonzero reconciles the apparent DA gap**: LGBM-Tweedie (0.84) is competitive with CatBoost (0.79–0.82) when measured on the same basis — changing months only.
- **Complementarity message (§7.5, §9):** CAST+ wins on the primary metric and evaluation. LGBM-Tweedie contributes (1) a cross-model ablation study showing which feature layers matter regardless of architecture, and (2) evidence that country-hierarchy features help absolute-count models in diverse conflict settings but not in the delta-target two-stage framework.

---

*Results sourced from: `outputs/ablation_results.csv`, `outputs/ablation_results_v2.csv`,  
`outputs/improvements_6_summary.csv`, `outputs/improvements_7_results.csv`,  
`outputs/rosa_features_top10.csv`, `outputs/label_encoding_top10.csv`,  
`outputs/full_region_eval.csv` (updated May 2026 with DA-nonzero metric),  
`outputs/ablation_paper_top10.csv` (paper's top-10 re-run, May 2026),  
`change_target_track.ipynb` Rosa-Branch cells 66 & 80 (CatBoost delta results).*  
*Compiled May 2026.*
