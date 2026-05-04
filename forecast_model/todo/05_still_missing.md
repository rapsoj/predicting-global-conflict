# Task 05 — New Work Required (Not in MD, Not Adaptable)

**Status:** Mostly complete — Items 1, 3, 5 done; Items 2 and 4 content drafted below (paper text still needs to be written into LaTeX)
**Priority:** HIGH

---

## Item 1: Re-run LGBM-Tweedie on the Paper's Top-10 Regions

**Status: DONE**

- [x] Verify region name strings in the CSV — all 10 confirmed
- [x] Run the ablation script — `run_paper_top10_eval.py` created and executed
- [x] Save `outputs/ablation_paper_top10.csv`
- [x] Extract best LGBM-Tweedie overall MAE and RF Baseline MAE on paper's top-10
- [ ] Use these numbers to update §6.4 in the joint paper (writing task)

**Numbers extracted:**

| Model | Battles MAE | Remote MAE | VaC MAE | Overall MAE |
|---|---|---|---|---|
| RF Baseline | 64.34 | 132.56 | 11.51 | **69.47** |
| LGBM-Tweedie +Country (best) | 66.68 | 150.78 | 12.29 | **73.06** |
| CAST+ (paper §6.3) | 29.62 | 120.41 | 6.84 | **52.29** |

---

## Item 2: News Scraping Pipeline Description

**Status: DONE — prose written into `PAPER_DRAFT.md` §3.2 and §4.3**

**Pipeline summary (from `src/scraping/`):**

The pipeline runs in four sequential stages per country:

1. **Article discovery (GNews API):** For each country × metric × year combination,
   the `GNewsFetcher` class queries the Google News RSS feed via the Python `gnews` library.
   Searches are constructed from per-metric keyword templates (e.g. "crop failure OR harvest
   crisis OR food shortage" for crop failure). Up to `max_results` article URLs are fetched
   per query; duplicate URLs are tracked globally and skipped.

2. **Full-text scraping (Playwright):** The `AsyncPlaywrightBrowser` class opens each URL in
   a headless Chromium browser (up to 75 concurrent tabs across 5 browser instances). Pages
   are scraped after a network-idle wait; text below a minimum length threshold is discarded.
   Boilerplate phrases (nav menus, footers) are removed via repeated-phrase detection.

3. **Fuzzy keyword pre-filter:** Before LLM classification, `AsyncKeywordFilter` checks
   whether the article text contains sufficient keyword overlap with the target metric's
   vocabulary (fuzzy similarity threshold 0.5). Articles failing this filter are discarded,
   reducing LLM API cost.

4. **LLM classification (OpenAI):** `AsyncTextParser` sends each filtered article to an
   OpenAI model with a structured prompt. The prompt includes descriptions of all 7 metrics
   and an output format specifying `{country, metric, date}`. The LLM assigns each article
   to the most relevant metric (or none). Results are flattened to a CSV row per
   (country, metric, month-year) observation.

**Output — `master_raw.csv`:**
- 26,530 records; 4 columns: `country`, `metric`, `date` (MM-YYYY), `source_file` (country name)
- Covers 236 countries; date range January 2018 – December 2024
- The 7 metric values: `contested election`, `crop failure`, `economic concern`,
  `ethnic tension`, `military coup`, `natural disaster`, `political assassination`

**Merge methodology (`utils/risk_merge.py`):**
- Country names from `source_file` are mapped to ISO3 codes via `pycountry` + a manual
  override dictionary (45 non-standard names resolved)
- Records are pivoted to (ISO3, month_year) × metric count matrix — i.e. how many
  articles mentioned each risk for that country-month
- Joined to the Admin-1 panel on (country_code, month_year); country-level counts are
  broadcast to all Admin-1 regions within that country
- All 7 risk columns are then lagged by 1 month (t-1) to prevent data leakage

**Coverage in enriched dataset (non-zero rows out of 289,890 total):**

| Risk column | Non-zero rows | Coverage |
|---|---|---|
| risk_crop_failure (t-1) | 61,024 | 21.1% |
| risk_natural_disaster (t-1) | 55,904 | 19.3% |
| risk_economic_concern (t-1) | 52,851 | 18.2% |
| risk_ethnic_tension (t-1) | 31,621 | 10.9% |
| risk_political_assassination (t-1) | 18,398 | 6.3% |
| risk_contested_election (t-1) | 20,152 | 7.0% |
| risk_military_coup (t-1) | 15,865 | 5.5% |

> Sparse coverage (5–21%) is expected: most country-months have no news mentioning these
> risks. The ablation (§4.10 in PROJECT_REPORT.md / Table 1 in paper §4.10) shows these
> features add noise rather than signal — removing them improves MAE across all targets.

**Task checklist:**
- [x] Read `src/scraping/` and document pipeline stages
- [x] List all 7 risk metric names from `master_raw.csv`
- [x] Document merge methodology from `utils/risk_merge.py`
- [x] Compute coverage statistics
- [x] Write §3.2 prose — 3 paragraphs in `PAPER_DRAFT.md` §3.2
- [x] Write §4.3 feature description — 1 paragraph in `PAPER_DRAFT.md` §4.3
- [ ] **Paste both sections into LaTeX source** (requires Rosa's `.tex` file)

---

## Item 3: Compute DA on Changing Months Only

**Status: DONE**

- [x] Modified `evaluators.py` — `evaluate_model()` now returns both `directional_accuracy`
  (all months) and `directional_accuracy_nonzero` (changing months only)
- [x] Re-ran `run_full_region_eval.py` — updated `outputs/full_region_eval.csv`
- [x] DA-nonzero globally = **0.84** (Battles 0.81, Remote 0.84, VaC 0.85) — directly
  comparable to CatBoost's reported 79–82%
- [ ] Update §6.4 in the paper to report DA-nonzero with a clear label (writing task)

---

## Item 4: Appendix B — Full Feature Reference Table

**Status: DONE — complete LaTeX `longtable` written in `PAPER_DRAFT.md` Appendix B**

All 78 columns of `model_data_v2_enriched.csv` classified by layer, source, lag policy,
and which model uses each feature. CatBoost column reflects Rosa's pipeline (confirmed
from paper §4 and `final_rosa.ipynb`; features not in CatBoost are marked ✗).

| Feature | Layer | Source | Lag | LGBM-Tweedie | CatBoost |
|---|---|---|---|---|---|
| **Layer 1 — Autoregressive Lag Features (ACLED)** | | | | | |
| Battles (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Explosions/Remote violence (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Protests (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Riots (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Strategic developments (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Violence against civilians (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Excessive force against protesters (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Agreement (t-1) | 1 | ACLED | t-1 | ✓ | ✓ |
| Battles_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| Explosions/Remote violence_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| Protests_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| Riots_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| Strategic developments_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| Violence against civilians_neighbours (t-1) | 1 | ACLED spatial | t-1 | ✓ | ✓ |
| **Layer 2 — Temporal Features** | | | | | |
| linear_month_trend | 2 | Derived | none | ✓ | ✓ |
| year | 2 | Derived | none | ✓ | ✓ |
| month_sin | 2 | Derived (cyclic) | none | ✓ | ✓ |
| month_cos | 2 | Derived (cyclic) | none | ✓ | ✓ |
| quarter_sin | 2 | Derived (cyclic) | none | ✓ | ✓ |
| quarter_cos | 2 | Derived (cyclic) | none | ✓ | ✓ |
| **Layer 3 — News-Scraped Risk Indicators (Giray)** | | | | | |
| risk_contested_election (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_crop_failure (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_economic_concern (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_ethnic_tension (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_military_coup (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_natural_disaster (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| risk_political_assassination (t-1) | 3 | GNews/LLM | t-1 | ✓ | ✓ |
| **Layer 4 — World Bank Macroeconomic Features** | | | | | |
| inflation_py | 4 | World Bank (CPI) | prior-year | ✓ | ✓ |
| youth_unemployment_py | 4 | World Bank (SL.UEM.NEET.ZS) | prior-year | ✓ | ✓ |
| income_inequality_py | 4 | World Bank (SI.POV.GINI) | prior-year | ✓ | ✓ |
| income_level_code | 4 | World Bank metadata | none (structural) | ✓ | ✓ |
| **Layer 5 — Engineered World Bank Features (Rosa)** | | | | | |
| structural_vulnerability | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| inflation_shock | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| shock_vulnerability | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| capacity_adjusted_risk | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| inflation_change | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| high_inequality_flag | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| inflation_shock_poor | 5 | Derived (WB) | prior-year | ✗ | ✓ |
| conflict_momentum | 5 | Derived (ACLED) | t-1 | ✗ | ✓ |
| violence_escalation | 5 | Derived (ACLED) | t-1 | ✗ | ✓ |
| macro_conflict_pressure | 5 | Derived (ACLED+WB) | t-1 | ✗ | ✓ |
| **Layer 6 — Holiday Features** | | | | | |
| holiday_count_month | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| is_holiday_month | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| christian_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| islam_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| shia_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| hindu_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| buddhist_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| jewish_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| cultural_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| nonreligious_holiday_count | 6 | Public holiday calendar | none (current month) | ✓ | ✓ |
| **Layer 7 — Holiday × Religion Interaction Features (Rosa)** | | | | | |
| weighted_holidays_majority | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| weighted_holidays_minority1 | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| weighted_holidays_minority2 | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| minority_tension_minority1 | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| minority_tension_minority2 | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| minority_tension_total | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| total_religious_mobilization | 7 | Derived (holiday × WRP) | t-1 | ✗ | ✓ |
| **Layer 5 (Religion) — Religious Composition (WRP 2010)** | | | | | |
| majority_religion | 5 | World Religion Project | none (static) | ✓ | ✓ |
| majority_pct | 5 | World Religion Project | none (static) | ✓ | ✓ |
| minority1_religion | 5 | World Religion Project | none (static) | ✓ | ✓ |
| minority1_pct | 5 | World Religion Project | none (static) | ✓ | ✓ |
| minority2_religion | 5 | World Religion Project | none (static) | ✓ | ✓ |
| minority2_pct | 5 | World Religion Project | none (static) | ✓ | ✓ |
| nonreligpct | 5 | World Religion Project | none (static) | ✓ | ✓ |
| **Layer 8 — Country-Level Hierarchy Features (Giray)** | | | | | |
| country_battles_excl (t-1) | 8 | Derived (ACLED LOO) | t-1 | ✓ | ✗ |
| country_remote_excl (t-1) | 8 | Derived (ACLED LOO) | t-1 | ✓ | ✗ |
| country_vac_excl (t-1) | 8 | Derived (ACLED LOO) | t-1 | ✓ | ✗ |
| country_total_excl (t-1) | 8 | Derived (ACLED LOO) | t-1 | ✓ | ✗ |
| **Layer 9 — Engineered Cross-Variable Features (Giray)** | | | | | |
| Battles (t-2) | 9 | ACLED | t-2 | ✓ | ✗ |
| Explosions/Remote violence (t-2) | 9 | ACLED | t-2 | ✓ | ✗ |
| Violence against civilians (t-2) | 9 | ACLED | t-2 | ✓ | ✗ |
| organized_violence (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |
| is_active (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |
| battles_x_remote (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |
| Battles_3mo_avg (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |
| Remote_3mo_avg (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |
| VaC_3mo_avg (t-1) | 9 | Derived (ACLED) | t-1 | ✓ | ✗ |

> Note: Layer 5 (WBD engineered) and Layer 7 (holiday interactions) are Rosa's additions;
> they are present in `final_rosa.ipynb` but not in `model_data_v2_enriched.csv`.
> Layer 8 and 9 are Giray's additions; CatBoost does not use them (country hierarchy
> provided <1% gain in the two-stage framework per §7.1 of the paper).

**Task checklist:**
- [x] Run `df.columns.tolist()` on enriched CSV — 78 columns catalogued
- [x] Classify each column into layer, source, lag policy
- [x] Determine which model uses each feature
- [x] Format as LaTeX `longtable` — done in `PAPER_DRAFT.md` Appendix B
- [ ] **Paste LaTeX `longtable` into paper's Appendix B** (requires Rosa's `.tex` file)

---

## Item 5: §7.5 Per-Target Model Comparison on Shared Regions

**Status: DONE — content ready for §7.5**

Per-target MAE on the paper's top-10 regions (both models now on the same region set):

| Target | LGBM-Tweedie best | Best FS | CAST+ MAE | Winner | Gap |
|---|---|---|---|---|---|
| Battles | 66.68 | +Country | **29.62** | CAST+ | −55.5% |
| Remote violence | 138.37 | +Engineered | **120.41** | CAST+ | −13.0% |
| Violence against civilians | 12.29 | +Country | **6.84** | CAST+ | −44.4% |
| **Overall** | **73.06** | +Country | **52.29** | **CAST+** | **−28.4%** |

**§7.5 answer (ready to write into paper):**

On the paper's top-10 most active regions, CAST+ outperforms LGBM-Tweedie on all three
targets. The gap is largest for Battles (−55.5%) and Violence against civilians (−44.4%),
where the two-stage architecture's ability to first predict whether change occurs — rather
than always estimating a positive count — provides the greatest advantage. Remote violence
shows the smallest gap (−13.0%), suggesting that for extreme-scale explosive events (the
Ukraine eastern front dominates this target), both approaches struggle with the same
fundamental forecasting challenge: the scale and volatility of peak war zones is unlike
anything in the training distribution, so neither model substantially outperforms the
other relative to that baseline uncertainty.

LGBM-Tweedie is preferable for neither Remote violence nor VAC on the paper's top-10.
CAST+ is the recommended model for all three targets in high-activity conflict zones.
The complementarity noted in §7.5 is architectural, not regional: LGBM-Tweedie's
country-hierarchy feature adds lift in lower-scale, more diverse conflict regions (see
§4.8 full-region evaluation), while CAST+'s two-stage design dominates in the zero-inflated,
high-volatility setting that characterises the global evaluation set.

**Task checklist:**
- [x] Complete Item 1 first — done
- [x] Extract per-target LGBM best from `outputs/ablation_paper_top10.csv`
- [x] Compare with Rosa's CatBoost per-target (29.62, 120.41, 6.84 from paper Table 4)
- [x] Write the §7.5 comparison text — content above
- [ ] Copy §7.5 text into the paper (writing task)
