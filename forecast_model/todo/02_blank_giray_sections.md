# Task 02 — Fill [GIRAY] Placeholder Sections in the Joint Paper

**Status:** Draft complete — all section prose written in `PAPER_DRAFT.md`; remaining step is pasting into LaTeX source  
**Priority:** HIGH — these sections are literally blank in the pre_report.pdf

---

## Context

The joint paper (`forecast_model/pre_report.pdf`) has red `[GIRAY: ...]` instruction blocks throughout. These sections were left empty for Giray to fill. The paper file itself is a PDF. The actual LaTeX source must be obtained from Rosa or from the Rosa-Branch of the repository. The filled content is now drafted in `PAPER_DRAFT.md` — Rosa can paste each section directly.

---

## Section-by-Section Task List

---

### [GIRAY §3.2] News-Scraped Risk Indicators

**Location in paper:** Section 3.2, page 6

**Status: DRAFTED** — 3 paragraphs in `PAPER_DRAFT.md` §3.2

Content covers:

- GNews API discovery with per-metric keyword templates
- Playwright full-text scraping (75 concurrent tabs, boilerplate removal)
- Fuzzy keyword pre-filter + OpenAI LLM structured classification
- Output: 26,530 records, 236 countries, Jan 2018–Dec 2024, 7 metrics

**Task checklist:**

- [x] Read `src/scraping/` to understand the pipeline
- [x] List all 7 risk indicator column names from master_raw.csv
- [x] Write prose description (done — `PAPER_DRAFT.md` §3.2)
- [x] Add coverage statistics (5.5–21.1% per column — in `PAPER_DRAFT.md` §4.3 table)
- [ ] **Paste into LaTeX source** (requires access to Rosa's .tex file)

---

### [GIRAY §3.6] Data Merging — Dataset Dimensions

**Location in paper:** Section 3.6, page 8

**Status: DRAFTED** — 2-sentence fill-in in `PAPER_DRAFT.md` §3.6

Numbers to fill in:

- Rows: **289,890**
- Regions: **3,221** (total before validity filtering; 2,262 valid for analysis)
- Columns: **78** (final enriched dataset after all feature engineering)

**Task checklist:**

- [x] Confirm row/region/column count — from `model_data_v2_enriched.csv` shape
- [x] Write 2–3 sentences filling in the bracketed values (done — `PAPER_DRAFT.md` §3.6)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §4.3] Layer 3: News-Scraped Risk Indicators

**Location in paper:** Section 4.3, page 9

**Status: DRAFTED** — 1 paragraph in `PAPER_DRAFT.md` §4.3

Content covers:

- All 7 column names with t-1 lag notation
- ISO3 broadcast merge methodology
- t-1 lag leakage rationale
- Sparse coverage note (5–21%); ablation result (no MAE improvement)

**Task checklist:**

- [x] List all 7 risk column names from master_raw.csv
- [x] Write one-line description context for each (names self-descriptive)
- [x] Write construction and lag rationale paragraph (done — `PAPER_DRAFT.md` §4.3)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §4.8] Layer 8: Country-Level Hierarchy Features

**Location in paper:** Section 4.8, page 10

**Status: DRAFTED** — 3 paragraphs in `PAPER_DRAFT.md` §4.8

Content covers:

- 4 LOO feature names and formula
- Why LOO (avoid multicollinearity with Layer 1)
- Why t-1 lag (prevent leakage via current-month country total)
- −4.54 MAE (−7.9%) improvement figure
- Why it helps LGBM-Tweedie (compound Poisson-Gamma + zero-mass discrimination)
- Why CatBoost does not benefit (<1% gain — two-stage onset classifier already models this)

**Task checklist:**

- [x] Draft 2–3 paragraph prose (done — `PAPER_DRAFT.md` §4.8)
- [x] Include the −4.54 MAE improvement figure
- [x] Explain the Tweedie-specific mechanism
- [x] Note why CatBoost doesn't benefit
- [ ] **Paste into LaTeX source**

---

### [GIRAY §4.9] Layer 9: Engineered Cross-Variable Features

**Location in paper:** Section 4.9, page 10

**Status: DRAFTED** — 1 paragraph in `PAPER_DRAFT.md` §4.9

Content covers:

- Layer 8: lag-2 targets, organized_violence, is_active, battles_x_remote, 3-month rolling averages (9 features)
- Layer 9: slope, acceleration, neighbour slope for 3 targets (9 features)
- V2 ablation result: +Engineered adds +0.56 MAE (marginal degradation after +Country)

**Task checklist:**

- [x] Draft prose (~1.5 paragraphs) (done — `PAPER_DRAFT.md` §4.9)
- [x] Note V2 ablation result (+0.56 MAE marginal degradation)
- [x] Note trajectory features result (−0.25 MAE from Round 7)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §5.4] Alternative Model: LGBM-Tweedie with Country Hierarchy

**Location in paper:** Section 5.4, page 13

**Status: DRAFTED** — 3 paragraphs in `PAPER_DRAFT.md` §5.4

Content covers:

- Model: Tweedie p=1.5 motivation (compound Poisson-Gamma, non-negative, zero-inflated, overdispersed)
- Hyperparameters: n_estimators=200, lr=0.05, num_leaves=31, α=0.1, λ=1.0
- Sample weighting: exp(−0.05 × months_since_latest)
- Ablation design: 5 models × 6 feature sets × 10 regions × 3 targets = 900 evaluations (V2)
- Key finding: +Country = −4.54 MAE (−7.9%), largest single improvement
- Both top-10 results: historical (53.05, −7.3% vs RF) and paper's (73.06, +5.2% vs RF)
- Explanation of reversal (Ukraine-dominated holdout set)

**Task checklist:**

- [x] Write model description paragraph (Tweedie motivation) (done)
- [x] Write ablation design paragraph (V1 then V2, what changed, why) (done)
- [x] State key finding: +Country as the breakout improvement (done)
- [x] Mention sample weighting (done)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §6.4] LGBM-Tweedie Ablation Results

**Location in paper:** Section 6.4, page 15

**Status: DRAFTED** — Full section with 4 tables + prose in `PAPER_DRAFT.md` §6.4

Content includes:

- V1 overall MAE table (5 models × 5 feature sets) — historical top-10
- V2 overall MAE table (5 models × 6 feature sets) — historical top-10
- V2 marginal gain table for LGBM-Tweedie
- Paper's top-10 comparison table (RF Baseline 69.47 / LGBM 73.06 / CAST+ 52.29)
- Per-target comparison table (Battles, Remote, VaC)
- Region-count difference explanation (2,262 vs 2,205)
- DA-nonzero explanation and reconciliation

**Task checklist:**

- [x] Format V2 overall MAE table for paper (done)
- [x] Add marginal gain table for LGBM-Tweedie V2 (done)
- [x] Write comparison paragraph vs CAST+ (done)
- [x] Note the region-set difference caveat (done)
- [x] Add DA-nonzero alongside all-months DA with clear labels (done)
- [x] Add region count explanation (2,262 vs 2,205, different validity criteria) (done)
- [x] Format Markdown tables as LaTeX `tabular` (done — `PAPER_DRAFT.md` §6.4 now has 4 `tabular` environments)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §7.5] Per-Target Model Comparison

**Location in paper:** Section 7.5, page 17

**Status: DRAFTED** — 1 paragraph in `PAPER_DRAFT.md` §7.5 (also in `todo/05_still_missing.md` Item 5)

Key answer: CAST+ wins all three targets on the paper's top-10. Gaps: Battles −55.5%, Remote −13.0%, VaC −44.4%. LGBM-Tweedie is not preferable for any target on this evaluation set.

**Task checklist:**

- [x] Extract per-target LGBM best MAE on paper's top-10 (from `outputs/ablation_paper_top10.csv`)
- [x] Compare with CatBoost per-target (29.62, 120.41, 6.84 from paper Table 4)
- [x] Write 2–3 sentence answer (done — `PAPER_DRAFT.md` §7.5)
- [ ] **Paste into LaTeX source**

---

### [GIRAY §9] Conclusion — Complementarity Paragraph

**Location in paper:** Section 9, page 18/19

**Status: DRAFTED** — 1 paragraph in `PAPER_DRAFT.md` §9

Content: LGBM-Tweedie ablation value is the feature-layer map (country hierarchy as the single largest enrichment); CAST+ wins on primary evaluation; both achieve comparable DA-nonzero (~80–84%) on changing months; two streams are structurally complementary.

**Task checklist:**

- [x] Confirm numbers after Task 05 re-run (done — paper's top-10 MAE confirmed)
- [x] Draft paragraph (done — `PAPER_DRAFT.md` §9)
- [ ] **Coordinate exact wording with Rosa; paste into LaTeX**

---

### [GIRAY Appendix B] Feature Reference Table

**Location in paper:** Appendix B, page 20

**Status: DRAFTED** — Complete LaTeX `longtable` in `PAPER_DRAFT.md` Appendix B

Table covers all 78 columns of the enriched dataset plus Rosa's Layer 5b and Layer 7 additions (not in Giray's CSV). Columns: Feature | Layer | Source | Lag | LGBM | CAST+.

**Task checklist:**

- [x] Extract complete feature list from code (78 columns catalogued in `todo/05_still_missing.md` Item 4)
- [x] For each feature: confirm source, lag, and which model uses it
- [x] Format as LaTeX `longtable` (done — `PAPER_DRAFT.md` Appendix B)
- [ ] **Paste LaTeX code into the paper's Appendix B**
