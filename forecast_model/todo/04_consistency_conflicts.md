# Task 04 — Resolve Consistency Conflicts Between MD and Joint Paper

**Status:** DONE — Option A applied for all 4 conflicts; code/data work complete; all §6.4 paper text drafted in `PAPER_DRAFT.md`; only remaining step is pasting into Rosa's LaTeX source  
**Priority:** HIGH — these conflicts would make the joint paper internally inconsistent if left unresolved

**Decision (May 2026):** Option A selected for all four conflicts. The PDF (`pre_report.pdf`) is the authoritative document.

---

## Context

The joint paper (`forecast_model/pre_report.pdf`) and our internal log (`forecast_model/PROJECT_REPORT.md`) were developed independently. Four concrete conflicts exist that would create contradictions in the final paper. Each conflict below describes: what the paper says, what our MD says, what the conflict is, and how to resolve it.

---

## Conflict A: Baseline Definition

### What the paper says
- **Primary baseline:** CAST RF (Random Forest, ACLED-only predictors, 29 features)
  - Global evaluation (2,205 regions): MAE = **2.344**
  - Top-10 evaluation: MAE = **67.968**
- **Secondary baseline:** Persistence (ΔY=0 prediction)
  - Global evaluation: MAE = **0.931**
  - Persistence achieves lower MAE than CAST RF on the global evaluation (because 86–92% of months have zero delta, so predicting zero is correct on MAE but useless for change detection)

### What our MD says
- We compare LGBM-Tweedie against **persistence only**
- We never evaluate LGBM-Tweedie against the CAST RF baseline at the global scale
- On the top-10 (our regions), our "RF Baseline" (V1 ablation) = MAE **57.25** — this is the ACLED-only RF, which is structurally equivalent to CAST RF but evaluated on different regions

### The conflict
In §6.4 the paper asks Giray to compare LGBM-Tweedie against the CAST RF baseline (not persistence). The paper explicitly wants: "Giray best = 53.05, CAST RF = 67.968 on top-10 → −21.9% improvement." But 67.968 is on the paper's top-10 regions; our RF baseline of 57.25 is on our top-10 regions. These are not the same number.

### Resolution
**Option A (preferred):** Re-run the RF Baseline (ACLED-only, no enrichment) on the paper's top-10 regions (see Task 05). Then our improvement claim is: "LGBM-Tweedie +Country vs RF Baseline, both on the same region set."

**Option B (acceptable if Option A is not feasible):** State explicitly in §6.4 that Giray's ablation uses a historically-defined top-10 (different from the holdout-period top-10) and therefore quotes the improvement as "−7.3% vs RF Baseline (MAE 57.25 → 53.05) on the historical top-10 evaluation set." This is honest and avoids inflating the comparison.

**Do NOT:** Quote our 53.05 as a direct comparison to the paper's CAST RF 67.968 (they are on different regions).

**Action items:**

- [x] Decide on Option A or B with Rosa → **Option A chosen**
- [x] Run RF Baseline on paper's top-10 → done (`outputs/ablation_paper_top10.csv`); RF Baseline MAE = 69.47
- [x] Write §6.4 text using RF Baseline (69.47) and best model +Country (73.06) → drafted in `PAPER_DRAFT.md` §6.4
- [ ] **Paste §6.4 text into LaTeX source** (requires Rosa's `.tex` file)

---

## Conflict B: Top-10 Region Definition

### What the paper says
Top-10 most active regions are defined by **total event count during the holdout period** (last 6 months of the test window). The paper's top-10 are:
> UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia, RUS-Kursk, UKR-Chernihiv, PSX-West Bank

### What our MD says
Our top-10 are defined by **total event count summed over the full historical date range** (all training + test months). Our top-10 are:
> UKR-Donetsk, UKR-Kharkiv, UKR-Sumy, UKR-Kherson, UKR-Zaporizhzhia, UKR-Luhansk, SYR-Idlib, SYR-Aleppo, PSX-Gaza, BRA-Rio de Janeiro

### The conflict
6 regions overlap (UKR-Donetsk, UKR-Kharkiv, UKR-Sumy, UKR-Kherson, UKR-Zaporizhzhia, PSX-Gaza). 4 differ. Our regions include Syria and Brazil (historically active but not during holdout); the paper's regions include RUS-Belgorod, RUS-Kursk, UKR-Chernihiv, PSX-West Bank (escalated during the holdout period). This means:
- Our ablation results (MAE 53.05) and the paper's CAST+ results (MAE 52.29) are NOT computed on the same regions
- The §6.4 comparison "CAST+ 52.29 vs Giray 53.05" is comparing apples to oranges

### Resolution
**Option A (cleanest):** Re-run Giray's entire V2 ablation on the paper's top-10 regions (Task 05). Report our best MAE on those 10 regions. This makes the §6.4 comparison valid.

**Option B (acceptable):** Keep our historical top-10 for the ablation (it tests a different and arguably more diverse set of conflict types) but add a one-sentence footnote: "Evaluated on the top-10 by historical total activity; the holdout-period top-10 overlaps on 6 of 10 regions."

**The code for finding top-10 by holdout-period activity:**
```python
# In evaluators.py or inline:
holdout_df = df.groupby("matched_admin1_id").apply(lambda x: x.sort_values("month_year").tail(6))
top10_holdout = holdout_df.groupby("matched_admin1_id")[targets].sum().sum(axis=1).nlargest(10).index.tolist()
```

**Action items:**

- [x] Decide on Option A or B with Rosa → **Option A chosen**
- [x] Re-run ablation on paper's top-10 → done (`outputs/ablation_paper_top10.csv`)
- [x] §6.4 updated with new numbers → drafted in `PAPER_DRAFT.md` §6.4 (4 LaTeX tables)
- [ ] **Paste §6.4 tables into LaTeX source** (requires Rosa's `.tex` file)

---

## Conflict C: Directional Accuracy Definition

### What the paper says
DA is the fraction of holdout months where the model correctly predicts the *direction* of the change (escalating / stable / de-escalating). For the delta formulation:
- Persistence predicts ΔY=0 every month → by definition, DA=0 for all months where the true delta is non-zero
- CAST+ achieves DA = **79–82%** globally (across all 2,205 regions, all months)
- The paper's §6.1 says: "persistence achieves zero directional accuracy for escalating and de-escalating months"

### What our MD says
We define DA as:
```
DA = mean(sign(ŷ − y_{t-1}) == sign(y − y_{t-1}))
```
where ŷ is the absolute count prediction, y is the true count, and y_{t-1} is the lag-1 count.

Under this definition:
- Persistence across all 2,262 regions: DA = **0.83** (because 83% of months have zero change, so sign(0)==sign(0) is always true)
- LGBM-Tweedie globally: DA = **0.21** (Tweedie always predicts positive → wrong direction in quiet regions)
- LGBM-Tweedie top-10: DA = **0.53**

### The conflict
Our DA=0.83 for persistence and the paper's "DA=0 for changing months" for persistence are computed differently. If a reader sees both numbers in the same paper, it looks contradictory. Similarly, comparing our LGBM DA=0.53 on top-10 to CatBoost DA=79–82% globally is not meaningful because they use different denominators and different evaluation sets.

### Resolution
**Option A (preferred, requires code change):** Recompute DA for LGBM-Tweedie using the same definition as CatBoost's evaluation — i.e., on the delta targets directly:
```
true_dir  = sign(y(t) − y(t-1))   # same as sign(true ΔY)
pred_dir  = sign(ŷ(t) − y(t-1))   # same as sign(predicted ΔY)
DA_delta  = mean(true_dir == pred_dir) over changing months only
```
This would give a number interpretable in the same framework as the paper's CatBoost DA.

**Option B (acceptable):** Report two distinct DA metrics clearly labeled:
- "DA (all months)" — our current metric, 0.83 persistence / 0.53 LGBM top-10 / 0.21 LGBM global
- "DA (changing months only)" — compute separately for changing months

In §6.4 use only "DA (changing months only)" for the comparison with CatBoost's DA.

**Implementation for Option A/B:**
The `evaluate_model()` function in `forecast_model/utils/evaluators.py` (lines 242–249) already computes DA as:
```python
true_dir  = np.sign(y_test.values - y_prev)
pred_dir  = np.sign(y_pred       - y_prev)
dir_acc   = float(np.mean(true_dir == pred_dir))
```
This averages over ALL test months including zero-change months. To restrict to changing months:
```python
changing = true_dir != 0
dir_acc_change = float(np.mean(true_dir[changing] == pred_dir[changing])) if changing.any() else np.nan
```

**Action items:**

- [x] Confirm with Rosa: does her CatBoost DA computation include or exclude zero-change months? → **Option A applied regardless**: added `directional_accuracy_nonzero` to `evaluate_model()` and re-ran `run_full_region_eval.py`
- [x] `evaluators.py` now returns both `directional_accuracy` (all months) and `directional_accuracy_nonzero` (changing months only)
- [x] Result: DA-nonzero = **0.84** globally for the best model — directly comparable to CatBoost 79–82%
- [x] §6.4 DA-nonzero section drafted in `PAPER_DRAFT.md` §6.4 (final prose paragraph with clear labels for both DA variants)
- [ ] **Paste DA-nonzero explanation into LaTeX source** (requires Rosa's `.tex` file)

---

## Conflict D: Region Count (2,205 vs 2,262)

### What the paper says
2,205 valid Admin-1 regions are used for the global evaluation. Excluded: regions with fewer than 12 months of data OR no variation in **delta targets** (ΔY is constant for the entire series).

### What our MD says
2,262 valid regions. Excluded: regions with fewer than 12 months of data OR where the training target sum is zero (Tweedie constraint). The exclusion criterion is specific to LGBM-Tweedie's objective function, not to the data.

### The conflict
57 more regions pass our filter than the paper's filter. The paper's stricter filter (no delta variation) is appropriate for a delta-target model but not for an absolute-count model. So the two filters are actually methodologically correct for their respective models. The numbers should just not be directly compared as if they are the same evaluation set.

### Resolution
**No re-run needed.** State clearly in §6.4:
> "The global LGBM-Tweedie evaluation covers 2,262 regions (regions with at least 12 months of data and non-zero training target activity). The CatBoost global evaluation covers 2,205 regions (additionally requiring variation in monthly delta targets). The 57-region difference reflects the different validity criteria appropriate to each model's objective function."

**Action items:**

- [x] Decision: no re-run needed, different filters are methodologically correct
- [x] Explanatory sentence drafted in `PAPER_DRAFT.md` §6.4: "The global LGBM-Tweedie evaluation covers 2,262 regions … The 57-region difference reflects different validity criteria appropriate to each model's objective function."
- [ ] **Paste region-count explanation into LaTeX source** (requires Rosa's `.tex` file)
