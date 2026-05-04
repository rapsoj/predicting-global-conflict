# CAST+ Report Gap Tracking

**Joint paper:** "CAST+: Augmenting the ACLED Conflict Alert System"  
**Authors:** Rosa Daneshmandnia, Giray Ünlü  
**Working directory:** `d:\gconflict\predicting-global-conflict2\forecast_model\`

## Context

The pre-report (`forecast_model/pre_report.pdf`) is a 20-page joint paper draft.
Giray's sections are marked with red `[GIRAY: ...]` placeholders throughout.
Our internal technical log (`forecast_model/PROJECT_REPORT.md`) contains most of the
underlying numbers but is not structured or framed to fill those placeholders directly.

This folder tracks the five categories of work needed to close the gap.

---

## Files in This Folder

| File | Category | Priority | Status |
|---|---|---|---|
| [01_fundamental_framing.md](01_fundamental_framing.md) | Reframe our narrative to fit joint paper | HIGH | Open |
| [02_blank_giray_sections.md](02_blank_giray_sections.md) | Fill [GIRAY] placeholders | HIGH | Open |
| [03_md_content_to_adapt.md](03_md_content_to_adapt.md) | Adapt existing MD content | MEDIUM | Open |
| [04_consistency_conflicts.md](04_consistency_conflicts.md) | Resolve contradictions | HIGH | Open |
| [05_still_missing.md](05_still_missing.md) | New work / new runs needed | HIGH | Open |

---

## Key Files to Know

| File | Purpose |
|---|---|
| `forecast_model/pre_report.pdf` | The joint paper draft (20 pages). Read this first. |
| `forecast_model/PROJECT_REPORT.md` | Our internal technical log. Contains numbers but wrong framing. |
| `forecast_model/outputs/full_region_eval.csv` | Full-region LGBM-Tweedie + Persistence results (2,262 regions). |
| `forecast_model/outputs/ablation_results_v2.csv` | V2 ablation tables (5 models × 6 feature sets × 10 regions). |
| `forecast_model/outputs/ablation_results.csv` | V1 ablation tables. |
| `forecast_model/utils/evaluators.py` | evaluate_model() — core evaluation function. |
| `forecast_model/run_full_region_eval.py` | Script that produced full_region_eval.csv. |

---

## Global Numbers to Keep in Mind

### From the joint paper (pre_report.pdf)
- CAST RF baseline (ACLED-only): MAE **2.344** globally (2,205 regions), **67.968** top-10
- Persistence baseline: MAE **0.931** globally
- CAST+ (Two-Stage CatBoost): MAE **1.589** globally (+24.6% over CAST RF), **52.290** top-10 (+23.1%)
- CAST+ Directional Accuracy: **79–82%** globally across three targets

### From our PROJECT_REPORT.md
- LGBM-Tweedie +Country best (top-10, our regions): MAE **53.05**, DA **0.53**
- LGBM-Tweedie +Country (all 2,262 regions): MAE **1.54**, DA **0.21**
- Persistence (all 2,262 regions): MAE **0.91**, DA **0.83**
- Persistence (top-10, our regions): MAE **45.92**, DA **0.17**

---

## Update Instructions

When a task in any file is completed, mark it `[x]` and update the Status column above.
Each task file is standalone — it can be opened in a fresh chat without reading this index.
