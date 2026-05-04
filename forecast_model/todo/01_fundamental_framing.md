# Task 01 — Fundamental Framing Reconciliation

**Status:** DONE — PROJECT_REPORT.md reframed; all paper sections drafted in `PAPER_DRAFT.md`; only remaining step is pasting into Rosa's LaTeX source  
**Priority:** HIGH

---

## Context

The joint paper (`forecast_model/pre_report.pdf`) is titled **"CAST+: Augmenting the ACLED Conflict Alert System"**. It is a unified academic paper with a single narrative:

> CAST+ improves on the CAST baseline through (1) enriched features and (2) a Two-Stage CatBoost architecture (Rosa). Giray's LGBM-Tweedie ablation study is an **alternative model stream** used to triangulate findings and establish which features help.

The PDF is the authoritative document. All framing in PROJECT_REPORT.md has been updated to match it.

---

## Resolved Items

### 1. PROJECT_REPORT.md reframed as alternative model stream

- [x] Header updated — document now identifies itself as Giray's alternative model contribution to the CAST+ paper
- [x] LGBM-Tweedie correctly positioned as the alternative stream; CatBoost is the primary model
- [x] MAPE explicitly excluded from paper-facing sections (note added to header and §1.2)

### 2. CAST RF established as primary baseline

- [x] §1.3 restructured — CAST RF Baseline is now the primary comparison (MAE 69.47 on paper's top-10, 2.344 globally)
- [x] Persistence demoted to secondary baseline with explanation of why it wins on MAE (zero-dominated data) but is useless for change detection
- [x] §5 Summary rewritten — all improvements now expressed vs CAST RF, not persistence

### 3. DA framing aligned with the paper

- [x] DA-nonzero added to `evaluators.py` — DA computed on changing months only
- [x] `run_full_region_eval.py` re-run — DA-nonzero = **0.84** globally for LGBM-Tweedie
- [x] DA-nonzero = **0.65** on paper's top-10
- [x] §1.2 updated — DA-nonzero defined as the paper-comparable metric; DA (all months) retained for completeness only
- [x] Explanation written: the apparent gap (0.21 vs 0.79–0.82) is a definitional artifact, not a real performance gap

### 4. Evaluation scope reconciled

- [x] §1.2 now has three clearly named evaluation tiers:
  - **Paper's top-10** (holdout-period, primary for §6.4) — UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia, RUS-Kursk, UKR-Chernihiv, PSX-West Bank
  - **Historical top-10** (full-range, internal ablation benchmark V1/V2) — different regions, not comparable to Rosa's numbers
  - **Full-region** (2,262 regions, global representativeness)
- [x] Re-run on paper's top-10 complete (`outputs/ablation_paper_top10.csv`)

### 5. Complementarity message established

- [x] §5.4 Key Findings now explicitly states the complementarity: CAST+ wins on primary metric; LGBM-Tweedie contributes (1) cross-model ablation evidence on feature layers and (2) country-hierarchy insight for diverse conflict settings
- [x] Country-hierarchy uniquely helps LGBM-Tweedie but not CatBoost — this cross-model difference is a finding in its own right (noted in §5.4)

---

## Updated Numbers (post Option A re-run)

The original framing claim in this file was:
> LGBM-Tweedie +Country achieves MAE 53.05 on top-10 (vs CAST RF 67.968, −21.9%)

This was based on mismatched region sets. The corrected numbers after Option A re-run:

| Evaluation set | LGBM-Tweedie best | CAST RF | vs CAST RF |
|---|---|---|---|
| Paper's top-10 (holdout-period) | 73.06 (+Country) | 69.47 | **+5.2% worse** |
| Historical top-10 (full-range) | 53.05 (+Country) | 57.25 | **−7.3% better** |

The §6.4 comparison in the paper must use the paper's top-10 numbers (73.06 vs 69.47).
The −7.3% improvement is still a valid finding but scoped to the historical ablation set.

---

## Open Questions (resolved or answered)

- [x] **Which DA definition does Rosa use?** — Applied DA-nonzero regardless (Option A).
  LGBM-Tweedie DA-nonzero = 0.84 globally, comparable to CatBoost 79–82%. Resolved.
- [x] **Does §6.4 want V1 and V2 tables?** — Yes, PDF placeholder says "Present the V1 and
  V2 ablation tables." Both are in PROJECT_REPORT.md §4.1 and §4.2.
- [x] **Should MAPE be dropped?** — Yes. Paper §7.2 explicitly states it is inappropriate.
  Removed from all paper-facing comparisons; still in raw CSV outputs if needed.

---

## Remaining Paper Writing Tasks

All prose is drafted in `PAPER_DRAFT.md`. Remaining step for each is pasting into Rosa's LaTeX source.

- [x] §5.4 — LGBM-Tweedie model description drafted (`PAPER_DRAFT.md` §5.4, 3 paragraphs:
  Tweedie motivation, ablation design V1→V2, key finding +Country −7.9%)
- [x] §6.4 — Ablation tables drafted (`PAPER_DRAFT.md` §6.4: V1 table, V2 table, marginal
  gain table, paper's top-10 comparison — all as LaTeX `tabular`; LGBM does not beat RF on
  paper's top-10 framed explicitly as ablation insight)
- [x] §7.5 — Per-target comparison drafted (`PAPER_DRAFT.md` §7.5: CAST+ wins all 3 targets;
  Battles −55.5%, Remote −13.0%, VaC −44.4%)
- [x] §9 — Concluding paragraph drafted (`PAPER_DRAFT.md` §9: ablation evidence +
  country-hierarchy finding + DA-nonzero parity ~80–84%)
- [ ] **Paste all sections into Rosa's LaTeX source** (requires `.tex` file from Rosa-Branch)
