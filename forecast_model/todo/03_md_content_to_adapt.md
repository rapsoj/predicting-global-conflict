# Task 03 — Adapt Existing MD Content for the Joint Paper

**Status:** DONE — all content mapped and drafted; every section in `PAPER_DRAFT.md`  
**Priority:** MEDIUM — this is copy-and-reformat work, not new content

---

## Context

`forecast_model/PROJECT_REPORT.md` contains technical content adaptable into the joint paper. For the joint paper, this content needed to be:

1. Reframed to position LGBM-Tweedie as the *alternative model*, not the primary
2. Shortened — the MD is exhaustive; paper sections are 0.5–1 page each
3. Reconciled with CatBoost's numbers (see Task 04 for conflicts — all resolved)

All paper-ready prose is now in `PAPER_DRAFT.md`. The mapping below records where each MD section ended up.

---

## Content Map: MD → Paper → PAPER_DRAFT.md

| MD section | Paper section | Status | PAPER_DRAFT.md location |
| --- | --- | --- | --- |
| §2 Layer 7 (Country hierarchy) | §4.8 | Done | §4.8 |
| §2 Layers 8+9 (Engineered/Traj) | §4.9 | Done | §4.9 |
| §3 Models table | §5.4 | Done | §5.4 |
| §4.1–4.2 Ablation tables | §6.4 | Done | §6.4 |
| §4.3 Per-region table | §6.4 (omitted — too granular) | Done | — |
| §4.4 V1 vs V2 | §5.4 (1 sentence) | Done | §5.4 para 3 |
| §4.6–4.7 Rounds 6+7 | §7.2 (contribution to Rosa's section) | Done | §7.2 |
| §4.8 Full-region LGBM | §6.4 footnote / DA reconciliation | Done | §6.4 prose |
| §4.10 Paper's top-10 re-run | §6.4 primary comparison table | Done | §6.4 |

---

## Adaptation Notes per Section

### MD §2 (Layers 7–9) → Paper §4.8 and §4.9

- Layer 7 prose trimmed from ~200 words to ~100 words in `PAPER_DRAFT.md` §4.8
- "This feature benefits LGBM-Tweedie specifically but not CatBoost" sentence added
- Layers 8+9 merged into one paragraph in `PAPER_DRAFT.md` §4.9
- "Why leave-one-out" technical detail retained but condensed

### MD §3 (Models table) → Paper §5.4

- Model table converted to prose in `PAPER_DRAFT.md` §5.4
- Focus on LGBM-Tweedie; other 4 models mentioned only to explain ablation design
- "Why Tweedie" text used almost verbatim — paper-appropriate
- Sample weighting formula included

### MD §4.1 + §4.2 (V1 and V2 Ablation Tables) → Paper §6.4

- V1 overall MAE table included (5 models × 5 feature sets)
- V2 overall MAE table included (5 models × 6 feature sets)
- V2 marginal gain table for LGBM-Tweedie included
- Per-target breakdowns summarised in prose (not separate tables — too granular)
- Paper's top-10 comparison table added (RF 69.47 / LGBM 73.06 / CAST+ 52.29)
- Region-set mismatch caveat added (historical top-10 vs holdout-period top-10)

### MD §4.3 (Per-Region Table) → Omitted

Too granular for the paper. UKR-Donetsk outlier (Battles MAE 239.70) mentioned parenthetically in §6.4 prose as needed.

### MD §4.4 (V1 vs V2 Baseline) → Paper §5.4

Reduced to one sentence in `PAPER_DRAFT.md` §5.4: cyclic time encoding gives LGBM-Tweedie −0.64 MAE improvement at baseline tier; other models show minor regression.

### MD §4.6–4.7 (Rounds 6 and 7) → Paper §7.2

**Status: DONE** — 1 paragraph in `PAPER_DRAFT.md` §7.2. Contribution goes into Rosa's §7.2 "What Did Not Work" block; coordinate insertion point with Rosa.

Content drafted:

- Two-stage LGBM (Round 7): +16.60 MAE — classifier fires correctly only in the full global distribution, not in always-active evaluation regions; explains why CatBoost's two-stage success is context-dependent
- Log-target training: +10.08 MAE — Tweedie already handles the heavy tail structurally; log-transform plus exp back-transformation amplifies errors in high-count months
- Delta target LGBM-Poisson (Round 6 Step 5): MAE 54.42, comparable to absolute-count approach but without CatBoost's directional-accuracy gains — confirms the onset classifier is the critical missing component

### MD §4.8 (Full-Region Evaluation) → Paper §6.4 footnote / DA reconciliation

Incorporated into `PAPER_DRAFT.md` §6.4 prose:

- 2,262 vs 2,205 region count explained (different validity criteria — both correct)
- DA-nonzero reconciliation: LGBM 0.21 all-months vs 0.84 changing-months-only
- DA-nonzero 0.84 directly comparable to CatBoost 79–82%
- Structural reason for reversal (Tweedie always predicts positive → wrong in quiet months)
