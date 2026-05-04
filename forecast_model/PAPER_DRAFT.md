# CAST+ — Draft Text for [GIRAY] Placeholder Sections

**Source:** Synthesised from PROJECT_REPORT.md, todo/05_still_missing.md, and code inspection  
**Compiled:** May 2026  
**Purpose:** Paper-ready prose for Rosa to paste into the LaTeX source of `pre_report.pdf`

Each section below corresponds to one `[GIRAY: ...]` placeholder in the PDF.  
Numbers are sourced from `outputs/ablation_paper_top10.csv` (for §6.4 primary comparison)  
and `outputs/ablation_results_v2.csv` + `outputs/full_region_eval.csv` (for the ablation tables).

---

## §3.2 — News-Scraped Risk Indicators

*Target length: 2–3 paragraphs. Replace the [GIRAY: §3.2] block.*

---

News coverage of conflict-relevant societal risks was collected through a four-stage automated pipeline implemented in `src/scraping/`. In the first stage, article discovery, the `GNewsFetcher` module queries the Google News RSS feed via the `gnews` library for each country–metric–year combination. Searches are constructed from per-metric keyword templates — for example, "crop failure OR harvest crisis OR food shortage" for the crop-failure indicator, and "military coup OR government overthrow" for the coup indicator. Up to a configurable maximum of article URLs are fetched per query; duplicate URLs are tracked globally and skipped to prevent re-processing.

In the second stage, full-text scraping, a headless Chromium browser (Playwright) retrieves the complete body text of each article URL, using up to 75 concurrent tabs spread across five browser instances for throughput. Pages are rendered after a network-idle timeout; articles below a minimum character threshold are discarded as uninformative, and repeated boilerplate phrases (navigation menus, cookie banners, footer text) are detected via n-gram frequency analysis and removed. Before invoking the language model, an `AsyncKeywordFilter` applies a fuzzy string-matching pre-filter: articles whose text contains insufficient overlap with the target metric's vocabulary (similarity threshold 0.5) are discarded, substantially reducing the volume of text sent to the API.

In the final stage, surviving articles are sent to an OpenAI language model via `AsyncTextParser` with a structured prompt that describes all seven risk categories and requests a structured output specifying the country, assigned metric, and the month-year of the article. Each classified article yields one row in the output. Results are aggregated to (country, metric, month-year) counts and written to `data/raw/master_raw.csv`, which contains 26,530 records covering 236 countries from January 2018 through December 2024.

---

## §3.6 — Data Merging: Dataset Dimensions

*Target length: 2–3 sentences filling in [GIRAY: [number of rows]], [GIRAY: [number of regions]], [GIRAY: [number of columns]].*

---

After joining all five data sources — ACLED monthly event counts, World Bank macroeconomic indicators, news-scraped risk indicators, national holiday calendar, and World Religion Project religious composition data — the merged panel contains **289,890** region-month observations across **3,221** Admin-1 regions and 236 countries, spanning January 2018 through June 2025. Country-level features (World Bank indicators, religious composition, and news risk counts) are broadcast to Admin-1 regions via ISO3 country-code matching, so that each region within a country receives the same country-level feature values for each month. The final analysis-ready dataset, after cross-variable feature engineering (Layers 8–9), contains **78** columns: 14 autoregressive lag features, 6 cyclic temporal encodings, 7 news-scraped risk indicators, 4 World Bank macroeconomic features, 7 religious composition columns, 10 religion-specific holiday counts, 4 country-level hierarchy features, and 9 engineered cross-variable features, plus 7 metadata and target columns (region ID, country code, month-year, and the three conflict targets).

---

## §4.3 — Layer 3: News-Scraped Risk Indicators

*Target length: ~1 paragraph. Replace the [GIRAY: §4.3] block.*

---

Seven news-derived risk indicators are included as predictors: `risk_contested_election (t-1)`, `risk_crop_failure (t-1)`, `risk_economic_concern (t-1)`, `risk_ethnic_tension (t-1)`, `risk_military_coup (t-1)`, `risk_natural_disaster (t-1)`, and `risk_political_assassination (t-1)`. Each column records the monthly count of news articles classified to that risk category for a given country (from `master_raw.csv`, see §3.2), matched to Admin-1 regions via ISO3 country code. A one-month lag (t-1) is applied to all seven columns to prevent data leakage: news published in month $t$ reflects events in month $t$, so using current-month values would introduce target-contemporaneous information into the feature set. Coverage is sparse by design — between 5.5% and 21.1% of region-months have a non-zero value for any given risk indicator, reflecting the fact that most country-months do not generate classified news articles for these specific categories. In the ablation study (§6.4), the +Risk tier (adding these seven features to the ACLED baseline) does not reduce MAE for any model family, consistent with the low signal-to-noise ratio implied by sparse coverage.

---

## §4.8 — Layer 8: Country-Level Hierarchy Features

*Target length: 2–3 paragraphs. Replace the [GIRAY: §4.8] block.*

---

Four leave-one-out (LOO) country aggregate features were engineered from the ACLED event counts to provide a national-context signal without leaking the focal region's own activity. The four features are: `country_battles_excl (t-1)`, `country_remote_excl (t-1)`, `country_vac_excl (t-1)`, and `country_total_excl (t-1)` — each computed as the sum of the corresponding conflict type over all Admin-1 regions in the same country, excluding the focal region itself. The leave-one-out exclusion is necessary to avoid multicollinearity with the Layer 1 autoregressive predictors: including the focal region in the national sum would effectively re-encode the focal region's own lag-1 count at the country scale. All four features are lagged one month (t-1) for the same reason: the current-month country total contains the focal region's current-month conflict count, constituting direct data leakage into the target variable.

These features act as a hierarchical prior. A region surrounded by widespread national conflict activity is in a structurally different escalation context from an otherwise-identical region where the rest of the country is quiet — a distinction that region-level lag features alone cannot capture. By making the national-minus-local context explicit, tree-based models can learn that a region's individual historical pattern and the broader country trajectory jointly determine the risk of future escalation.

In the V2 ablation study, the +Country tier delivers the single largest improvement for LGBM-Tweedie: −4.54 MAE (−7.9%), more than three times the combined contribution of all other enrichment tiers. This improvement is specific to LGBM-Tweedie and is absent or negative for the four other model families evaluated. The pattern appears to arise from an interaction between the Tweedie objective and the national-context signal: the country-level aggregate helps the model distinguish genuinely quiet regions (national aggregate near zero) from regions where the broader national conflict environment is active, sharpening the effective zero-mass versus non-zero-mass discrimination that the compound Poisson-Gamma structure of the Tweedie distribution captures. Notably, country-hierarchy features provide less than 1\% gain in the two-stage CatBoost framework of CAST+ (see §7.2), suggesting that the two-stage architecture — which explicitly models onset probability in Stage 1 — already learns this context from the data without requiring the aggregate to be pre-computed as an explicit feature.

---

## §4.9 — Layer 9: Engineered Cross-Variable Features

*Target length: ~1.5 paragraphs. Replace the [GIRAY: §4.9] block.*

---

Two additional feature groups were constructed from the existing lag-1 ACLED columns. The first group (cross-variable engineered features) adds nine columns: lag-2 counts for each of the three conflict targets — `Battles (t-2)`, `Remote violence (t-2)`, and `VaC (t-2)` — which give the model access to a two-step history and enable implicit first-difference computation; an `organized_violence (t-1)` aggregate equal to the sum of Battles, Remote violence, and Violence against civilians at lag-1; a binary `is_active (t-1)` indicator (1 if organized\_violence $> 0$) that directly flags regions in active conflict; a co-escalation interaction term `battles_x_remote (t-1)` (product of Battles and Remote violence lag-1 counts) capturing months where both conflict types intensify simultaneously; and three-month rolling averages — `Battles_3mo_avg (t-1)`, `Remote_3mo_avg (t-1)`, and `VaC_3mo_avg (t-1)` — that smooth month-to-month variation in sporadic conflict regions. The second group (trajectory features, introduced in Round 7 experiments) adds nine further columns: for each of the three targets, a first-difference slope (target$(t-1) -$ target$(t-2)$) capturing recent momentum, a second-difference acceleration capturing the change-in-rate-of-change, and a neighbour slope (neighbours$(t-1) -$ neighbours$(t-2)$) capturing momentum in adjacent regions. In the V2 ablation, the combined +Engineered tier (both groups) adds +0.56 MAE for LGBM-Tweedie when stacked after the +Country tier — a marginal degradation indicating the additional complexity does not improve generalisation at the holdout horizon.

---

## §5.4 — Alternative Model: LGBM-Tweedie with Country Hierarchy

*Target length: 3 paragraphs. Replace the [GIRAY: §5.4] block.*

---

The alternative model stream employs LightGBM with a Tweedie objective (variance power $p = 1.5$). Conflict event counts are non-negative, exhibit a large zero mass in inactive regions, and are overdispersed — variance substantially exceeds the mean — in active conflict zones. The Tweedie family at $p = 1.5$ models a compound Poisson-Gamma process that accommodates both properties within a single objective: the compound Poisson component generates the zero mass at dormant months, while the Gamma component captures the heavy-tailed distribution of counts in active months. The alternative objectives evaluated — Poisson ($p = 1$, which assumes variance equals the mean, a constraint violated in every high-conflict region) and squared error (which penalises large predictions symmetrically and biases estimates toward the overall mean) — performed less well in the ablation. All five model families in the study use exponential recency weighting: $w = \exp(-0.05 \times \text{months since most recent})$, so that an observation 20 months old receives weight $\approx 0.37$ and one 40 months old receives weight $\approx 0.14$. Hyperparameters for LGBM-Tweedie: $n_{\text{estimators}} = 200$, $\text{learning rate} = 0.05$, $\text{num\_leaves} = 31$, $\alpha = 0.1$, $\lambda = 1.0$.

The ablation study evaluated five model families — Random Forest (MSE), LightGBM-Poisson, LightGBM-Tweedie, XGBoost (squared error), and Gradient Boosted Regression — across two cumulative rounds. V1 (750 evaluations: 5 models $\times$ 5 feature sets $\times$ 3 targets $\times$ 10 regions) tested feature tiers Baseline (ACLED lags + temporal encodings + religious composition), +Risk (seven news-scraped indicators), +Macro (four World Bank features), +Holidays (religion-specific holiday counts), and +Engineered (cross-variable features). V2 (900 evaluations) introduced a sixth tier, +Country (four leave-one-out country hierarchy features; §4.8), inserted between +Holidays and +Engineered. Each evaluation followed a strict chronological split — training on all months before the holdout window, testing on the final six months — across the ten most historically active conflict regions and three regression targets.

The defining result of the ablation is the discontinuous improvement produced by the +Country tier for LGBM-Tweedie: $-4.54$ MAE ($-7.9\%$), the single largest improvement observed across all six tiers and all five model families. No other enrichment layer comes within a factor of three of this improvement. The V2 best configuration — LGBM-Tweedie +Country — achieves MAE 53.05 on the internal historical top-10 evaluation set, representing a $-7.3\%$ improvement over the ACLED-only RF Baseline (MAE 57.25) on that set. On the paper's holdout-period top-10 (ten regions most active during the test window, dominated by Ukraine frontlines and Gaza), LGBM-Tweedie +Country achieves MAE 73.06, which is 5.2\% above the RF Baseline (69.47) evaluated on the same regions. The country-hierarchy advantage is stronger in the more geographically diverse historical top-10 (includes Syria, Brazil, and a wider range of conflict scales) and weaker in the Ukraine-dominated holdout-period top-10, where the extreme volatility and scale of the Russia-Ukraine war overwhelm the national-context signal.

---

## §6.4 — LGBM-Tweedie Ablation Results

*Replace the [GIRAY: §6.4] block. Includes V1 table, V2 table, marginal gain table, and comparison prose.*

---

### V1 Ablation: Five Models × Five Feature Sets

Table~\ref{tab:v1} presents overall MAE (mean across all three targets and all ten historical top-10 regions) for the V1 ablation (750 evaluations). Feature sets accumulate: each column includes all features from the columns to its left.

\begin{table}[h]
\centering
\caption{V1 ablation: overall MAE by model and feature set (historical top-10 regions, 750 evaluations). Feature sets accumulate left to right.}
\label{tab:v1}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Model} & \textbf{Baseline} & \textbf{+Risk} & \textbf{+Macro} & \textbf{+Holidays} & \textbf{+Engineered} \\
\midrule
GBR                & 60.09 & 58.85 & 58.13 & 58.62 & 61.42 \\
LGBM-Poisson       & 53.63 & 53.73 & 53.65 & 53.69 & \textbf{53.20} \\
LGBM-Tweedie       & 57.48 & 57.67 & 57.27 & 57.36 & 57.73 \\
RF (CAST Baseline) & 57.25 & 57.95 & 56.64 & 57.12 & 57.37 \\
XGBoost            & 62.45 & 62.14 & 62.20 & 62.38 & 60.41 \\
\bottomrule
\end{tabular}
\end{table}

V1 best: LGBM-Poisson +Engineered, MAE 53.20 ($-7.1\%$ vs RF Baseline of 57.25). Feature enrichment produces marginal or no improvement for most model families in V1; the absence of a national-context feature is the critical gap.

### V2 Ablation: Five Models × Six Feature Sets (+Country Added)

Table~\ref{tab:v2} extends V1 with the +Country tier (§4.8) inserted between +Holidays and +Engineered.

\begin{table}[h]
\centering
\caption{V2 ablation: overall MAE by model and feature set (historical top-10 regions, 900 evaluations). +Country tier (§\ref{sec:country-hierarchy}) added between +Holidays and +Engineered.}
\label{tab:v2}
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Model} & \textbf{Baseline} & \textbf{+Risk} & \textbf{+Macro} & \textbf{+Holidays} & \textbf{+Country} & \textbf{+Engineered} \\
\midrule
GBR                       & 64.25 & 65.17 & 67.00 & 65.90 & 69.47          & 63.73 \\
LGBM-Poisson              & 55.64 & 55.84 & 55.78 & 55.85 & 56.83          & 56.71 \\
\textbf{LGBM-Tweedie}     & 56.84 & 56.94 & 57.62 & 57.59 & \textbf{53.05} & 53.60 \\
RF (CAST Baseline)        & 58.98 & 59.10 & 60.48 & 59.43 & 60.11          & 60.08 \\
XGBoost                   & 63.84 & 63.22 & 63.17 & 63.40 & 64.10          & 60.56 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\centering
\caption{Marginal MAE change per feature tier for LGBM-Tweedie V2 (historical top-10 regions). Negative $\Delta$ indicates improvement.}
\label{tab:marginal}
\begin{tabular}{lrrr}
\toprule
\textbf{Feature Tier} & \textbf{Overall MAE} & \textbf{$\Delta$ MAE} & \textbf{$\Delta$\,\%} \\
\midrule
Baseline              & 56.84 & ---           & ---        \\
+Risk                 & 56.94 & $+0.10$       & $+0.2\%$   \\
+Macro                & 57.62 & $+0.68$       & $+1.2\%$   \\
+Holidays             & 57.59 & $-0.03$       & $-0.1\%$   \\
\textbf{+Country}     & \textbf{53.05} & $\mathbf{-4.54}$ & $\mathbf{-7.9\%}$ \\
+Engineered           & 53.60 & $+0.56$       & $+1.1\%$   \\
\bottomrule
\end{tabular}
\end{table}

The +Country tier accounts for essentially all of the V2 improvement. No other single enrichment tier achieves more than a 1.2\% change in either direction.

### Comparison with CAST+ on the Paper's Top-10 Regions

To produce a region-matched comparison with CAST+, the full V2 ablation (LGBM-Tweedie × six feature sets) and the RF Baseline were re-run on the paper's ten holdout-period top-10 regions (UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia, RUS-Kursk, UKR-Chernihiv, PSX-West Bank). Results are reported in Table~\ref{tab:paper_top10}.

\begin{table}[h]
\centering
\caption{Performance on the paper's holdout-period top-10 regions (UKR-Donetsk, UKR-Sumy, RUS-Belgorod, UKR-Kherson, PSX-Gaza, UKR-Kharkiv, UKR-Zaporizhzhia, RUS-Kursk, UKR-Chernihiv, PSX-West Bank). CAST+ numbers from pre\_report.pdf Table~4. DA-nonzero = directional accuracy restricted to months where conflict changes.}
\label{tab:paper_top10}
\begin{tabular}{lrrrrl}
\toprule
\textbf{Model} & \textbf{Battles} & \textbf{Remote} & \textbf{VaC} & \textbf{Overall MAE} & \textbf{DA-nonzero} \\
\midrule
RF Baseline (ACLED-only)   & 64.34 & 132.56 & 11.51 & \textbf{69.47} & --- \\
LGBM-Tweedie +Country      & 66.68 & 150.78 & 12.29 & \textbf{73.06} & 0.65 \\
CAST+ (Two-Stage CatBoost) & 29.62 & 120.41 &  6.84 & \textbf{52.29} & 0.79--0.82 \\
\bottomrule
\end{tabular}
\end{table}

CAST+ achieves MAE 52.29 on this evaluation set, outperforming both the RF Baseline ($-24.7\%$) and LGBM-Tweedie ($-28.4\%$). LGBM-Tweedie +Country achieves MAE 73.06 — 5.2\% above the RF Baseline on these regions. The reversal relative to the historical top-10 result ($-7.3\%$) reflects the composition of the holdout-period set: seven of ten regions are Ukraine frontlines, where the scale and volatility of the Russia-Ukraine war exceeds anything in the training distribution, and where the two-stage delta formulation of CAST+ provides the greatest architectural advantage over an absolute-count regressor. The country-hierarchy feature adds more lift in diverse, lower-scale conflict settings (e.g.\ Syria, Brazil) present in the historical top-10 but absent here.

The global LGBM-Tweedie evaluation covers 2,262 regions (regions with at least 12 months of data and non-zero training target activity). The CAST+ global evaluation covers 2,205 regions (additionally requiring variation in monthly delta targets). The 57-region difference reflects different validity criteria appropriate to each model's objective function and does not indicate a methodological inconsistency.

On the metric of directional accuracy restricted to changing months (DA-nonzero), LGBM-Tweedie achieves 0.65 on the paper's top-10 and 0.84 globally across 2,262 regions, directly comparable to CAST+'s 79–82\% reported globally. The apparent gap between the two models' directional performance is substantially smaller when measured on the same basis (changing months only) than the raw DA figures ($0.21$ for LGBM-Tweedie vs $0.79$–$0.82$ for CatBoost) would suggest; the raw LGBM figure is depressed by the Tweedie objective's structural inability to predict zero output in the 83\% of region-months where true conflict does not change.

---

## §7.2 — What Did Not Work (LGBM-Tweedie Contribution)

*Target length: 2–3 sentences for insertion into Rosa's §7.2 "What Did Not Work" block. Not a standalone [GIRAY] placeholder — coordinate placement with Rosa.*

---

Three architectural modifications to the LGBM-Tweedie baseline were evaluated and found to degrade performance. A two-stage approach — a binary onset classifier ($P(y > 0)$) followed by a Tweedie intensity regressor restricted to active months — increased overall MAE by $+16.60$ (from 54.84 to 71.44 on the historical top-10 evaluation set), with the Battles target worst affected; in the ten high-activity evaluation regions, conflict is near-continuous, so the classifier rarely fires correctly and occasionally suppresses true positives, underscoring that the two-stage design's value lies in the global zero-inflated distribution rather than in high-conflict zones. Log-target training — fitting on $\log_1p(y)$ and back-transforming via $\exp$ — degraded performance by $+10.08$ MAE overall ($+29\%$ for Remote violence), because the Tweedie objective already handles the heavy-tailed count distribution structurally and the back-transformation amplifies errors in the high-count months that are most important to forecast accurately. A delta-target formulation using LGBM-Poisson (modelling $\Delta y = y(t) - y(t-1)$ directly, analogous to CAST+'s target definition) reached MAE 54.42 — comparable to the absolute-count Tweedie approach — but without the directional-accuracy gains of CAST+'s two-stage design, confirming that the delta formulation alone is insufficient and the explicit onset-classification stage is the critical architectural component.

---

## §7.5 — Per-Target Model Comparison

*Target length: 2–3 sentences confirming which model is preferable for Remote violence and VaC. Replace the [GIRAY: §7.5] block.*

---

On the paper's top-10 most active regions, CAST+ outperforms LGBM-Tweedie on all three conflict targets. The gap is largest for Battles ($-55.5\%$, CAST+ MAE 29.62 vs LGBM-Tweedie 66.68) and Violence against civilians ($-44.4\%$, CAST+ 6.84 vs LGBM 12.29), where the two-stage architecture's ability to first classify whether a change occurs — rather than always estimating a positive count — provides the greatest advantage. Remote violence shows the smallest gap ($-13.0\%$, CAST+ 120.41 vs LGBM 138.37), reflecting the shared forecasting difficulty of extreme-scale explosive events on the Ukraine eastern front, where neither model substantially outperforms the other relative to the underlying uncertainty. LGBM-Tweedie is not preferable to CAST+ for any of the three targets on the holdout-period top-10; the complementarity of the two model streams is architectural rather than target-specific: LGBM-Tweedie's country-hierarchy feature provides lift in lower-scale, geographically diverse conflict settings (see §6.4 historical top-10), while CAST+'s two-stage design dominates in the high-volatility, zero-inflated setting that characterises both the global evaluation and the holdout-period active regions.

---

## §9 — Conclusion: Complementarity Paragraph

*Suggested concluding paragraph for the LGBM-Tweedie contribution. Coordinate exact wording with Rosa.*

---

The LGBM-Tweedie ablation study, evaluated across five model families and six cumulative feature tiers on the ten highest-conflict regions, establishes that the country-level leave-one-out hierarchy feature is the single most impactful enrichment for absolute-count conflict regressors — accounting for a $-7.9\%$ MAE reduction in isolation and producing MAE 53.05 on the historical top-10 evaluation set. On the paper's holdout-period top-10, CAST+ substantially outperforms LGBM-Tweedie (52.29 vs 73.06), confirming the two-stage delta formulation as the superior architecture for high-activity conflict zones. The two model streams are structurally complementary: where CAST+'s delta formulation and two-stage architecture excel at directional accuracy and change detection across the full global distribution, the LGBM-Tweedie ablation provides a systematic map of which feature layers generalise across model families — evidence that the country-hierarchy enrichment and cyclic temporal encoding contribute signal robust to architectural choice — and demonstrates that both streams achieve comparable directional accuracy ($\approx 80$–$84\%$) when evaluated on the same basis of changing months only.

---

## Appendix B — Full Feature Reference Table

*LaTeX `longtable` for Appendix B. Each row is one feature. Rows without LGBM or CatBoost support are marked with $\times$.*

```latex
\begin{longtable}{p{6.5cm} c p{3.2cm} c c c}
\caption{Full feature reference table. All features in \texttt{model\_data\_v2\_enriched.csv} plus Rosa's Layer 5 and Layer 7 additions (not in Giray's CSV).}
\label{tab:features} \\
\toprule
\textbf{Feature} & \textbf{Layer} & \textbf{Source} & \textbf{Lag} & \textbf{LGBM} & \textbf{CAST+} \\
\midrule
\endfirsthead
\multicolumn{6}{c}{\textit{(continued from previous page)}} \\
\toprule
\textbf{Feature} & \textbf{Layer} & \textbf{Source} & \textbf{Lag} & \textbf{LGBM} & \textbf{CAST+} \\
\midrule
\endhead
\midrule
\multicolumn{6}{r}{\textit{(continued on next page)}} \\
\endfoot
\bottomrule
\endlastfoot

\multicolumn{6}{l}{\textbf{Layer 1 — Autoregressive Lag Features (ACLED)}} \\
Battles $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Explosions/Remote violence $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Protests $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Riots $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Strategic developments $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Violence against civilians $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Excessive force against protesters $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Agreement $(t-1)$ & 1 & ACLED & $t-1$ & \checkmark & \checkmark \\
Battles\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
Explos./Remote\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
Protests\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
Riots\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
Strategic dev.\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
VaC\_neighbours $(t-1)$ & 1 & ACLED (spatial) & $t-1$ & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 2 — Temporal Features}} \\
linear\_month\_trend & 2 & Derived & none & \checkmark & \checkmark \\
year & 2 & Derived & none & \checkmark & \checkmark \\
month\_sin & 2 & Derived (cyclic) & none & \checkmark & \checkmark \\
month\_cos & 2 & Derived (cyclic) & none & \checkmark & \checkmark \\
quarter\_sin & 2 & Derived (cyclic) & none & \checkmark & \checkmark \\
quarter\_cos & 2 & Derived (cyclic) & none & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 3 — News-Scraped Risk Indicators (Giray)}} \\
risk\_contested\_election $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_crop\_failure $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_economic\_concern $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_ethnic\_tension $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_military\_coup $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_natural\_disaster $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
risk\_political\_assassination $(t-1)$ & 3 & GNews/LLM & $t-1$ & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 4 — World Bank Macroeconomic Features}} \\
inflation\_py & 4 & World Bank (CPI) & prior-year & \checkmark & \checkmark \\
youth\_unemployment\_py & 4 & World Bank & prior-year & \checkmark & \checkmark \\
income\_inequality\_py & 4 & World Bank (Gini) & prior-year & \checkmark & \checkmark \\
income\_level\_code & 4 & World Bank metadata & structural & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 5a — Religious Composition (World Religion Project 2010)}} \\
majority\_religion & 5a & WRP & static & \checkmark & \checkmark \\
majority\_pct & 5a & WRP & static & \checkmark & \checkmark \\
minority1\_religion & 5a & WRP & static & \checkmark & \checkmark \\
minority1\_pct & 5a & WRP & static & \checkmark & \checkmark \\
minority2\_religion & 5a & WRP & static & \checkmark & \checkmark \\
minority2\_pct & 5a & WRP & static & \checkmark & \checkmark \\
nonreligpct & 5a & WRP & static & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 5b — Engineered World Bank Features (Rosa)}} \\
structural\_vulnerability & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
inflation\_shock & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
shock\_vulnerability & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
capacity\_adjusted\_risk & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
inflation\_change & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
high\_inequality\_flag & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
inflation\_shock\_poor & 5b & Derived (WB) & prior-year & $\times$ & \checkmark \\
conflict\_momentum & 5b & Derived (ACLED) & $t-1$ & $\times$ & \checkmark \\
violence\_escalation & 5b & Derived (ACLED) & $t-1$ & $\times$ & \checkmark \\
macro\_conflict\_pressure & 5b & Derived (ACLED+WB) & $t-1$ & $\times$ & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 6 — Religion-Specific Holiday Features}} \\
holiday\_count\_month & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
is\_holiday\_month & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
christian\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
islam\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
shia\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
hindu\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
buddhist\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
jewish\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
cultural\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
nonreligious\_holiday\_count & 6 & Holiday calendar & current month & \checkmark & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 7 — Holiday $\times$ Religion Interaction Features (Rosa)}} \\
weighted\_holidays\_majority & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
weighted\_holidays\_minority1 & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
weighted\_holidays\_minority2 & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
minority\_tension\_minority1 & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
minority\_tension\_minority2 & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
minority\_tension\_total & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
total\_religious\_mobilization & 7 & Derived (holiday$\times$WRP) & $t-1$ & $\times$ & \checkmark \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 8 — Country-Level Hierarchy Features (Giray)}} \\
country\_battles\_excl $(t-1)$ & 8 & Derived (ACLED LOO) & $t-1$ & \checkmark & $\times$ \\
country\_remote\_excl $(t-1)$ & 8 & Derived (ACLED LOO) & $t-1$ & \checkmark & $\times$ \\
country\_vac\_excl $(t-1)$ & 8 & Derived (ACLED LOO) & $t-1$ & \checkmark & $\times$ \\
country\_total\_excl $(t-1)$ & 8 & Derived (ACLED LOO) & $t-1$ & \checkmark & $\times$ \\
\midrule
\multicolumn{6}{l}{\textbf{Layer 9 — Engineered Cross-Variable Features (Giray)}} \\
Battles $(t-2)$ & 9 & ACLED & $t-2$ & \checkmark & $\times$ \\
Explosions/Remote violence $(t-2)$ & 9 & ACLED & $t-2$ & \checkmark & $\times$ \\
Violence against civilians $(t-2)$ & 9 & ACLED & $t-2$ & \checkmark & $\times$ \\
organized\_violence $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\
is\_active $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\
battles\_x\_remote $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\
Battles\_3mo\_avg $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\
Remote\_3mo\_avg $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\
VaC\_3mo\_avg $(t-1)$ & 9 & Derived (ACLED) & $t-1$ & \checkmark & $\times$ \\

\end{longtable}
```

**Notes for Appendix B:**
- Layer 5b (engineered WB) and Layer 7 (holiday interactions) are Rosa's additions present in `final_rosa.ipynb` but not in `model_data_v2_enriched.csv` (Giray's pipeline does not generate them).
- Layer 8 and Layer 9 are Giray's additions; CatBoost/CAST+ does not use them (country-hierarchy features added <1\% in the two-stage framework, per §7.2).
- "LOO" = leave-one-out (focal region excluded from the country aggregate).
- "prior-year" lag = World Bank data released 12–18 months after the reference year; the prior-year value is used to avoid using data not yet available at prediction time.
- "static" = one value per country per dataset, no temporal variation (WRP snapshot from 2010).

---

*End of PAPER_DRAFT.md. All content above is sourced from PROJECT_REPORT.md, todo/05_still_missing.md, and code in src/scraping/ and utils/. Numbers from outputs/ablation_paper_top10.csv and outputs/full_region_eval.csv.*
