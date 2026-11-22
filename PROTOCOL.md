# Evaluation protocol (frozen 2026-11-21, before any result was computed)

This file records the protocol as it was written before the experiments, and every amendment made afterwards, with its date and reason.

## Data
- Source: NVD API 2.0, downloaded on 2026-11-21. All CVE records published from 2023-01-01 to 2026-11-20, excluding rejected records.
- File: `data/splits.jsonl` (SHA-256 of the frozen file used in the experiments: `15cfe7803b5d6ddddfda80057899f6e7c9220246bd2554c5b85fa563460696af`; the published copy masks the names of one vendor's products in 90 descriptions, see README).
- Label: the eleven base metrics of the CVSS v4.0 vector published by the CVE's own CNA. The CVSS-B score is recomputed from the base metrics (see amendment A4 for the rounding), without threat metrics.
- Inputs: the description, the CWE identifiers attributed by the CNA, and the CNA's CVSS v3.1 vector when present.
- Temporal splits:
  - training of perception modules: published before 2026-03-01;
  - development: 2026-03-01 to 2026-03-31, held out from model training (knowledge layer, hyperparameters); its first half fits the per-metric combiner, its second half selects the compatibility weight λ;
  - calibration: 2026-04-01 to 2026-06-30 (abstention thresholds only);
  - test: 2026-07-01 to 2026-11-20, read once per system after all choices were frozen.
- Provenance groups: VulDB, VulnCheck, GitHub, all other CNAs.

## Systems
| Code | System |
|---|---|
| B0 | Majority value per metric; majority per CWE group |
| B1 | TF-IDF (word 1-2 and character 3-5 n-grams) and one logistic regression per metric |
| B2 | DeBERTa-v3-base, eleven heads, three seeds |
| B3 | Most frequent v4.0 vector observed in training for each v3.1 vector (only where a v3.1 vector exists) |
| B4 | Qwen2.5-7B-Instruct without fine-tuning: v4.0 metric summary in the prompt and eight BM25-retrieved training examples, on a stratified sample of 2,000 test CVEs |
| B5 | Qwen2.5-7B with 4-bit low-rank adapters, eleven heads, per-metric argmax (three seeds if the free GPU budget allows, otherwise fewer, declared) |
| SEVERA | B5 perception module, knowledge layer (per-metric temperature, pairwise compatibility, CWE factor, v3.1 factor, fitted on development data), minimum-Bayes-risk decoding, per-group risk-controlled abstention, explanation |
| SEVERA-enc | the same layer on B2 |

## Decoding
- Absolute score loss: median of the predictive score distribution, then the most probable vector with that score.
- Band loss: most probable band, then the most probable vector in that band.

## Abstention
- Confidence: probability of the returned band.
- Thresholds chosen on the calibration split with Learn-then-Test, exact binomial p-values, δ = 0.05.
- Target band-error levels α ∈ {0.05, 0.10, 0.15}.
- Two variants: one global threshold, one threshold per provenance group (Mondrian).
- Check: realised selective error and coverage on the test split, overall and per group.

## Measures
- Exact vector; exact MacroVector; accuracy and macro-F1 per metric.
- Mean absolute error of the CVSS-B score; exact band; quadratically weighted kappa of the band.
- Risk-coverage curve; AURC.

## Statistics
- Percentile bootstrap with 2,000 resamples (see A3).
- Exact McNemar tests for exact band and exact vector: SEVERA vs B5, SEVERA vs B1, B5 vs B2; Holm correction.

## Analyses fixed in advance
- Per provenance group; CNAs unseen in training; with and without a v3.1 input.
- Ablations: without pairwise compatibility, without CWE factor, without v3.1 factor, argmax instead of minimum-Bayes-risk decoding, global instead of per-group thresholds.

## Amendments
- **A1 (2026-11-21, before the test split was read).** Learn-then-Test uses a fixed threshold grid: 1 − 10^k for 13 values of k evenly spaced in [−4, −1], then 0.89 to 0.50 in steps of 0.01 (53 values). The first implementation used the observed confidences as the grid, so the sequence started on selections of one or two CVEs whose p-value can never fall below δ. An empty selection has zero selective risk and does not stop the sequence.
- **A2 (2026-11-21, before the test split was read).** The primary procedure becomes Learn-then-Test with Bonferroni correction over the fixed grid (level δ/53 per threshold); the fixed-sequence procedure is still reported for comparison. Reason: on development data with the TF-IDF module, the fixed sequence stopped at the first strict threshold, where a few errors suffice to prevent rejection, and per-group calibration then certified VulDB only. The primary variant is per group with Bonferroni.
- **A3 (2026-11-21, before the test split was read).** Intervals for absolute levels use a bootstrap over CVEs stratified by provenance group; intervals for paired differences between systems use a bootstrap clustered by CNA (2,000 resamples each). Reason: three CNAs hold most labels, so a clustered bootstrap of absolute levels mostly measures the heterogeneity between those three CNAs and gives uninformative intervals.
- **Freeze (2026-11-22, 02:00, before the test split was read).** Primary decoder: score decoder; secondary: band decoder. Primary abstention: per group with Bonferroni. λ chosen on the second half of the development split by joint log-likelihood for every system.
- **A4 (2026-11-22, after the first reading of the test split).** Correction of the scoring function's rounding. The Red Hat `cvss` 3.6 library adds an epsilon before rounding to one decimal and disagrees with FIRST's calculator validation set (`FIRSTdotorg/cvss-resources`, `vectorFiles/reference-scores`) on 46 of its 10,000 base-only vectors (always +0.1). With the rounding of FIRST's calculator, `Math.round(x * 10) / 10` on the raw double, all 41,270 scored vectors of the set agree. The score table of the 104,976 base vectors was recomputed: 522 scores change by 0.1 and no band changes. All results were recomputed with the corrected table; no other choice changed.
- **Additional and post hoc analyses (2026-11-22).** Requested by an internal review and labelled as such in the paper: practitioner baselines (v3.1 band read directly; B3 where a v3.1 vector exists, the model elsewhere), additional ablation rows (K2 with Bayes decision; recalibration with Bayes decision; joint MAP), Clopper-Pearson intervals for realised selective errors, realised errors per (month, group) cell, CNA-level sign-flip permutation tests, a severity-cue audit of descriptions, the explanation-facility check, and a per-group calibration split by availability of a v3.1 vector.
