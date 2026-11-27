# Knowing When Not to Score: Automated CVSS v4.0 Vector Assignment with Risk-Controlled Abstention

Code, data and predictions of SEVERA, the method of the paper.

This repository contains the data snapshot, code, model outputs and analysis scripts of the paper. Every number, table and figure of the paper can be regenerated on a CPU from the files here; only the training of the perception modules needs a GPU (free Kaggle T4), and their outputs are included.

## What SEVERA does

Given what is known when a CVE record is published (the description, the CNA's CWE identifiers and, if any, the CNA's CVSS v3.1 vector), SEVERA predicts the eleven CVSS v4.0 base metrics:

1. a perception module (any per-metric classifier; here TF-IDF, DeBERTa-v3-base or Qwen2.5-7B with 4-bit LoRA) gives per-metric probabilities;
2. a knowledge base (CWE priors, v3.1 to v4.0 correspondences, metric compatibility) and a per-metric combiner recalibrate them;
3. an inference engine forms an exactly normalised distribution over all 104,976 base vectors and returns the vector that minimises the expected error under the official CVSS v4.0 scoring function (about 20 ms per CVE on a CPU);
4. an abstention gate calibrated with Learn-then-Test within provenance groups answers only when the band error of its answers is bounded with high probability;
5. an explanation facility names the metric whose uncertainty moves most probability across severity bands.

## Layout

| Path | Content |
|---|---|
| `data/splits.jsonl.gz` | Frozen benchmark: 48,256 CVEs with a CNA-published CVSS v4.0 vector, temporal splits, inputs and labels |
| `data/deploy.jsonl.gz` | Deployment population: 46,443 CVEs published Jul to Nov 2026 without any v4.0 vector |
| `data/cvss4_score_table.json.gz` | Score, severity and MacroVector of all 104,976 base vectors |
| `data/SHA256SUMS_uncompressed.txt` | Digests of the uncompressed published data files |
| `code/` | Data collection, splits, knowledge base, inference engine, abstention, analyses, tables, figures |
| `kaggle/` | Training and prompting scripts run on Kaggle |
| `kaggle_out/` | Per-metric probabilities of every perception module (dev, calibration, test, and deployment for DeBERTa): DeBERTa-v3-base and Qwen2.5-7B, three seeds each; B4 generations |
| `results/` | Per-system result files and per-CVE predictions on the test split |
| `figures/` | Figures of the paper and the graphical abstract |
| `PROTOCOL.md` | The evaluation protocol frozen before the experiments, with every amendment |

## Reproducing the results

Requirements (CPU part): Python 3.12, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `cvss==3.6`.

```
gunzip -k data/*.gz
python code/score_table.py data/cvss4_score_table.json     # optional: rebuilds the score table (FIRST rounding)
cd code
python run_pipeline.py --system tfidf --final
python run_pipeline.py --system npz --npz_dir ../kaggle_out/deberta/deberta_s1 --name deberta_s1 --final
python run_pipeline.py --system npz --npz_dir ../kaggle_out/qwen7_s1/qwen7_s1 --name qwen7_s1 --final
python baselines_b0_b3.py
python revision_analyses.py --npz_dir ../kaggle_out/qwen7_s1/qwen7_s1 --name qwen7_s1
python deploy.py --npz_dir ../kaggle_out/deberta/deberta_s1 --name deberta_s1
python eval_b4.py --b4 ../kaggle_out/b4/b4/b4_predictions.jsonl --systems tfidf=results/tfidf_FINAL_arrays.npz deberta_s1=results/deberta_s1_FINAL_arrays.npz qwen7_s1=results/qwen7_s1_FINAL_arrays.npz --out results/b4_comparison.json
python make_tables.py
```

Other seeds are run the same way with their `kaggle_out` folders.

In the published copies of `splits.jsonl` and `deploy.jsonl`, the names of one vendor and its products were replaced by `[vendor product]` in 128 descriptions (90 and 38). The experiments used the original NVD text; the per-metric probabilities in `kaggle_out/` were computed on it, so every result based on them is reproduced exactly, and the original records can be retrieved from the NVD with `code/fetch_nvd.py`. Retraining B1 from the published copy may differ marginally. To rebuild the snapshot from scratch, run `code/fetch_nvd.py`, `code/build_dataset.py`, `code/build_splits.py` and `code/build_deploy.py`; a new download will differ from the frozen snapshot because NVD records change over time.

The perception modules are trained with `code/kaggle_train_heads.py` (see `kaggle/*/run.py` for the exact arguments) and B4 with `code/kaggle_b4_prompt.py`.

## Scoring function

Scores are computed with the Red Hat `cvss` library with the rounding of FIRST's calculator (`Math.round(x * 10) / 10` on the raw value). With this rounding the library agrees with all 41,270 scored vectors of FIRST's calculator validation set (`FIRSTdotorg/cvss-resources`); with the library's default rounding it disagrees on 46 of the set's 10,000 base vectors. See `PROTOCOL.md`, amendment A4.

## Data

The vulnerability records come from the NVD API 2.0 (snapshot of 21 November 2026). This product uses data from the NVD API but is not endorsed or certified by the NVD. CVE records are subject to the CVE Program's terms of use.

## License

Code: MIT (see `LICENSE`).
