#!/bin/bash
# re-run every final evaluation with the corrected score table (amendment A4), 3 at a time
cd "$(dirname "$0")"
run() { PYTHONIOENCODING=utf-8 python -u run_pipeline.py "$@" ; }
run --system tfidf --final > ../results_tfidf_FINAL.log 2>&1 &
run --system npz --npz_dir ../kaggle_out/deberta/deberta_s1 --name deberta_s1 --final > ../results_deberta_s1_FINAL.log 2>&1 &
run --system npz --npz_dir ../kaggle_out/qwen7_s1/qwen7_s1 --name qwen7_s1 --final > ../results_qwen7_s1_FINAL.log 2>&1 &
wait
run --system npz --npz_dir ../kaggle_out/deberta/deberta_s2 --name deberta_s2 --final > ../results_deberta_s2_FINAL.log 2>&1 &
run --system npz --npz_dir ../kaggle_out/deberta/deberta_s3 --name deberta_s3 --final > ../results_deberta_s3_FINAL.log 2>&1 &
PYTHONIOENCODING=utf-8 python baselines_b0_b3.py > ../results_b0b3.log 2>&1 &
wait
echo ALL_DONE
