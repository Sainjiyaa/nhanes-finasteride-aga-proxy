#!/usr/bin/env bash
set -euo pipefail

python scripts/01_download_nhanes.py
python scripts/02_build_proxy_cohort.py
python scripts/03_validation_analyses.py
Rscript scripts/04_survey_weighted_analyses.R
