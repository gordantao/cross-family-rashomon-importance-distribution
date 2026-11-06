#!/bin/bash
# Download the raw GDSC drug-response and Cell Model Passports omics files into
# experiments/gdsc/data/raw/ (gitignored). ~1.2GB in total, almost all of it the
# merged RNA-seq zip. Re-running resumes partial files and skips finished ones.
#
# Terms: the Sanger Data Usage Policy grants a non-exclusive, non-transferable
# right to use these files for internal research and education, and excludes
# resale and commercial services (https://depmap.sanger.ac.uk/documentation/
# data-usage-policy/). Don't commit them or re-host them; this repo is public.
#
# Sources:
#   GDSC release 8.5 (Oct 2023): https://www.cancerrxgene.org/downloads
#   Cell Model Passports:        https://cellmodelpassports.sanger.ac.uk/downloads

set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/raw"
mkdir -p "$RAW_DIR"

GDSC=https://cog.sanger.ac.uk/cancerrxgene/GDSC_release8.5
CMP=https://cog.sanger.ac.uk/cmp/download

FILES=(
    "$GDSC/GDSC2_fitted_dose_response_27Oct23.xlsx"   # LN_IC50 / AUC / Z_SCORE per (cell line, drug)
    "$CMP/model_list_latest.csv.gz"                   # model_id metadata: tissue, MSI status, data availability
    "$CMP/rnaseq_merged_20250117.zip"                 # Sanger+Broad RNA-seq, TPM/FPKM/counts
    "$CMP/mutations_summary_latest.csv.gz"            # driver mutations per model (cancer_driver calls)
    "$CMP/cnv_summary_latest.csv.gz"                  # per-gene copy-number category per model (~785 cancer genes)
)

for url in "${FILES[@]}"; do
    out="$RAW_DIR/$(basename "$url")"
    if [[ -f "$out.done" ]]; then
        echo "skip  $(basename "$out")"
        continue
    fi
    echo "fetch $(basename "$out")"
    curl -fL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
    touch "$out.done"
done

echo "Downloaded to $RAW_DIR:"
ls -lh "$RAW_DIR" | grep -v '\.done$'
