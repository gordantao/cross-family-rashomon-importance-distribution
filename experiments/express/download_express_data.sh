#!/bin/bash
# Download the public stand-in data for the EXPRESS experiment into
# experiments/express/data/raw/ (gitignored). ~430MB in total, most of it the
# BeatAML2 expression and count matrices. Re-running resumes partial files,
# skips finished ones, and re-checks every file's sha256.
#
# EXPRESS (Janizek et al. 2023) used non-public Beat AML drug-combination data.
# The public stand-in pairs:
#
#   Outcome: Eide et al. 2023, "Clinical correlates of venetoclax-based combination
#     sensitivities to augment acute myeloid leukemia therapy", Blood Cancer
#     Discovery 4(6):452. Supplementary Table S9 ("Full ex vivo AUC values and
#     clinical annotations", ~4MB xlsx). Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10618724
#     Used under the journal's (AACR) terms. Both PMC and aacrjournals.org put
#     downloads behind a browser challenge, and the article isn't in the PMC
#     open-access subset, so this script can't fetch it. Download it by hand in a
#     browser and save it as data/raw/bcd-23-0014_table_s9_suppst9.xlsx; the script
#     then checks it's a real xlsx rather than a challenge page.
#
#   Expression, sample mapping, clinical, mutations, single-agent AUCs: BeatAML2
#     (Bottomly et al. 2022, Cancer Cell 40:850), https://github.com/biodev/beataml2.0_data,
#     licensed CC-BY-4.0. The files are stored in Git LFS, so they're fetched from
#     media.githubusercontent.com (raw.githubusercontent.com returns LFS pointers),
#     pinned to commit 6e403e1. The sha256 values are the LFS object ids.
#
# Don't commit or re-host any of these files: this repo is public, and the
# files hold patient-level values.

set -euo pipefail

RAW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/data/raw"
mkdir -p "$RAW_DIR"

BEATAML=https://media.githubusercontent.com/media/biodev/beataml2.0_data/6e403e142469dfb8ab20af0a4e6bf0fdaaf474e8

# file  sha256  description
FILES=(
    "beataml_waves1to4_norm_exp_dbgap.txt d5745b9dbf46dba866a3c7370bb0ba73b363ecdd21e01cc1d916b4e3021e6f87"    # normalized log2 expression, genes x RNA samples
    "beataml_waves1to4_counts_dbgap.txt 26302116e26138e980a7816303a0886a89a3c951f3ba673051a46c96a658680d"      # raw read counts, genes x RNA samples
    "beataml_waves1to4_sample_mapping.xlsx 5581140cedd3348ec78bed878c27aa12c9538a5fab115324b91ca56aa138f0fc"   # labId <-> RNA/DNA sample <-> patient
    "beataml_wv1to4_clinical.xlsx bc692f647f93945e1cf883271af5501bf75c8af3e681676241093c198ed167ad"            # per-sample clinical: FAB, ELN, diagnosis, etc.
    "beataml_probit_curve_fits_v4_dbgap.txt d4bc5f0d91f66314107411e0f2511adc987e29df5b83d9b03df56d3d12928314"  # single-agent inhibitor AUC/IC50
    "beataml_wes_wv1to4_mutations_dbgap.txt 5a5a5eb8f492b1385aebe85c490b9333f65590f09391a7c1951b04dd1dba1680"  # WES mutation calls
    "beataml_drug_families.xlsx 636e27515bc868194686689c635657293b334741dd64b0253e3d1d0b4adc35a0"              # inhibitor -> drug family
)

for entry in "${FILES[@]}"; do
    read -r name sha <<< "$entry"
    out="$RAW_DIR/$name"
    if [[ -f "$out.done" ]]; then
        echo "skip  $name"
        continue
    fi
    echo "fetch $name"
    curl --fail --location --retry 5 --retry-delay 10 -C - -o "$out" "$BEATAML/$name"
    echo "$sha  $out" | shasum -a 256 -c -
    touch "$out.done"
done

S9="$RAW_DIR/bcd-23-0014_table_s9_suppst9.xlsx"
if [[ ! -f "$S9" ]]; then
    echo
    echo "MISSING: Eide et al. 2023 Table S9. Download it in a browser from"
    echo "  https://pmc.ncbi.nlm.nih.gov/articles/instance/10618724/bin/bcd-23-0014_table_s9_suppst9.xlsx"
    echo "(or the Supplementary Data on the Blood Cancer Discovery article page) and save it as"
    echo "  $S9"
elif [[ "$(head -c 2 "$S9")" != "PK" ]]; then
    echo "ERROR: $S9 isn't an xlsx (zip) file; it's probably a saved challenge page." >&2
    exit 1
else
    # sha256 of the copy downloaded 2026-11-12 (4,243,458 bytes). A mismatch only
    # warns, since the publisher may reissue the file; check the column layout then.
    S9_SHA=3090edf4e1e76ea35bd5e58a851b225b07ffbd87783672a12fdd3af7d166bc1b
    if echo "$S9_SHA  $S9" | shasum -a 256 -c - >/dev/null 2>&1; then
        echo "ok    $(basename "$S9") (manual download, sha256 matches)"
    else
        echo "WARN  $(basename "$S9") is an xlsx but its sha256 differs from the 2026-11-12 copy" >&2
    fi
fi

echo "Downloaded to $RAW_DIR:"
ls -lh "$RAW_DIR" | grep -v '\.done$'
