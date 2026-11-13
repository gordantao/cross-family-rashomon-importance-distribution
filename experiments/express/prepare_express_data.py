#!/usr/bin/env python3
"""Build the EXPRESS stand-in row table and expression matrix from the raw downloads.

EXPRESS (Janizek et al. 2023) explained ex vivo drug-combination synergy in AML
from patient expression, but its Beat AML combination data are not public. This
experiment uses a public stand-in instead, so it is NOT the original EXPRESS data:

    Outcome      Eide et al. 2023 (Blood Cancer Discovery 4(6):452), Supplementary
                 Table S9: ex vivo probit AUCs of venetoclax (VEN), 26 partner
                 drugs and the 26 "VEN + X" combinations, one row per (specimen,
                 drug). Downloaded by hand (see download_express_data.sh); used
                 under the journal's (AACR) terms.
    Expression   BeatAML2 (Bottomly et al. 2022, Cancer Cell 40:850; CC-BY-4.0):
                 normalized log2 expression, genes x RNA samples, and the sample
                 mapping that links a specimen's labId to its RNA sample.

Reads data/raw/ (download_express_data.sh) and writes data/processed/
(gitignored -- these are patient-level values, never commit them):

    rows.csv             one row per (patient, combo): patient_id, lab_id, combo,
                         partner, auc_combo, auc_ven, auc_partner (probit AUCs,
                         floored), cr, log_cr, specimen_type
    expression.csv       one row per patient: patient_id x genes (protein-coding
                         genes plus the answer-key genes, gene symbols as columns)
    prepare_summary.json counts, the AUC floor, per-combo row counts, the
                         all-rows class balance (sanity check only) and the raw
                         files' sha256

Join: Table S9's lab_id <-> the mapping's labId, keeping specimens that aren't
healthy controls (rna_control == "No") and whose RNA sample is a column of the
expression matrix. There is no diagnosis filter beyond Table S9's own cohort:
its dxAtSpecimenAcquisition is AML for almost every specimen, but a few are
acute leukaemias of ambiguous lineage, and those are kept (counts in
prepare_summary.json). Every matched patient has exactly one specimen and one
RNA sample (checked).

Combination ratio, per (patient, combo) row:
    CR = AUC_combo / min(AUC_VEN, AUC_partner)
with every AUC first floored at 1% of the maximum possible AUC (the scale
behind Table S9's auc_pct_of_max, recovered as probit_auc / auc_pct_of_max * 100
and checked constant). About 3% of AUCs sit at the probit fit's floor (~1e-9),
which would otherwise make the ratio explode. CR < 1 means the combination kills
more than either drug alone. log CR (natural log) is kept as is: the label --
a per-combo z-score of log CR, or CR < 0.75 -- is fit in
run_express_comparison.py on the train patients only, so nothing here is fit
on outcomes beyond computing the raw ratio.

Every combo with an RNA-matched row is kept, and every specimen type (peripheral
blood, bone marrow, leukapheresis); specimen_type is carried for a later
sensitivity analysis and is not a feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("prepare_express_data")

TABLE_S9_FILE = "bcd-23-0014_table_s9_suppst9.xlsx"
TABLE_S9_SHEET = "Supp Table S9"
EXPRESSION_FILE = "beataml_waves1to4_norm_exp_dbgap.txt"
MAPPING_FILE = "beataml_waves1to4_sample_mapping.xlsx"
EXPRESSION_META_COLUMNS = ("stable_id", "display_label", "description", "biotype")

VEN = "VEN"
COMBO_PREFIX = "VEN + "

# Genes the EXPRESS paper and the venetoclax literature tie to VEN-combination
# response: BCL2 (the target), monocytic-differentiation markers that mark VEN
# resistance (CD14, S100A8, S100A9), MEIS1/CITED2 (EXPRESS's stemness /
# synergy genes), CDA and NT5C2 (nucleoside-analogue metabolism) and DLL3.
# Forced into the expression table (and into the run script's feature panel)
# whatever their biotype or variance; run_express_comparison.py reports where
# each method ranks them.
ANSWER_KEY_GENES = ("BCL2", "CD14", "S100A8", "S100A9", "MEIS1", "CITED2", "CDA", "NT5C2", "DLL3")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_table_s9(raw_dir: Path) -> tuple[pd.DataFrame, float]:
    """Table S9 plus the maximum possible AUC (the auc_pct_of_max scale)."""

    s9 = pd.read_excel(raw_dir / TABLE_S9_FILE, sheet_name=TABLE_S9_SHEET)
    missing = {"patient_id", "lab_id", "drug", "drug_type", "probit_auc", "auc_pct_of_max", "specimenType",
               "dxAtSpecimenAcquisition"} - set(s9.columns)
    if missing:
        raise ValueError(f"{TABLE_S9_FILE} lacks columns {sorted(missing)}; has the publisher changed the layout?")
    if s9.duplicated(["lab_id", "drug"]).any():
        raise ValueError("Table S9 has more than one row per (lab_id, drug)")

    scale = (s9["probit_auc"] / s9["auc_pct_of_max"]).to_numpy()
    if not np.allclose(scale, scale[0], rtol=1e-6):
        raise ValueError("probit_auc / auc_pct_of_max isn't constant; can't recover the maximum AUC")
    return s9, float(np.median(scale) * 100.0)


def load_mapping(raw_dir: Path, expression_samples: set[str]) -> pd.DataFrame:
    """labId -> RNA sample for specimens that aren't healthy controls, with expression."""

    mapping = pd.read_excel(raw_dir / MAPPING_FILE)
    mapping = mapping[mapping["dbgap_rnaseq_sample"].isin(expression_samples) & mapping["rna_control"].eq("No")]
    if mapping["labId"].duplicated().any():
        raise ValueError("a labId maps to more than one expressed RNA sample")
    return mapping[["labId", "patientId", "dbgap_rnaseq_sample"]]


def build_rows(s9: pd.DataFrame, mapping: pd.DataFrame, auc_floor: float) -> pd.DataFrame:
    """One row per (patient, combo) with the floored AUCs and the combination ratio."""

    matched = s9.merge(mapping, left_on="lab_id", right_on="labId", how="inner", validate="many_to_one")
    if not (matched["patient_id"] == matched["patientId"]).all():
        raise ValueError("Table S9 patient_id disagrees with the BeatAML mapping's patientId")
    if matched.groupby("patient_id")["lab_id"].nunique().max() > 1:
        raise ValueError("a patient has more than one RNA-matched specimen; pick one per patient first")

    auc = matched.pivot(index="lab_id", columns="drug", values="probit_auc").clip(lower=auc_floor)
    specimen = matched.drop_duplicates("lab_id").set_index("lab_id")

    frames = []
    combos = sorted(d for d in matched.loc[matched["drug_type"] == "combo", "drug"].unique())
    for combo in combos:
        if not combo.startswith(COMBO_PREFIX):
            raise ValueError(f"unexpected combo name {combo!r}; expected 'VEN + <partner>'")
        partner = combo[len(COMBO_PREFIX):]
        sub = auc[[combo, VEN, partner]].dropna()
        cr = sub[combo] / np.minimum(sub[VEN], sub[partner])
        frames.append(pd.DataFrame({
            "patient_id": specimen.loc[sub.index, "patient_id"].to_numpy(),
            "lab_id": sub.index.to_numpy(),
            "combo": combo,
            "partner": partner,
            "auc_combo": sub[combo].to_numpy(),
            "auc_ven": sub[VEN].to_numpy(),
            "auc_partner": sub[partner].to_numpy(),
            "cr": cr.to_numpy(),
            "log_cr": np.log(cr.to_numpy()),
            "specimen_type": specimen.loc[sub.index, "specimenType"].to_numpy(),
            "diagnosis": specimen.loc[sub.index, "dxAtSpecimenAcquisition"].to_numpy(),
            "rna_sample": specimen.loc[sub.index, "dbgap_rnaseq_sample"].to_numpy(),
        }))
    rows = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "combo"], ignore_index=True)
    n_combo_rows = int((matched["drug_type"] == "combo").sum())
    if len(rows) != n_combo_rows:
        logger.warning(f"{n_combo_rows - len(rows)} RNA-matched combo rows lack a VEN or partner single-agent AUC; dropped")
    return rows


def load_expression(raw_dir: Path, samples_by_patient: pd.Series) -> tuple[pd.DataFrame, dict]:
    """patient_id x gene log2 expression: protein-coding genes plus the answer key."""

    path = raw_dir / EXPRESSION_FILE
    header = pd.read_csv(path, sep="\t", nrows=0).columns
    wanted = [c for c in samples_by_patient.to_numpy() if c in set(header)]
    if len(wanted) != len(samples_by_patient):
        raise ValueError("some matched RNA samples are missing from the expression matrix")
    expr = pd.read_csv(path, sep="\t", usecols=list(EXPRESSION_META_COLUMNS) + wanted)
    if expr["display_label"].isna().any() or expr["display_label"].duplicated().any():
        raise ValueError("expected one unique gene symbol (display_label) per expression row")

    keep = expr["biotype"].eq("protein_coding") | expr["display_label"].isin(ANSWER_KEY_GENES)
    expr = expr[keep]
    matrix = expr.set_index("display_label")[wanted].T
    patient_of_sample = pd.Series(samples_by_patient.index, index=samples_by_patient.to_numpy())
    matrix.index = patient_of_sample.loc[matrix.index].to_numpy()
    matrix.index.name = "patient_id"
    matrix.columns.name = None
    matrix = matrix.sort_index()[sorted(matrix.columns)]
    if matrix.isna().any().any():
        raise ValueError("expression matrix has missing values")
    info = {
        "n_genes": int(matrix.shape[1]),
        "n_protein_coding": int(expr["biotype"].eq("protein_coding").sum()),
        "answer_key_biotypes": dict(zip(expr.loc[expr["display_label"].isin(ANSWER_KEY_GENES), "display_label"],
                                        expr.loc[expr["display_label"].isin(ANSWER_KEY_GENES), "biotype"])),
    }
    return matrix, info


def build(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_dir = Path(args.raw_dir)
    s9, max_auc = load_table_s9(raw_dir)
    auc_floor = args.auc_floor_fraction * max_auc
    n_floored = int((s9["probit_auc"] < auc_floor).sum())
    logger.info(f"max AUC {max_auc:.4f}; floor {auc_floor:.4f} ({n_floored}/{len(s9)} Table S9 AUCs below it)")

    expression_samples = set(pd.read_csv(raw_dir / EXPRESSION_FILE, sep="\t", nrows=0).columns) - set(EXPRESSION_META_COLUMNS)
    mapping = load_mapping(raw_dir, expression_samples)
    rows = build_rows(s9, mapping, auc_floor)

    samples_by_patient = rows.drop_duplicates("patient_id").set_index("patient_id")["rna_sample"]
    expression, expression_info = load_expression(raw_dir, samples_by_patient)
    patients_per_diagnosis = rows.drop_duplicates("patient_id")["diagnosis"].value_counts(dropna=False)
    rows = rows.drop(columns=["rna_sample", "diagnosis"])

    missing_answer_key = sorted(set(ANSWER_KEY_GENES) - set(expression.columns))
    if missing_answer_key:
        logger.warning(f"answer-key genes absent from the expression matrix: {missing_answer_key}")

    all_combos = sorted(s9.loc[s9["drug_type"] == "combo", "drug"].unique())
    log_cr_by_combo = rows.groupby("combo")["log_cr"]
    summary = {
        "data_note": "Public stand-in for EXPRESS: Eide et al. 2023 Table S9 outcomes + BeatAML2 expression.",
        "auc_floor_fraction": args.auc_floor_fraction,
        "max_auc": max_auc,
        "auc_floor": auc_floor,
        "n_table_s9_aucs_below_floor": n_floored,
        "n_table_s9_aucs": int(len(s9)),
        "n_rows_aucs_floored": {col: int(np.isclose(rows[col], auc_floor).sum())
                                for col in ("auc_combo", "auc_ven", "auc_partner")},
        "n_table_s9_patients": int(s9["patient_id"].nunique()),
        "n_patients": int(rows["patient_id"].nunique()),
        "n_rna_samples": int(len(samples_by_patient)),
        "n_rows": int(len(rows)),
        "n_combos": int(rows["combo"].nunique()),
        "combos_without_rna_matched_rows": sorted(set(all_combos) - set(rows["combo"])),
        "rows_per_combo": rows["combo"].value_counts().sort_index().to_dict(),
        "rows_per_specimen_type": rows["specimen_type"].value_counts().to_dict(),
        "patients_per_specimen_type": rows.drop_duplicates("patient_id")["specimen_type"].value_counts().to_dict(),
        "patients_per_diagnosis": {str(k): int(v) for k, v in patients_per_diagnosis.items()},
        "expression": expression_info,
        "answer_key_genes": list(ANSWER_KEY_GENES),
        "missing_answer_key_genes": missing_answer_key,
        # A sanity check over every row, not the label: the real label is fit on the
        # train patients in run_express_comparison.py.
        "all_rows_sanity_check": {
            "log_cr_mean": float(rows["log_cr"].mean()),
            "log_cr_sd": float(rows["log_cr"].std()),
            "fraction_cr_lt_1": float((rows["cr"] < 1).mean()),
            "fraction_cr_lt_075": float((rows["cr"] < 0.75).mean()),
            "fraction_below_own_combo_mean": float((rows["log_cr"] < log_cr_by_combo.transform("mean")).mean()),
        },
        "settings": {key: value for key, value in vars(args).items() if key not in ("raw_dir", "out_dir")},
        "raw_file_sha256": {name: _sha256(raw_dir / name) for name in (TABLE_S9_FILE, EXPRESSION_FILE, MAPPING_FILE)},
    }
    return rows, expression, summary


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", default=str(SCRIPT_DIR / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "data" / "processed"))
    parser.add_argument("--auc-floor-fraction", type=float, default=0.01,
                        help="Floor every AUC at this fraction of the maximum possible AUC before the ratio (default: 0.01)")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    args = _parse_args()
    rows, expression, summary = build(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out_dir / "rows.csv", index=False)
    expression.to_csv(out_dir / "expression.csv")
    (out_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"{summary['n_rows']} (patient, combo) rows, {summary['n_patients']} patients, "
                f"{summary['n_combos']} combos, {summary['expression']['n_genes']} genes")
    logger.info(f"all-rows sanity check: {summary['all_rows_sanity_check']}")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    sys.exit(main())
