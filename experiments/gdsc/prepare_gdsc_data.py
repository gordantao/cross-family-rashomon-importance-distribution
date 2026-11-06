#!/usr/bin/env python3
"""Build the GDSC feature panel and drug-response labels from the raw downloads.

Reads data/raw/ (download_gdsc_data.sh) and writes data/processed/:

    features.csv        one row per cell line (model_id), one shared,
                        drug-agnostic feature panel for every task
    responses.csv       one row per (task, cell line): LN_IC50, Z_SCORE, AUC and
                        the binary label `sensitive` = Z_SCORE < --z-threshold
    panel_summary.json  block sizes, class balance per task, and a sanity check of
                        each literature biomarker's univariate association with
                        its task's label

The panel is chosen without looking at any drug response, so no task's labels
leak into its own features:
    mut_<GENE>    driver mutation (Cell Model Passports `cancer_driver`), genes
                  mutated in >= --min-mutation-freq of cell lines
    amp_<GENE>    amplification, genes amplified in >= --min-amp-freq
    del_<GENE>    deletion, genes deleted in >= --min-del-freq
    expr_<GENE>   log2(TPM + 1), the --n-variable-genes most variable expressed genes
    tissue_<X>    tissues with >= --min-tissue-lines cell lines
    msi_high      microsatellite-instability high
plus every feature in any task's answer key (gdsc_tasks.biomarker_features()),
whatever its frequency or variance. Including all five tasks' biomarkers in every
task's panel is deliberate: each task's biomarkers are decoys for the others.

Cell lines enter if they were screened for at least one task and have mutation,
copy-number and RNA-seq data.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gdsc_tasks import TASKS, biomarker_features  # noqa: E402

logger = logging.getLogger("prepare_gdsc_data")

DOSE_RESPONSE_FILE = "GDSC2_fitted_dose_response_27Oct23.xlsx"
MODEL_LIST_FILE = "model_list_latest.csv.gz"
MUTATIONS_FILE = "mutations_summary_latest.csv.gz"
CNV_FILE = "cnv_summary_latest.csv.gz"
RNASEQ_ZIP = "rnaseq_merged_20250117.zip"
RNASEQ_TPM_MEMBER = "rnaseq_merged_rsem_tpm_20250117.csv"


def _feature_name(prefix: str, label: str) -> str:
    return f"{prefix}_{re.sub(r'[^0-9A-Za-z]+', '_', str(label)).strip('_')}"


def load_responses(raw_dir: Path, z_threshold: float) -> pd.DataFrame:
    dose = pd.read_excel(raw_dir / DOSE_RESPONSE_FILE)
    frames = []
    for task_name, task in TASKS.items():
        sub = dose[dose["DRUG_ID"] == task["drug_id"]]
        if sub.empty:
            raise ValueError(f"GDSC2 has no rows for {task_name} (DRUG_ID {task['drug_id']})")
        if sub["SANGER_MODEL_ID"].duplicated().any():
            raise ValueError(f"{task_name}: more than one curve per cell line")
        frames.append(pd.DataFrame({
            "task": task_name,
            "model_id": sub["SANGER_MODEL_ID"].to_numpy(),
            "cell_line_name": sub["CELL_LINE_NAME"].to_numpy(),
            "drug_id": task["drug_id"],
            "ln_ic50": sub["LN_IC50"].to_numpy(),
            "z_score": sub["Z_SCORE"].to_numpy(),
            "auc": sub["AUC"].to_numpy(),
            "max_conc": sub["MAX_CONC"].to_numpy(),
            "sensitive": (sub["Z_SCORE"] < z_threshold).astype(int).to_numpy(),
        }))
    return pd.concat(frames, ignore_index=True)


def load_model_list(raw_dir: Path) -> pd.DataFrame:
    models = pd.read_csv(raw_dir / MODEL_LIST_FILE, low_memory=False).set_index("model_id")
    for flag in ("mutation_data", "cnv_data", "rnaseq_data"):
        models[flag] = models[flag].astype(str).str.lower().eq("true")
    return models


def mutation_matrix(raw_dir: Path, model_ids: pd.Index) -> pd.DataFrame:
    mutations = pd.read_csv(raw_dir / MUTATIONS_FILE, low_memory=False)
    drivers = mutations[(mutations["cancer_driver"] == "t") & mutations["model_id"].isin(model_ids)]
    pairs = drivers.drop_duplicates(["model_id", "gene_symbol"])
    matrix = pd.crosstab(pairs["model_id"], pairs["gene_symbol"]).clip(upper=1)
    # Sequenced lines with no driver call are absent from the file but are all-zero, not missing.
    return matrix.reindex(model_ids, fill_value=0)


def copy_number_matrices(raw_dir: Path, model_ids: pd.Index) -> tuple[pd.DataFrame, pd.DataFrame]:
    cnv = pd.read_csv(raw_dir / CNV_FILE, low_memory=False)
    cnv = cnv[cnv["model_id"].isin(model_ids)]
    # ~80k (model, gene) pairs have both a Sanger and a Broad call; prefer Sanger.
    cnv = cnv.assign(_sanger_first=(cnv["source"] != "Sanger")).sort_values("_sanger_first")
    cnv = cnv.drop_duplicates(["model_id", "symbol"])
    matrices = []
    for category in ("Amplification", "Deletion"):
        hits = cnv[cnv["cn_category"] == category]
        matrix = pd.crosstab(hits["model_id"], hits["symbol"]).clip(upper=1)
        matrices.append(matrix.reindex(model_ids, fill_value=0))
    return matrices[0], matrices[1]


def expression_matrix(raw_dir: Path, model_ids: pd.Index) -> pd.DataFrame:
    """log2(TPM + 1), cell lines x genes, from the merged Sanger+Broad RSEM TPM matrix.

    The file is genes x models with three header rows (model_id, model_name,
    data_source) above the gene rows (gene_symbol, ensembl_gene_id, gene_id).
    Values are already log2(TPM + 1) per the Sanger DepMap expression docs.
    """

    with zipfile.ZipFile(raw_dir / RNASEQ_ZIP) as archive:
        with archive.open(RNASEQ_TPM_MEMBER) as handle:
            header = pd.read_csv(handle, header=None, nrows=3, index_col=0)
        model_columns = header.loc["model_id"].iloc[2:].to_numpy()
        wanted_ids = set(model_ids)
        wanted = [i for i, model_id in enumerate(model_columns) if model_id in wanted_ids]
        with archive.open(RNASEQ_TPM_MEMBER) as handle:
            tpm = pd.read_csv(handle, header=None, skiprows=4, usecols=[0] + [i + 3 for i in wanted], dtype=str)
    tpm = tpm.set_index(0)
    tpm.columns = model_columns[wanted]
    # 16 obscure genes (readthroughs, pseudogenes) pack two space-separated values
    # into one cell; drop them rather than guess which value is meant.
    multi_valued = tpm.apply(lambda column: column.str.contains(" ", na=False)).any(axis=1)
    tpm = tpm[~multi_valued].apply(pd.to_numeric)
    # Sanger and Broad quantified slightly different gene sets: keep genes measured in every line.
    tpm = tpm.dropna()
    # A handful of symbols map to more than one Ensembl ID; keep the more expressed row.
    tpm = tpm.loc[tpm.mean(axis=1).sort_values(ascending=False).index]
    tpm = tpm[~tpm.index.duplicated(keep="first")]
    if tpm.columns.duplicated().any():
        raise ValueError("a model_id appears in more than one TPM column")
    if tpm.to_numpy().max() > 25:
        raise ValueError("TPM values look untransformed (max > 25); expected log2(TPM + 1)")
    return tpm.T


def _select_by_frequency(matrix: pd.DataFrame, prefix: str, min_freq: float, forced: set[str]) -> pd.DataFrame:
    named = matrix.rename(columns=lambda gene: _feature_name(prefix, gene))
    frequency = named.mean()
    keep = set(frequency[frequency >= min_freq].index) | (forced & set(named.columns))
    return named[sorted(keep)]


def build_panel(args) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw_dir = Path(args.raw_dir)
    responses = load_responses(raw_dir, args.z_threshold)
    models = load_model_list(raw_dir)

    screened = pd.Index(sorted(responses["model_id"].unique()))
    has_omics = models.index[models["mutation_data"] & models["cnv_data"] & models["rnaseq_data"]]
    expression = expression_matrix(raw_dir, screened.intersection(has_omics))
    model_ids = screened.intersection(has_omics).intersection(expression.index)
    expression = expression.loc[model_ids]
    logger.info(f"{len(screened)} screened cell lines, {len(model_ids)} with mutation, CNV and RNA-seq data")

    forced = biomarker_features()
    mutations = _select_by_frequency(mutation_matrix(raw_dir, model_ids), "mut", args.min_mutation_freq, forced)
    amplifications, deletions = copy_number_matrices(raw_dir, model_ids)
    amplifications = _select_by_frequency(amplifications, "amp", args.min_amp_freq, forced)
    deletions = _select_by_frequency(deletions, "del", args.min_del_freq, forced)

    # Most variable expressed genes, judged across every panel cell line (no labels).
    expressed = expression.columns[expression.mean() >= args.min_mean_log_tpm]
    variable = expression[expressed].var().sort_values(ascending=False).index[: args.n_variable_genes]
    expression_named = expression.rename(columns=lambda gene: _feature_name("expr", gene))
    forced_expression = forced & set(expression_named.columns)
    expression_features = expression_named[
        sorted({_feature_name("expr", gene) for gene in variable} | forced_expression)
    ]

    tissue = models.loc[model_ids, "tissue"].fillna("Unknown")
    tissue_counts = tissue.value_counts()
    common_tissues = tissue_counts[tissue_counts >= args.min_tissue_lines].index
    tissues = pd.get_dummies(tissue.where(tissue.isin(common_tissues))).astype(int)
    tissues.columns = [_feature_name("tissue", name) for name in tissues.columns]
    msi = models.loc[model_ids, "msi_status"].eq("MSI").astype(int).rename("msi_high").to_frame()

    blocks = {
        "mutation": mutations,
        "amplification": amplifications,
        "deletion": deletions,
        "expression": expression_features,
        "tissue": tissues,
        "msi": msi,
    }
    features = pd.concat(blocks.values(), axis=1)
    features.index.name = "model_id"

    missing_forced = sorted(forced - set(features.columns))
    if missing_forced:
        logger.warning(f"answer-key features absent from the raw data: {missing_forced}")

    responses = responses[responses["model_id"].isin(model_ids)].reset_index(drop=True)
    summary = {
        "z_threshold": args.z_threshold,
        "n_cell_lines": len(model_ids),
        "n_features": features.shape[1],
        "block_sizes": {name: block.shape[1] for name, block in blocks.items()},
        "forced_answer_key_features": sorted(forced & set(features.columns)),
        "missing_answer_key_features": missing_forced,
        "settings": {key: value for key, value in vars(args).items() if key not in ("raw_dir", "out_dir")},
        "tasks": {name: _task_summary(name, responses, features) for name in TASKS},
    }
    return features, responses, summary


def _task_summary(task_name: str, responses: pd.DataFrame, features: pd.DataFrame) -> dict:
    """Class balance plus each biomarker member's univariate association with the label.

    A sanity check that labels and features line up (e.g. BRAF-mutant lines should
    be far more often dabrafenib-sensitive), not an input to any method.
    """

    task_responses = responses[responses["task"] == task_name].set_index("model_id")
    y = task_responses["sensitive"]
    X = features.loc[y.index]
    associations = {}
    for members in TASKS[task_name]["biomarkers"].values():
        for member in members:
            if member not in X.columns:
                continue
            values = X[member]
            if set(values.unique()) <= {0, 1}:
                associations[member] = {
                    "kind": "binary",
                    "n_with": int(values.sum()),
                    "sensitive_rate_with": float(y[values == 1].mean()) if values.sum() else None,
                    "sensitive_rate_without": float(y[values == 0].mean()),
                }
            else:
                associations[member] = {
                    "kind": "continuous",
                    "mean_sensitive": float(values[y == 1].mean()),
                    "mean_resistant": float(values[y == 0].mean()),
                    "spearman_with_z_score": float(values.corr(task_responses["z_score"], method="spearman")),
                }
    return {
        "drug_name": TASKS[task_name]["drug_name"],
        "drug_id": TASKS[task_name]["drug_id"],
        "n_cell_lines": int(len(y)),
        "n_sensitive": int(y.sum()),
        "sensitive_fraction": float(y.mean()),
        "biomarker_associations": associations,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--raw-dir", default=str(SCRIPT_DIR / "data" / "raw"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "data" / "processed"))
    parser.add_argument("--z-threshold", type=float, default=-1.0,
                        help="sensitive = GDSC2 Z_SCORE (per-drug standardized LN_IC50) below this (default: -1.0)")
    parser.add_argument("--n-variable-genes", type=int, default=150,
                        help="Most variable expressed genes to include (default: 150)")
    parser.add_argument("--min-mean-log-tpm", type=float, default=1.0,
                        help="A gene counts as expressed if its mean log2(TPM+1) is at least this (default: 1.0)")
    parser.add_argument("--min-mutation-freq", type=float, default=0.03)
    parser.add_argument("--min-amp-freq", type=float, default=0.05)
    parser.add_argument("--min-del-freq", type=float, default=0.03)
    parser.add_argument("--min-tissue-lines", type=int, default=20)
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    args = _parse_args()
    features, responses, summary = build_panel(args)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features.csv")
    responses.to_csv(out_dir / "responses.csv", index=False)
    (out_dir / "panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"{summary['n_cell_lines']} cell lines x {summary['n_features']} features: {summary['block_sizes']}")
    for name, task in summary["tasks"].items():
        logger.info(f"  {name:12s} n={task['n_cell_lines']} sensitive={task['n_sensitive']} "
                    f"({task['sensitive_fraction']:.1%})")
    logger.info(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
