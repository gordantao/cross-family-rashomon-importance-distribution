"""GDSC drug-sensitivity tasks and their literature biomarkers (the answer key).

Shared by prepare_gdsc_data.py (which needs every biomarker gene in the feature
panel) and run_gdsc_top40_comparison.py (which reports where each method ranks
them). Feature names follow prepare_gdsc_data.py's prefixes:

    expr_<GENE>   log2(TPM + 1) expression
    mut_<GENE>    1 if the cell line carries a driver mutation in GENE
    amp_<GENE>    1 if GENE is amplified
    del_<GENE>    1 if GENE is deleted
    tissue_<X>    1 if the cell line's tissue is X
    msi_high      1 if microsatellite-instability high

Each task's ``biomarkers`` maps a short label to a group of interchangeable
features: an amplified gene and its expression carry the same signal, so a
method gets credit for whichever member it ranks highest. That mirrors the
nonlinear simulation's equivalence recall for redundant drivers.

Picked so each task stresses a different kind of signal:
    dabrafenib   one mutation (BRAF), a positive control
    trametinib   OR-logic across three MAPK genes (BRAF | KRAS | NRAS)
    nutlin3a     a mutation (TP53, resistance) that p53-target expression substitutes for
    lapatinib    an amplification (ERBB2) that its own expression substitutes for
    talazoparib  an expression-only biomarker (SLFN11)
"""

from __future__ import annotations

# p53 transcriptional targets whose expression reads out wild-type TP53 activity
# (Jeay et al. 2015, eLife, "A distinct p53 target gene set predicts for
# response to the selective p53-HDM2 inhibitor NVP-CGM097").
P53_TARGET_GENES = ("MDM2", "CDKN1A", "ZMAT3", "RPS27L", "EDA2R", "SPATA18", "FDXR", "SESN1", "DDB2", "TNFRSF10B")

TASKS = {
    "dabrafenib": {
        "drug_id": 1373,
        "drug_name": "Dabrafenib",
        "target": "BRAF",
        "biomarkers": {
            "BRAF mutation": ["mut_BRAF"],
        },
    },
    "trametinib": {
        "drug_id": 1372,
        "drug_name": "Trametinib",
        "target": "MEK1/2",
        "biomarkers": {
            "BRAF mutation": ["mut_BRAF"],
            "KRAS mutation": ["mut_KRAS"],
            "NRAS mutation": ["mut_NRAS"],
        },
    },
    "nutlin3a": {
        "drug_id": 1047,
        "drug_name": "Nutlin-3a (-)",
        "target": "MDM2",
        "biomarkers": {
            "TP53 status": ["mut_TP53"],
            "p53 target expression": [f"expr_{gene}" for gene in P53_TARGET_GENES],
            "MDM2 amplification": ["amp_MDM2"],
        },
    },
    "lapatinib": {
        "drug_id": 1558,
        "drug_name": "Lapatinib",
        "target": "ERBB2, EGFR",
        "biomarkers": {
            # GRB7 sits on the ERBB2 amplicon, so its expression tracks ERBB2 copy number.
            "ERBB2 amplification": ["amp_ERBB2", "expr_ERBB2", "expr_GRB7"],
            "EGFR": ["expr_EGFR", "amp_EGFR", "mut_EGFR"],
        },
    },
    "talazoparib": {
        "drug_id": 1259,
        "drug_name": "Talazoparib",
        "target": "PARP1/2",
        "biomarkers": {
            "SLFN11 expression": ["expr_SLFN11"],
        },
    },
}


def biomarker_features() -> set[str]:
    """Every feature any task's answer key names. The shared panel always includes
    them, for every task, so each task's biomarkers compete with the others' as decoys."""

    return {
        feature
        for task in TASKS.values()
        for group in task["biomarkers"].values()
        for feature in group
    }
