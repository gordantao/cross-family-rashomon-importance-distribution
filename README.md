# Bioavailability Project

## Goal
Identify data-driven, meaningful molecular features—guided by **Lipinski’s Rule of Five**—to inform drug candidate screening and outline heuristic approaches for early-stage filtering.

## Data Sources & References

### **1. Kim et al. (2014)**
**Reference:**  
Kim, M.T., Sedykh, A., Chakravarti, S.K. *et al.* “Critical Evaluation of Human Oral Bioavailability for Pharmaceutical Drugs by Using Various Cheminformatics Approaches.” *Pharmaceutical Research* 31, 1002–1014 (2014).  
[https://doi.org/10.1007/s11095-013-1222-1](https://doi.org/10.1007/s11095-013-1222-1)

**Dataset Summary:**  
- Compiled from multiple public and private sources (Refs. 3,5,8,12–17).  
- Initially contained **~1,300 compounds**; curated down to **995 unique molecules**.  
- **Tools used:** CASE Ultra, ChemAxon Standardizer, and ChemAxon Structure Checker.  
- **Curation rules:**  
  - Removed duplicates, salts (neutralized), and mixtures (largest component retained).  
  - Excluded metal-organics and inorganics.  
  - Retained stereoisomer with **highest activity**.  
- **Experimental %F harmonization:**  
  - Preferred values from *Goodman & Gilman’s The Pharmacological Basis of Therapeutics*.  
  - When multiple %F values existed:  
    - Averaged if within ±10%.  
    - Selected best-defined experimental methodology otherwise.  
- **Classification:**  
  - **High:** %F ≥ 50 (n = 540)  
  - **Low:** %F < 50 (n = 455)  
  - Used sigmoid transformation to obtain **logK(%F)** for more balanced model distribution.

### **2. Falcón-Cano et al. (2020)**
**Reference:**  
Falcón-Cano, G., Molina, C., Cabrera-Pérez, M.Á. “ADME Prediction with KNIME: Development and Validation of a Publicly Available Workflow for the Prediction of Human Oral Bioavailability.” *J. Chem. Inf. Model.* 60 (6), 2660–2667 (2020).  
[https://doi.org/10.1021/jm901371v](https://doi.org/10.1021/jm901371v)

**Dataset Composition:**  
- Integrated **four curated data sources** from the last decade:  
  1. Tian et al. – 1013 molecules.  
  2. Varma et al. – 309 molecules.  
  3. Kim et al. – 995 molecules.  
  4. Dörwald – experimental bioavailability records.  
- **Consistency checks:**  
  - Manual correction of **CAS numbers** and **SMILES** via PubChem.  
  - Rectified %F discrepancies >5% (based on reliable literature).  
  - Averaged %F values differing <5%.  
- **Binary classification:**  
  - Cutoff at **F = 50%** for balanced classes.  
- **Curation workflow (three-stage pipeline):**  
  1. **Filtering:** Removed molecules with unusual valences, heavy atoms (non H,C,N,O,S,F,Cl,Br,P,B,I), MW > 1200 g/mol, and unconnected fragments.  
  2. **Standardization:** Neutralized zwitterions, aromatized structures, standardized charges, removed stereochemistry, added explicit hydrogens.  
  3. **Deduplication:** Used **InChI** identifiers; averaged values within ±5% or retained higher %F for stereoisomers.  

### **3. Varma et al. (2022)**
**Reference:**  
Varma, M. V. S., Obach, R. S., Rotter, C. *et al.* “Physicochemical Space for Optimum Oral Bioavailability: Contribution of Human Intestinal Absorption and First-Pass Elimination” *J. Med. Chem.* 53, 3 (2010).  
[https://doi.org/10.1186/s13321-021-00580-6](https://doi.org/10.1186/s13321-021-00580-6)

**Dataset Usage:**  
- Adopted the **HOB dataset** from Falcón-Cano et al. (2020).  
  - **Training set:** 1157 molecules.  
  - **Test set 1:** 290 molecules (20% random split).  
  - **Test set 2:** 141 molecules (includes additional 27 compounds + ChEMBL data).  
- **Data validation:**  
  - Corrected 3 mislabelled compounds via literature verification.  
  - Deduplicated using both **2D structure** and **molecular fingerprints**.  
- **Descriptor processing:**  
  - 3D structures were generated (via RDKit), but **3D descriptors were excluded** from training due to low variance or missing values.  
- **Diversity analysis:**  
  - Tanimoto similarity (test vs. training):  
    - Test 1 avg = **0.655**  
    - Test 2 avg = **0.612**  
  - Indicates moderate structural diversity between sets.

---

**File prepared by:** Gordan Tao  
**Updated:** October 2025  
**Purpose:** Research reference for interpretable ML on human oral bioavailability datasets.

## RID Module

The RID implementation now lives in the `rid` package rather than entirely inside `run_rashomon_falcon_cano.py`.
The main entry points mirror a sklearn-style workflow: instantiate an estimator, call `fit(X, y)`, then inspect rankings or raw metric results.

```python
from sklearn.svm import SVC

from rid import (
  CrossFamilyRashomonImportanceDistribution,
  LassoClassifier,
  RidgeClassifier,
)

estimator = CrossFamilyRashomonImportanceDistribution(
  model_configs={
    "Lasso": (LassoClassifier, {}),
    "Ridge": (RidgeClassifier, {}),
    "SVM": (SVC, {}),
  },
  epsilon=0.05,
  n_bootstraps=100,
  n_models_per_class=50,
  n_jobs=4,
)

estimator.fit(X, y)
top_features = estimator.rank_features("sub_mr")[:10]
```

`vi_metrics` and `performance_metrics` can be provided as strings or callables. Strings keep the built-in behavior (`"sub_mr"`, `"loco"`, `"coef"` and `"accuracy"`, `"f1"`, `"auprc"`), while callables let you inject custom feature-importance or model-performance functions directly.

```python
from sklearn.metrics import accuracy_score

from rid import compute_rid, performance_auprc, vi_loco

def hit_rate(y_true, y_pred):
  return (y_true == y_pred).mean()

hit_rate.metric_name = "hit_rate"

metric_results, perf_stats, _ = compute_rid(
  X,
  y,
  model_class=RidgeClassifier,
  vi_metrics=[vi_loco],
  performance_metrics=[accuracy_score, hit_rate, performance_auprc],
)
```

Each `model_configs` entry can also override the candidate search grid used to populate the Rashomon set. The override may be a tuple form like `(SVC, {}, {"C": [0.1, 1], "gamma": [0.001, 0.01]})` or a dict form:

```python
model_configs = {
  "SVM": {
    "model": SVC,
    "kwargs": {},
    "search_grid": {
      "C": [0.1, 1, 10],
      "gamma": [0.001, 0.01, 0.1],
    },
  },
}
```

`run_rashomon_falcon_cano.py` remains the reproducible CLI pipeline for the Falcon-Cano study, but it now delegates RID computation to the reusable package.
