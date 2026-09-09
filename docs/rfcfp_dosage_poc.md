# rFCFP: dosage-aware fingerprints as a Rashomon-set POC

Source: Nature Communications 2024, DOI [10.1038/s41467-024-53457-1](https://www.nature.com/articles/s41467-024-53457-1)
("PRnet: Predicting transcriptional responses to novel chemical perturbations using
a deep generative model for drug discovery"). Full text (open): [PMC11513139](https://pmc.ncbi.nlm.nih.gov/articles/PMC11513139/).

## What rFCFP actually is

Standard FCFP (Functional-Class Fingerprint) is a fixed-length circular molecular
fingerprint computed from a SMILES string alone -- the same molecule always produces
the same fingerprint, regardless of how much of it is administered. rFCFP ("rescaled
FCFP") does not change the fingerprint algorithm at all. It rescales the existing
fingerprint by a scalar function of dose, then sums across compounds when a
perturbation involves more than one:

```
rFCFP_i = sum_j  phi(d_i,j) * H(s_i,j)
phi(d)  = log10(d + 1)
```

- `H(s)` is the standard FCFP4 (radius=2, 1024-dim) bit vector for compound `j`'s SMILES `s`.
- `d_i,j` is compound `j`'s dose within perturbation `i`.
- `phi` is a log-dose scalar (chosen so `phi(0) = 0`, unmeasured/zero dose contributes nothing).
- `rFCFP_i` sums the dose-scaled vectors of every compound present in perturbation `i`
  (a single compound at one dose, or a combination of several).

The entire mechanism is a scalar multiply-and-sum on top of an off-the-shelf
fingerprint. The payoff: a fixed-length input vector can now represent "this compound
(or combination) at this dose," which a fixed structural fingerprint alone cannot --
enabling a downstream model to learn dose-response and compound-combination effects
using the same descriptor-based ML machinery normally reserved for pure structure.

## Datasets used in the paper

- **LINCS L1000**: 883,269 profiles, 82 cell lines, 175,549 compounds (bulk transcriptomics).
- **sci-Plex3** (Srivatsan et al., *Science* 2020): 290,888 single-cell profiles,
  3 cancer cell lines, 188 compounds, 4 doses each (10 nM / 100 nM / 1 uM / 10 uM).
  GEO accession: **GSE139944**.

## Why neither of our current datasets works as-is

Checked directly: neither `experiments/falcon_cano/falcon_cano_featured.csv` nor
`experiments/staellert/data/staellert_et_al/control_manifold_allfeatures.csv` has a
dose, concentration, or stimulus-strength column. rFCFP has nothing to rescale by
without one.

## Proposed POC: sci-Plex3 (GSE139944)

Chosen over full LINCS L1000 for tractability -- a few thousand conditions
(188 compounds x 4 doses x 3 cell lines) rather than L1000's six-figure compound
count -- and because it's the paper's own validated benchmark, not an untested
extrapolation.

**Data shape**: one row per (compound, dose, cell line) condition.
**Features (X)**: the 1024 rFCFP dimensions, i.e. `log10(dose + 1) * FCFP4(SMILES)`
per condition -- so the same compound produces four different rows (one per dose),
differing only in the scalar dose weighting, rather than one static row per compound.
**Target (y)**: a binarized transcriptional-response call per condition (e.g. "strong
responder" vs. not, by a DE-gene-count or expression-shift-distance threshold) --
reduced to a form compatible with `compute_rid`'s `predict_proba`-based classifier
convention.

**Wiring into this repo**: no changes needed to `rid/core.py`. `compute_rid` and
`compute_rid_cross_family` already bootstrap-resample rows, fit each family's
candidate pool, and compute per-feature `sub_mr` / `loco` / `coef` importance across
the surviving Rashomon set -- an rFCFP dimension is just another column, exactly like
an RDKit descriptor in Falcon-Cano. The only new work is in dataset construction
(building `X`/`y` from sci-Plex3 + rFCFP), not in the RID machinery itself.

**The actually novel angle**: re-run RID separately within low/mid/high dose strata
(or bin by `phi(d)`) and compare each fingerprint bit's importance distribution
across strata. This asks a question RID is well-suited to answer and that a single
pooled-dose run cannot: is a given bit's importance itself dose-dependent (irrelevant
at low dose, load-bearing at high dose, or vice versa)? That's a direct, low-effort
extension of the existing per-feature VI machinery -- stratify the input rows before
calling `compute_rid`, nothing more.

## Open questions before this becomes a real experiment (not addressed here)

- Exact DE / response-shift definition and binarization threshold for `y`.
- SMILES recovery for sci-Plex3's 188 compounds (via PubChem/DrugBank lookup on
  compound name or CAS ID -- not confirmed to already be bundled with GSE139944).
- Whether 1024-dim rFCFP is itself a reasonable RID feature space size, or whether a
  reduced/curated subset (paralleling Falcon-Cano's correlation-filtering step) is
  needed first.
