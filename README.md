# ML Clones Study

**How Much ML Code Similarity Is Framework-Dictated? Measuring Accidental Clone Prevalence in PyTorch Projects**

Joseph C. Sremack

## Overview

This repository contains the replication package for an empirical study measuring how much code similarity between independent PyTorch projects is driven by framework conventions rather than developer choices.

The study analyzes 15 verified-independent PyTorch projects across five ML domains, classifies every function and class as framework boilerplate (A), API protocol (B), or custom implementation (C), then measures pairwise similarity before and after filtering framework-dictated code.

## Key Findings

- 52.7% of code units (55.7% by LOC) are framework-dictated (Categories A+B)
- Mean pairwise similarity between independent projects: 5.34% (5x above the no-framework baseline)
- Filtering framework code reduces similarity by 2.11 percentage points (39.5% relative reduction)
- Results are stable across five classification thresholds (effect range: 0.07pp)

## Repository Structure

```
scripts/
  ast_classifier.py          AST-based code classifier (Phase 2)
  filter_code.py             Code filtering for JPlag submissions (Phase 2)
  github_mining.py           GitHub candidate search (Phase 1)
  independence_check.py      Pairwise independence verification (Phase 1)
  jplag_runner.py            JPlag execution and result extraction (Phase 3)
  sensitivity_analysis.py    Full pipeline at multiple thresholds (Phase 4)
  compute_kappa.py           Cohen's kappa for intra-rater validation
  generate_round2.py         Round 2 validation workbook generator
  clone_candidates.bat       Windows batch script to clone all study repos
  build_replication_package.py  Package builder for Zenodo archival

data/
  manifests/                 JSON classification manifests (one per project)

results/
  similarity/
    pairwise_results.csv     All 105 pairwise comparisons x 3 thresholds
  sensitivity/
    sensitivity_summary.csv  Metrics at each classification threshold

paper/
  ml_clones_study.tex        LaTeX source
  ml_clones_study.pdf        Compiled paper
  figures/                   Generated figures (PDF + PNG)
```

## Reproducing the Study

### Prerequisites

- Python 3.9+
- Java 21+ (for JPlag)
- JPlag v5.1.0 jar ([download](https://github.com/jplag/JPlag/releases/tag/v5.1.0))
- Python packages: `openpyxl`
- GitHub personal access token (for mining)

### Step 1: Clone Study Projects

```bash
scripts/clone_candidates.bat
```

### Step 2: Classify Code

```bash
python scripts/ast_classifier.py --repos data/repos --output data/manifests --threshold 0.70
```

### Step 3: Filter and Prepare JPlag Submissions

```bash
python scripts/filter_code.py --repos data/repos --manifests data/manifests --output-full data/submissions_full --output-filtered data/submissions_filtered
```

### Step 4: Run JPlag

```bash
python scripts/jplag_runner.py --jplag-jar jplag.jar --submissions-full data/submissions_full --submissions-filtered data/submissions_filtered --output results/similarity
```

### Step 5: Sensitivity Analysis

```bash
python scripts/sensitivity_analysis.py --jplag-jar jplag.jar
```

### Step 6: Intra-Rater Validation

After completing Round 1 and Round 2 classification workbooks (14-day gap):

```bash
python scripts/compute_kappa.py --round1 results/intra_rater_round1.xlsx --round2 results/intra_rater_round2.xlsx
```

## Study Projects

| ID | Domain | LOC | Description |
|----|--------|-----|-------------|
| ANTsTorch | CV | 9,800 | Medical image registration |
| BenchmarkTL | CV | 17,111 | Transfer learning benchmarks |
| Diff-Distill-WL | Gen. | 4,206 | Diffusion distillation |
| PyImgClassV2 | CV | 1,505 | Image classification trainer |
| TaxaDiffusion | Gen. | 2,207 | Taxonomic diffusion models |
| Chatterbox-FT | NLP | 8,634 | Speech model fine-tuning |
| CINO | NLP | 497 | Cross-lingual NLI |
| FluCoMa-Torch | Audio | 1,520 | Audio descriptor learning |
| GFlowNet-Pep | Other | 11,123 | GFlowNet for peptide design |
| LaneDet-UNet | CV | 1,152 | Lane detection with U-Net |
| MIDV-500 | CV | 590 | Document recognition models |
| PyTorch-Ex | Other | 2,618 | PyTorch example collection |
| TorchEBM | Other | 23,742 | Energy-based models library |
| TrackNet | CV | 1,908 | Ball tracking networks |
| VALL-E | Audio | 1,419 | Neural codec language model |

## License

MIT

## Citation

If you use this work, please cite:

```bibtex
@article{sremack2026mlclones,
  title={How Much ML Code Similarity Is Framework-Dictated? Measuring Accidental Clone Prevalence in PyTorch Projects},
  author={Sremack, Joseph C.},
  journal={Empirical Software Engineering},
  year={2026}
}
```
