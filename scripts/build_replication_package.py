#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

README_CONTENT = """# Replication Package

**Paper:** "How Much ML Code Similarity Is Framework-Dictated?
Measuring Accidental Clone Prevalence in PyTorch Projects"

**Author:** Joseph C. Sremack

**Venue:** Submitted to Empirical Software Engineering (EMSE)

**Date:** {date}

---

```
replication_package/
├── README.md                   ← This file
├── LICENSE                     ← MIT (code), CC-BY-4.0 (data)
├── requirements.txt            ← Python dependencies
│
├── 01_project_list/
│   ├── selected_projects.csv   ← Final project list with GitHub URLs + commit SHAs
│   ├── candidates.csv          ← Full candidate list with screening results
│   └── independence_matrix.csv ← Pairwise independence verification
│
├── 02_boilerplate_templates/
│   └── pytorch_patterns.json   ← Boilerplate patterns from PyTorch docs
│
├── 03_classifier/
│   ├── ast_classifier.py       ← AST-based boilerplate classifier
│   ├── filter_code.py          ← Code filtering script (remove Cat A+B)
│   └── README.md               ← Classifier usage instructions
│
├── 04_manifests/
│   ├── <project>_manifest.json ← Per-project classification (one per project)
│   └── ...
│
├── 05_validation/
│   ├── codebook.md             ← Classification codebook
│   ├── round1.csv              ← Intra-rater Round 1 classifications
│   ├── round2.csv              ← Intra-rater Round 2 classifications
│   └── kappa_computation.py    ← Cohen's kappa computation script
│
├── 06_similarity_results/
│   ├── all_comparisons.csv     ← All pairwise similarity + filtration effect
│   ├── full_standard.jplag     ← JPlag raw output: full code, standard threshold
│   ├── filtered_standard.jplag ← JPlag raw output: filtered code, standard threshold
│   └── ...                     ← Additional threshold runs
│
├── 07_analysis/
│   ├── sensitivity_analysis.py ← Sensitivity analysis script
│   ├── tables/                 ← All generated CSV tables
│   └── figures/                ← All generated figures (PNG + PDF)
│
├── 08_pregenerated/
│   ├── tables/                 ← Pre-generated tables for reviewers
│   └── figures/                ← Pre-generated figures for reviewers
│
└── 09_pipeline/
    ├── github_mining.py        ← Phase 1: candidate discovery
    ├── independence_check.py   ← Phase 1: independence verification
    ├── jplag_runner.py         ← Phase 3: JPlag wrapper
    └── run_pipeline.sh         ← One-shot pipeline runner
```

- Python 3.8+ (3.10+ recommended)
- Java 21+ (for JPlag)
- JPlag v6.1.0 (download from https://github.com/jplag/JPlag/releases/tag/v6.1.0)

Python dependencies:
```
pip install -r requirements.txt
```

All tables and figures are pre-generated in `08_pregenerated/`. Reviewers can inspect
these directly without running any code.

1. **Clone repositories** at the pinned commit SHAs listed in `01_project_list/selected_projects.csv`

2. **Run the classifier:**
   ```bash
   python 03_classifier/ast_classifier.py \\
       --repos path/to/repos \\
       --output 04_manifests/ \\
       --threshold 0.70
   ```

3. **Filter code:**
   ```bash
   python 03_classifier/filter_code.py \\
       --repos path/to/repos \\
       --manifests 04_manifests/ \\
       --output-full submissions_full/ \\
       --output-filtered submissions_filtered/
   ```

4. **Run JPlag:**
   ```bash
   python 09_pipeline/jplag_runner.py \\
       --jplag-jar jplag.jar \\
       --submissions-full submissions_full/ \\
       --submissions-filtered submissions_filtered/ \\
       --output 06_similarity_results/
   ```

5. **Run sensitivity analysis:**
   ```bash
   python 07_analysis/sensitivity_analysis.py
   ```

To reproduce the intra-rater reliability assessment:
1. See `05_validation/codebook.md` for classification definitions
2. Round 1 and Round 2 raw data are in `05_validation/`
3. Run `python 05_validation/kappa_computation.py` to compute Cohen's kappa

All repositories used in this study are public GitHub repositories. URLs and pinned
commit SHAs are provided in `01_project_list/selected_projects.csv`. No proprietary
data was used.

- **Code** (scripts, classifier): MIT License
- **Data** (manifests, results, validation): CC-BY-4.0
- **Paper text**: Copyright the author; not included in this package

If you use this replication package, please cite:

```bibtex
@article{{sremack2026mlclones,
  title   = {{How Much ML Code Similarity Is Framework-Dictated?
              Measuring Accidental Clone Prevalence in PyTorch Projects}},
  author  = {{Sremack, Joseph C.}},
  journal = {{Empirical Software Engineering}},
  year    = {{2026}},
  note    = {{Under review}}
}}
```

Joseph C. Sremack

Copyright (c) {year} Joseph C. Sremack

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

Creative Commons Attribution 4.0 International (CC-BY-4.0) (for data)

The data files in this package (manifests, results, validation data) are
licensed under CC-BY-4.0. See: https://creativecommons.org/licenses/by/4.0/
pandas>=1.5.0
matplotlib>=3.5.0
scipy>=1.9.0
seaborn>=0.12.0
requests>=2.28.0
tqdm>=4.64.0
numpy>=1.23.0
Compute Cohen's kappa for intra-rater reliability.

Usage:
    python kappa_computation.py --round1 round1.csv --round2 round2.csv
    Compute Cohen's kappa between two sets of labels.

    Args:
        round1_labels: list of category labels from Round 1
        round2_labels: list of category labels from Round 2
        categories: list of possible categories (default: auto-detect)

    Returns:
        kappa, observed_agreement, expected_agreement, confusion_matrix
    if kappa < 0.00:
        return "Poor"
    elif kappa <= 0.20:
        return "Slight"
    elif kappa <= 0.40:
        return "Fair"
    elif kappa <= 0.60:
        return "Moderate"
    elif kappa <= 0.80:
        return "Substantial"
    else:
        return "Almost perfect"

def main():
    parser = argparse.ArgumentParser(description="Compute Cohen\'s kappa")
    parser.add_argument("--round1", required=True, help="Round 1 CSV (must have \'category\' column)")
    parser.add_argument("--round2", required=True, help="Round 2 CSV (must have \'category\' column)")
    args = parser.parse_args()

    def load_labels(filepath):
        labels = []
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("category", row.get("classification", "")).strip().upper())
        return labels

    r1 = load_labels(args.round1)
    r2 = load_labels(args.round2)

    if len(r1) != len(r2):
        print(f"ERROR: Round 1 has {len(r1)} items, Round 2 has {len(r2)}")
        sys.exit(1)

    kappa, po, pe, matrix = compute_cohens_kappa(r1, r2)
    categories = sorted(set(r1) | set(r2))

    print("=" * 50)
    print("INTRA-RATER RELIABILITY RESULTS")
    print("=" * 50)
    print(f"Items classified: {len(r1)}")
    print(f"Categories:       {categories}")
    print(f"\\nObserved agreement (Po): {po:.4f} ({100*po:.1f}%)")
    print(f"Expected agreement (Pe): {pe:.4f} ({100*pe:.1f}%)")
    print(f"Cohen\'s kappa (kappa):     {kappa:.4f}")
    print(f"Interpretation:          {interpret_kappa(kappa)}")

    print(f"\\nConfusion Matrix (Round 1 = rows, Round 2 = columns):")
    header = "        " + "  ".join(f"{c:>5}" for c in categories)
    print(header)
    for c1 in categories:
        row = f"  {c1:>4}  " + "  ".join(f"{matrix[c1][c2]:>5}" for c2 in categories)
        print(row)

    print(f"\\nPer-category agreement:")
    for c in categories:
        total_r1 = sum(matrix[c][c2] for c2 in categories)
        if total_r1 > 0:
            agree = matrix[c][c] / total_r1
            print(f"  {c}: {matrix[c][c]}/{total_r1} = {100*agree:.1f}%")

    target_met = kappa >= 0.70
    verdict = "MET" if target_met else "NOT MET"
    print(f"\\nTarget kappa >= 0.70: {verdict}")
    if not target_met and kappa >= 0.60:
        print("  (Marginal — consider revising codebook boundary cases)")
    elif not target_met:
        print("  (Below acceptable — revise codebook and re-run)")

if __name__ == "__main__":
    main()
