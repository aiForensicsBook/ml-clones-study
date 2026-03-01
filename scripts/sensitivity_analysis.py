#!/usr/bin/env python3

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

CLASSIFICATION_THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90]
JPLAG_TOKEN_THRESHOLD = 12

DOMAIN_MAP = {
    "ANTsTorch": "CV",
    "BenchmarkTransferLearning": "CV",
    "Diffusion-Distillation-WL": "Generative",
    "Pytorch-Img-Classification-Trainer-V2": "CV",
    "TaxaDiffusion": "CV",
    "chatterbox-finetuning": "NLP",
    "cino": "NLP",
    "flucoma-torch": "Audio",
    "gflownet-peptide": "Other",
    "lane-detection-unet-ncnn": "CV",
    "midv-500-models": "CV",
    "pytorch-examples": "Tabular",
    "torchebm": "Generative",
    "tracknet-series-pytorch": "CV",
    "vall-e": "NLP",
}

def run_command(cmd, description, timeout=600):
    print(f"  {description}...")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        if result.returncode != 0:
            print(f"    ERROR (exit code {result.returncode})")
            if result.stderr:
                print(f"    {result.stderr[:300]}")
            return False, result.stdout, result.stderr
        return True, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT after {timeout}s")
        return False, "", "timeout"
    except FileNotFoundError as e:
        print(f"    NOT FOUND: {e}")
        return False, "", str(e)

def extract_jplag_scores(zippath):
    scores = {}
    if not os.path.exists(zippath):
        print(f"    Warning: {zippath} not found")
        return scores

    try:
        with zipfile.ZipFile(zippath, "r") as z:
            overview_file = None
            for n in z.namelist():
                if "overview" in n.lower() and n.endswith(".json"):
                    overview_file = n
                    break
            if not overview_file:
                for n in z.namelist():
                    if n.endswith(".json"):
                        overview_file = n
                        break
            if overview_file:
                with z.open(overview_file) as f:
                    data = json.load(f)
                comps = data.get("top_comparisons", data.get("topComparisons", []))
                if not comps:
                    for key in data:
                        if isinstance(data[key], list) and len(data[key]) > 0:
                            if isinstance(data[key][0], dict) and (
                                "similarity" in data[key][0]
                                or "avg_similarity" in data[key][0]
                            ):
                                comps = data[key]
                                break
                for comp in comps:
                    first = comp.get(
                        "first_submission", comp.get("firstSubmission", "")
                    )
                    second = comp.get(
                        "second_submission", comp.get("secondSubmission", "")
                    )
                    sim = comp.get(
                        "similarity",
                        comp.get(
                            "avg_similarity",
                            comp.get("similarities", {}).get("AVG", 0),
                        ),
                    )
                    if isinstance(sim, dict):
                        sim = sim.get("AVG", 0)
                    first = first.split("/")[-1] if "/" in first else first
                    second = second.split("/")[-1] if "/" in second else second
                    pair = tuple(sorted([first, second]))
                    scores[pair] = float(sim)
    except Exception as e:
        print(f"    Error reading {zippath}: {e}")
    return scores

def main():
    parser = argparse.ArgumentParser(
        description="Sensitivity analysis: vary classifier threshold"
    )
    parser.add_argument(
        "--jplag-jar", default="jplag.jar", help="Path to jplag.jar"
    )
    parser.add_argument(
        "--repos", default="data/repos", help="Directory containing project repos"
    )
    parser.add_argument(
        "--scripts", default="scripts", help="Directory containing study scripts"
    )
    parser.add_argument(
        "--output", default="results/sensitivity", help="Output directory"
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated thresholds (default: 0.50,0.60,0.70,0.80,0.90)",
    )
    args = parser.parse_args()

    thresholds = CLASSIFICATION_THRESHOLDS
    if args.thresholds:
        thresholds = [float(t) for t in args.thresholds.split(",")]

    os.makedirs(args.output, exist_ok=True)

    classifier_script = os.path.join(args.scripts, "ast_classifier.py")
    filter_script = os.path.join(args.scripts, "filter_code.py")

    if not os.path.exists(classifier_script):
        print(f"ERROR: Cannot find {classifier_script}")
        sys.exit(1)
    if not os.path.exists(filter_script):
        print(f"ERROR: Cannot find {filter_script}")
        sys.exit(1)
    if not os.path.exists(args.jplag_jar):
        print(f"ERROR: Cannot find {args.jplag_jar}")
        sys.exit(1)

    projects = sorted([
        d for d in os.listdir(args.repos)
        if os.path.isdir(os.path.join(args.repos, d))
        and not d.startswith(".")
    ])

    print("=" * 70)
    print("SENSITIVITY ANALYSIS: CLASSIFIER THRESHOLD")
    print("=" * 70)
    print(f"Projects:    {len(projects)}")
    print(f"Thresholds:  {thresholds}")
    print(f"JPlag t:     {JPLAG_TOKEN_THRESHOLD} (standard)")
    print(f"Output:      {args.output}")
    print()

    summary_rows = []

    for threshold in thresholds:
        thresh_label = f"t{int(threshold * 100)}"
        print(f"\n{'='*70}")
        print(f"THRESHOLD: {threshold} ({thresh_label})")
        print(f"{'='*70}")

        manifests_dir = os.path.join(args.output, f"manifests_{thresh_label}")
        subs_full_dir = os.path.join(args.output, f"subs_full_{thresh_label}")
        subs_filt_dir = os.path.join(args.output, f"subs_filt_{thresh_label}")

        os.makedirs(manifests_dir, exist_ok=True)

        print(f"\n  Step 1: Classifying at threshold={threshold}")
        ok, stdout, _ = run_command(
            [
                sys.executable, classifier_script,
                "--repos", args.repos,
                "--output", manifests_dir,
                "--threshold", str(threshold),
                "--quiet",
            ],
            f"Classifying {len(projects)} projects at threshold={threshold}",
        )
        if not ok:
            print(f"  SKIPPING threshold {threshold} due to classifier error")
            continue

        total_a, total_b, total_c = 0, 0, 0
        total_loc_a, total_loc_b, total_loc_c = 0, 0, 0
        for proj in projects:
            mf = os.path.join(manifests_dir, f"{proj}_manifest.json")
            if os.path.exists(mf):
                with open(mf) as f:
                    m = json.load(f)
                s = m.get("summary", {})
                total_a += s.get("category_A", 0)
                total_b += s.get("category_B", 0)
                total_c += s.get("category_C", 0)
                total_loc_a += s.get("loc_A", 0)
                total_loc_b += s.get("loc_B", 0)
                total_loc_c += s.get("loc_C", 0)

        total_units = total_a + total_b + total_c
        total_loc = total_loc_a + total_loc_b + total_loc_c
        print(f"    Units: A={total_a}, B={total_b}, C={total_c} (total={total_units})")
        print(f"    LOC:   A={total_loc_a}, B={total_loc_b}, C={total_loc_c} (total={total_loc})")

        print(f"\n  Step 2: Filtering code")
        for d in [subs_full_dir, subs_filt_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)

        ok, stdout, _ = run_command(
            [
                sys.executable, filter_script,
                "--repos", args.repos,
                "--manifests", manifests_dir,
                "--output-full", subs_full_dir,
                "--output-filtered", subs_filt_dir,
            ],
            f"Filtering with threshold={threshold}",
        )
        if not ok:
            print(f"  SKIPPING threshold {threshold} due to filter error")
            continue

        full_jplag = os.path.join(args.output, f"full_{thresh_label}.jplag.zip")
        filt_jplag = os.path.join(args.output, f"filt_{thresh_label}.jplag.zip")

        print(f"\n  Step 3a: JPlag on FULL code")
        ok_full, _, _ = run_command(
            [
                "java", "-jar", args.jplag_jar,
                "-l", "python3",
                "-t", str(JPLAG_TOKEN_THRESHOLD),
                "-r", full_jplag,
                "--mode", "RUN",
                subs_full_dir,
            ],
            "JPlag full-code comparison",
            timeout=300,
        )

        print(f"\n  Step 3b: JPlag on FILTERED code")
        ok_filt, _, _ = run_command(
            [
                "java", "-jar", args.jplag_jar,
                "-l", "python3",
                "-t", str(JPLAG_TOKEN_THRESHOLD),
                "-r", filt_jplag,
                "--mode", "RUN",
                subs_filt_dir,
            ],
            "JPlag filtered-code comparison",
            timeout=300,
        )

        if not ok_full or not ok_filt:
            print(f"  SKIPPING threshold {threshold} due to JPlag error")
            continue

        print(f"\n  Step 4: Extracting results")
        full_scores = extract_jplag_scores(full_jplag)
        filt_scores = extract_jplag_scores(filt_jplag)
        print(f"    Full pairs: {len(full_scores)}, Filtered pairs: {len(filt_scores)}")

        all_pairs = set(full_scores.keys()) | set(filt_scores.keys())
        effects = []
        pair_rows = []
        for pair in sorted(all_pairs):
            full_sim = full_scores.get(pair, 0)
            filt_sim = filt_scores.get(pair, 0)
            effect = full_sim - filt_sim
            effects.append(effect)

            da = DOMAIN_MAP.get(pair[0], "Unknown")
            db = DOMAIN_MAP.get(pair[1], "Unknown")
            pair_rows.append({
                "project_a": pair[0],
                "project_b": pair[1],
                "domain_a": da,
                "domain_b": db,
                "pair_type": "within-domain" if da == db else "cross-domain",
                "full_similarity": round(full_sim, 4),
                "filtered_similarity": round(filt_sim, 4),
                "filtration_effect": round(effect, 4),
            })

        csv_path = os.path.join(args.output, f"pairs_{thresh_label}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pair_rows[0].keys() if pair_rows else [])
            writer.writeheader()
            writer.writerows(pair_rows)

        full_vals = [full_scores.get(p, 0) for p in all_pairs]
        filt_vals = [filt_scores.get(p, 0) for p in all_pairs]
        n = len(effects)
        if n > 0:
            mean_full = sum(full_vals) / n
            mean_filt = sum(filt_vals) / n
            mean_effect = sum(effects) / n
            pos_count = sum(1 for e in effects if e > 0)
            pct_pos = 100 * pos_count / n

            effects_sorted = sorted(effects)
            median_effect = effects_sorted[n // 2]

            removed_pct = (
                100 * (total_loc_a + total_loc_b) / max(total_loc, 1)
            )
        else:
            mean_full = mean_filt = mean_effect = median_effect = 0
            pos_count = 0
            pct_pos = 0
            removed_pct = 0

        row = {
            "threshold": threshold,
            "units_A": total_a,
            "units_B": total_b,
            "units_C": total_c,
            "loc_A": total_loc_a,
            "loc_B": total_loc_b,
            "loc_C": total_loc_c,
            "pct_removed": round(removed_pct, 1),
            "n_pairs": n,
            "mean_full_sim": round(mean_full, 4),
            "mean_filt_sim": round(mean_filt, 4),
            "mean_effect": round(mean_effect, 4),
            "median_effect": round(median_effect, 4),
            "pos_effect_count": pos_count,
            "pct_positive": round(pct_pos, 1),
        }
        summary_rows.append(row)

        print(f"\n  Results for threshold={threshold}:")
        print(f"    Code removed:          {removed_pct:.1f}%")
        print(f"    Mean full similarity:  {mean_full:.4f} ({mean_full*100:.2f}%)")
        print(f"    Mean filt similarity:  {mean_filt:.4f} ({mean_filt*100:.2f}%)")
        print(f"    Mean filtration effect: {mean_effect:.4f} ({mean_effect*100:.2f}pp)")
        print(f"    Positive effect:       {pos_count}/{n} ({pct_pos:.1f}%)")

    if summary_rows:
        summary_path = os.path.join(args.output, "sensitivity_summary.csv")
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n\n{'='*70}")
        print("SENSITIVITY ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"{'Thresh':>7} {'%Removed':>9} {'MeanFull':>9} {'MeanFilt':>9} "
              f"{'MeanEff':>9} {'MedEff':>9} {'%Pos':>6}")
        print("-" * 65)
        for r in summary_rows:
            print(f"  {r['threshold']:>5.2f} {r['pct_removed']:>8.1f}% "
                  f"{r['mean_full_sim']:>9.4f} {r['mean_filt_sim']:>9.4f} "
                  f"{r['mean_effect']:>9.4f} {r['median_effect']:>9.4f} "
                  f"{r['pct_positive']:>5.1f}%")

        print(f"\nSummary saved to: {summary_path}")
        print(f"Per-threshold CSVs in: {args.output}/")
    else:
        print("\nNo results collected. Check errors above.")

    print(f"\n{'='*70}")
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
