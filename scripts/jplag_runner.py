#!/usr/bin/env python3

import argparse
import csv
import json
import os
import subprocess
import sys
import zipfile
from itertools import combinations
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("Missing: pip install pandas numpy")
    sys.exit(1)

def run_jplag(jplag_jar: str, submissions_dir: str, output_file: str,
              min_tokens: int = 12, similarity_threshold: float = 0.0,
              language: str = 'python3') -> bool:
    cmd = [
        'java', '-jar', jplag_jar,
        '-l', language,
        '-t', str(min_tokens),
        '-m', str(similarity_threshold),
        '-r', output_file,
        '--mode', 'RUN',
        submissions_dir,
    ]

    print(f"  Running: java -jar jplag.jar -l {language} -t {min_tokens} "
          f"-m {similarity_threshold} -r {output_file} {submissions_dir}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"  ERROR: JPlag exited with code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print(f"  ERROR: JPlag timed out after 300s")
        return False
    except FileNotFoundError:
        print(f"  ERROR: Java not found. Install JDK 21+.")
        return False

def extract_results(jplag_output: str) -> list:
    if not os.path.exists(jplag_output):
        print(f"  Warning: {jplag_output} not found")
        return []

    try:
        with zipfile.ZipFile(jplag_output, 'r') as z:
            for name in z.namelist():
                if name.endswith('overview.json'):
                    with z.open(name) as f:
                        data = json.load(f)
                    break
            else:
                print(f"  Warning: No overview.json in {jplag_output}")
                return []
    except (zipfile.BadZipFile, json.JSONDecodeError) as e:
        print(f"  Warning: Could not parse {jplag_output}: {e}")
        return []

    comparisons = []

    for comp in data.get('topComparisons', []):
        comparisons.append({
            'first': comp.get('firstSubmission', ''),
            'second': comp.get('secondSubmission', ''),
            'similarity': comp.get('similarity', 0.0),
            'matched_tokens': comp.get('matchedTokenNumber', 0),
        })

    if not comparisons:
        for comp in data.get('comparisons', []):
            comparisons.append({
                'first': comp.get('firstSubmission', comp.get('first', '')),
                'second': comp.get('secondSubmission', comp.get('second', '')),
                'similarity': comp.get('similarity', comp.get('maxSimilarity', 0.0)),
                'matched_tokens': comp.get('matchedTokenNumber', 0),
            })

    return comparisons

def compute_filtration_effect(full_results: list, filtered_results: list) -> pd.DataFrame:
    filtered_lookup = {}
    for comp in filtered_results:
        key = tuple(sorted([comp['first'], comp['second']]))
        filtered_lookup[key] = comp['similarity']

    rows = []
    for comp in full_results:
        key = tuple(sorted([comp['first'], comp['second']]))
        sim_full = comp['similarity']
        sim_filtered = filtered_lookup.get(key, 0.0)

        rows.append({
            'project_a': key[0],
            'project_b': key[1],
            'similarity_full': round(sim_full, 4),
            'similarity_filtered': round(sim_filtered, 4),
            'filtration_effect': round(sim_full - sim_filtered, 4),
            'pct_reduction': round(
                100 * (sim_full - sim_filtered) / max(sim_full, 0.0001), 1
            ),
        })

    return pd.DataFrame(rows)

TOKEN_THRESHOLDS = {
    'sensitive': 5,
    'standard': 12,
    'strict': 25,
}

def main():
    parser = argparse.ArgumentParser(
        description="JPlag pipeline for ML Clones Study"
    )
    parser.add_argument('--jplag-jar', required=True,
                        help='Path to jplag.jar')
    parser.add_argument('--submissions-full', required=True,
                        help='Directory with full (unfiltered) submissions')
    parser.add_argument('--submissions-filtered', required=True,
                        help='Directory with filtered (Cat C only) submissions')
    parser.add_argument('--output', default='results/similarity',
                        help='Output directory for results')
    parser.add_argument('--thresholds', default='sensitive,standard,strict',
                        help='Comma-separated threshold labels to run')
    parser.add_argument('--domain-map', default=None,
                        help='Optional JSON file mapping project names to domains')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    thresholds = args.thresholds.split(',')

    domain_map = {}
    if args.domain_map and os.path.exists(args.domain_map):
        with open(args.domain_map) as f:
            domain_map = json.load(f)

    print("=" * 70)
    print("JPLAG SIMILARITY MEASUREMENT PIPELINE")
    print("=" * 70)
    print(f"Full submissions:     {args.submissions_full}")
    print(f"Filtered submissions: {args.submissions_filtered}")
    print(f"Thresholds:           {thresholds}")
    print()

    all_comparisons = []

    for label in thresholds:
        if label not in TOKEN_THRESHOLDS:
            print(f"  Unknown threshold label: {label}")
            continue

        min_tokens = TOKEN_THRESHOLDS[label]
        print(f"\n--- Threshold: {label} (min_tokens={min_tokens}) ---")

        full_output = os.path.join(args.output, f'full_{label}.jplag')
        print(f"\n  Comparison A: Full code")
        success = run_jplag(
            args.jplag_jar, args.submissions_full, full_output,
            min_tokens=min_tokens
        )

        if not success:
            print(f"  Skipping threshold {label} due to JPlag error")
            continue

        full_results = extract_results(full_output)
        print(f"  Extracted {len(full_results)} pairwise comparisons")

        filtered_output = os.path.join(args.output, f'filtered_{label}.jplag')
        print(f"\n  Comparison B: Filtered code (Cat C only)")
        success = run_jplag(
            args.jplag_jar, args.submissions_filtered, filtered_output,
            min_tokens=min_tokens
        )

        if not success:
            print(f"  Skipping filtered comparison for {label}")
            continue

        filtered_results = extract_results(filtered_output)
        print(f"  Extracted {len(filtered_results)} pairwise comparisons")

        print(f"\n  Computing filtration effect...")
        effect_df = compute_filtration_effect(full_results, filtered_results)

        if not effect_df.empty:
            effect_df['threshold'] = label
            effect_df['min_tokens'] = min_tokens

            if domain_map:
                effect_df['domain_a'] = effect_df['project_a'].map(domain_map).fillna('Unknown')
                effect_df['domain_b'] = effect_df['project_b'].map(domain_map).fillna('Unknown')
                effect_df['same_domain'] = effect_df['domain_a'] == effect_df['domain_b']

            all_comparisons.append(effect_df)

            print(f"\n  {'Pair':<40} {'Full':>7} {'Filt':>7} {'Effect':>8} {'%Red':>6}")
            print(f"  {'-'*70}")
            for _, row in effect_df.iterrows():
                pair = f"{row['project_a']} vs {row['project_b']}"
                print(f"  {pair:<40} {row['similarity_full']:>7.4f} "
                      f"{row['similarity_filtered']:>7.4f} "
                      f"{row['filtration_effect']:>8.4f} "
                      f"{row['pct_reduction']:>5.1f}%")

            csv_path = os.path.join(args.output, f'filtration_{label}.csv')
            effect_df.to_csv(csv_path, index=False)
            print(f"\n  Saved: {csv_path}")

    if all_comparisons:
        combined = pd.concat(all_comparisons, ignore_index=True)
        combined_path = os.path.join(args.output, 'all_comparisons.csv')
        combined.to_csv(combined_path, index=False)

        print(f"\n\n{'=' * 70}")
        print(f"AGGREGATE RESULTS")
        print(f"{'=' * 70}")

        for label in thresholds:
            subset = combined[combined['threshold'] == label]
            if subset.empty:
                continue
            print(f"\n  Threshold: {label}")
            print(f"    Mean similarity (full):     {subset['similarity_full'].mean():.4f}")
            print(f"    Mean similarity (filtered): {subset['similarity_filtered'].mean():.4f}")
            print(f"    Mean filtration effect:      {subset['filtration_effect'].mean():.4f}")
            print(f"    Mean % reduction:            {subset['pct_reduction'].mean():.1f}%")

            if 'same_domain' in subset.columns:
                within = subset[subset['same_domain'] == True]
                across = subset[subset['same_domain'] == False]
                if not within.empty:
                    print(f"    Within-domain mean sim:     {within['similarity_full'].mean():.4f}")
                if not across.empty:
                    print(f"    Cross-domain mean sim:      {across['similarity_full'].mean():.4f}")

        print(f"\n  Combined results: {combined_path}")
    else:
        print("\nNo results collected. Check JPlag output for errors.")

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nTo view JPlag reports interactively:")
    print(f"  java -jar {args.jplag_jar} --mode VIEW results/similarity/full_standard.jplag")

if __name__ == '__main__':
    main()
