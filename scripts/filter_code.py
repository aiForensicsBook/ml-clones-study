#!/usr/bin/env python3

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

@dataclass
class CodeRegion:
    line_start: int
    line_end: int
    category: str
    name: str

def load_manifest(manifest_path: str) -> dict:
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_regions_to_remove(manifest: dict, remove_categories: Set[str] = {'A', 'B'}) -> Dict[str, List[CodeRegion]]:
    regions_by_file = {}
    for c in manifest['classifications']:
        if c['category'] in remove_categories:
            filename = c['file']
            if filename not in regions_by_file:
                regions_by_file[filename] = []
            regions_by_file[filename].append(CodeRegion(
                line_start=c['line_start'],
                line_end=c['line_end'],
                category=c['category'],
                name=c['name'],
            ))

    for filename in regions_by_file:
        regions_by_file[filename].sort(key=lambda r: r.line_start)

    return regions_by_file

def filter_file(source_path: str, regions: List[CodeRegion],
                keep_stubs: bool = True) -> str:
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    lines_to_remove = set()
    stubs_to_insert = {}

    for region in regions:
        for line_num in range(region.line_start, region.line_end + 1):
            lines_to_remove.add(line_num)

        if keep_stubs:
            if region.line_start <= len(lines):
                original_line = lines[region.line_start - 1]
                indent = len(original_line) - len(original_line.lstrip())
                indent_str = ' ' * indent

                sig_line = original_line.rstrip()
                if sig_line.lstrip().startswith(('def ', 'class ')):
                    if sig_line.lstrip().startswith('class '):
                        stub = f"{sig_line}\n{indent_str}    pass  # [FILTERED: {region.category} — {region.name}]\n"
                    else:
                        stub = f"{sig_line}\n{indent_str}    pass  # [FILTERED: {region.category} — {region.name}]\n"
                    stubs_to_insert[region.line_start] = stub

    output_lines = []
    i = 1
    while i <= len(lines):
        if i in stubs_to_insert:
            output_lines.append(stubs_to_insert[i])
            while i in lines_to_remove and i <= len(lines):
                i += 1
        elif i in lines_to_remove:
            i += 1
        else:
            output_lines.append(lines[i - 1])
            i += 1

    return ''.join(output_lines)

def filter_file_no_stubs(source_path: str, regions: List[CodeRegion]) -> str:
    with open(source_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    lines_to_remove = set()
    for region in regions:
        for line_num in range(region.line_start, region.line_end + 1):
            lines_to_remove.add(line_num)

    output_lines = []
    for i, line in enumerate(lines, 1):
        if i not in lines_to_remove:
            output_lines.append(line)

    return ''.join(output_lines)

def prepare_full_submissions(repos_dir: str, output_dir: str,
                             project_names: List[str]):
    os.makedirs(output_dir, exist_ok=True)

    for proj_name in project_names:
        src_dir = os.path.join(repos_dir, proj_name)
        dst_dir = os.path.join(output_dir, proj_name)

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

        for py_file in Path(src_dir).rglob('*.py'):
            rel = py_file.relative_to(src_dir)
            flat_name = str(rel).replace(os.sep, '_')
            shutil.copy2(py_file, os.path.join(dst_dir, flat_name))

def prepare_filtered_submissions(repos_dir: str, manifests_dir: str,
                                  output_dir: str, project_names: List[str],
                                  remove_categories: Set[str] = {'A', 'B'},
                                  keep_stubs: bool = True):
    os.makedirs(output_dir, exist_ok=True)

    stats = {}
    for proj_name in project_names:
        src_dir = os.path.join(repos_dir, proj_name)
        dst_dir = os.path.join(output_dir, proj_name)
        manifest_path = os.path.join(manifests_dir, f'{proj_name}_manifest.json')

        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.makedirs(dst_dir)

        if not os.path.exists(manifest_path):
            print(f"  Warning: No manifest for {proj_name}, copying unfiltered")
            prepare_full_submissions(repos_dir, output_dir, [proj_name])
            continue

        manifest = load_manifest(manifest_path)
        regions_by_file = get_regions_to_remove(manifest, remove_categories)

        original_loc = 0
        filtered_loc = 0
        files_modified = 0
        files_emptied = 0

        for py_file in Path(src_dir).rglob('*.py'):
            rel = str(py_file.relative_to(src_dir))
            flat_name = rel.replace(os.sep, '_')

            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                original_content = f.read()
            original_loc += len(original_content.splitlines())

            if rel in regions_by_file:
                if keep_stubs:
                    filtered = filter_file(str(py_file), regions_by_file[rel])
                else:
                    filtered = filter_file_no_stubs(str(py_file), regions_by_file[rel])
                files_modified += 1
            else:
                filtered = original_content

            filtered_lines = [l for l in filtered.splitlines()
                              if l.strip() and not l.strip().startswith('#')]
            filtered_loc += len(filtered_lines)

            non_trivial_lines = [l for l in filtered.splitlines()
                                  if l.strip()
                                  and not l.strip().startswith('#')
                                  and l.strip() != 'pass']
            if len(non_trivial_lines) < 3:
                files_emptied += 1

            with open(os.path.join(dst_dir, flat_name), 'w', encoding='utf-8') as f:
                f.write(filtered)

        stats[proj_name] = {
            'original_loc': original_loc,
            'filtered_loc': filtered_loc,
            'loc_removed': original_loc - filtered_loc,
            'pct_removed': round(100 * (original_loc - filtered_loc) / max(original_loc, 1), 1),
            'files_modified': files_modified,
            'files_emptied': files_emptied,
        }

    return stats

def main():
    parser = argparse.ArgumentParser(
        description="Filter PyTorch projects to remove boilerplate code"
    )
    parser.add_argument('--repos', required=True,
                        help='Directory containing project subdirectories')
    parser.add_argument('--manifests', required=True,
                        help='Directory containing classification JSON manifests')
    parser.add_argument('--output-full', default='data/submissions_full',
                        help='Output dir for full (unfiltered) JPlag submissions')
    parser.add_argument('--output-filtered', default='data/submissions_filtered',
                        help='Output dir for filtered (Cat C only) JPlag submissions')
    parser.add_argument('--no-stubs', action='store_true',
                        help='Remove code completely instead of leaving pass stubs')
    parser.add_argument('--remove', default='A,B',
                        help='Categories to remove (comma-separated, default: A,B)')
    args = parser.parse_args()

    remove_cats = set(args.remove.split(','))
    keep_stubs = not args.no_stubs

    project_names = sorted([
        d for d in os.listdir(args.repos)
        if os.path.isdir(os.path.join(args.repos, d))
        and not d.startswith('.')
    ])

    print(f"{'=' * 60}")
    print(f"CODE FILTERING PIPELINE")
    print(f"{'=' * 60}")
    print(f"Projects:           {len(project_names)}")
    print(f"Removing categories: {remove_cats}")
    print(f"Keep stubs:          {keep_stubs}")
    print()

    print("Step 1: Preparing full (unfiltered) submissions...")
    prepare_full_submissions(args.repos, args.output_full, project_names)
    for p in project_names:
        n_files = len(list(Path(os.path.join(args.output_full, p)).glob('*.py')))
        print(f"  {p}: {n_files} files")

    print("\nStep 2: Preparing filtered (custom-only) submissions...")
    stats = prepare_filtered_submissions(
        args.repos, args.manifests, args.output_filtered,
        project_names, remove_cats, keep_stubs
    )

    print(f"\n{'=' * 60}")
    print(f"FILTERING RESULTS")
    print(f"{'=' * 60}")
    print(f"{'Project':<25} {'Orig LOC':>9} {'Filt LOC':>9} {'Removed':>9} {'% Rem':>7} {'Files Mod':>10} {'Emptied':>8}")
    print("-" * 80)

    for proj, s in stats.items():
        print(f"{proj:<25} {s['original_loc']:>9} {s['filtered_loc']:>9} "
              f"{s['loc_removed']:>9} {s['pct_removed']:>6.1f}% "
              f"{s['files_modified']:>10} {s['files_emptied']:>8}")

    total_orig = sum(s['original_loc'] for s in stats.values())
    total_filt = sum(s['filtered_loc'] for s in stats.values())
    total_rem = total_orig - total_filt
    print("-" * 80)
    print(f"{'TOTAL':<25} {total_orig:>9} {total_filt:>9} "
          f"{total_rem:>9} {100*total_rem/max(total_orig,1):>6.1f}%")

    print(f"\nFull submissions:     {args.output_full}/")
    print(f"Filtered submissions: {args.output_filtered}/")
    print(f"\nNext: Run JPlag on both directories and compare results.")

if __name__ == '__main__':
    main()
