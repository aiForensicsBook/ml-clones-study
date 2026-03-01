#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests tqdm")
    sys.exit(1)

SEARCH_QUERIES = [
    'language:Python "import torch" stars:5..500 pushed:>2023-01-01',
    'language:Python "torch.nn.Module" stars:1..300 pushed:>2023-01-01',
    'language:Python "torch" "optimizer" "training" stars:3..200',
    'language:Python "pytorch" "model" "train" stars:2..300 pushed:>2023-01-01',
    'language:Python "torch.optim" "loss.backward" stars:1..200',
]

EXCLUDE_PATTERNS = [
    "test", "tests", "test_", "_test.py",
    "docs", "doc", "documentation",
    "notebook", "notebooks", ".ipynb",
    "setup.py", "setup.cfg", "conftest.py",
    "migrations", "__pycache__", ".git",
    "examples", "example", "demo",
    "venv", "env", ".env", "site-packages",
]

TRAINING_INDICATORS = [
    r"optimizer\.zero_grad",
    r"loss\.backward",
    r"optimizer\.step",
    r"torch\.optim\.",
    r"nn\.CrossEntropyLoss|nn\.MSELoss|nn\.BCELoss",
    r"model\.train\(\)",
    r"model\.eval\(\)",
]

WRAPPER_INDICATORS = [
    r"import pytorch_lightning",
    r"from pytorch_lightning",
    r"import lightning",
    r"from lightning",
    r"import fastai",
    r"from fastai",
    r"import ignite",
    r"from ignite",
    r"Trainer\(",
]

class GitHubAPI:

    BASE = "https://api.github.com"

    def __init__(self, token=None):
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/vnd.github.v3+json"
        if token:
            self.session.headers["Authorization"] = f"token {token}"
        self.rate_remaining = 999
        self.rate_reset = 0

    def _wait_for_rate_limit(self):
        if self.rate_remaining < 5:
            wait = max(0, self.rate_reset - time.time()) + 5
            print(f"\n  Rate limit nearly exhausted. Waiting {wait:.0f}s ...")
            time.sleep(wait)

    def get(self, endpoint, params=None):
        self._wait_for_rate_limit()
        url = f"{self.BASE}{endpoint}" if endpoint.startswith("/") else endpoint
        resp = self.session.get(url, params=params)

        self.rate_remaining = int(resp.headers.get("X-RateLimit-Remaining", 999))
        self.rate_reset = int(resp.headers.get("X-RateLimit-Reset", 0))

        if resp.status_code == 403 and self.rate_remaining == 0:
            self._wait_for_rate_limit()
            return self.get(endpoint, params)
        elif resp.status_code == 422:
            print(f"  Warning: GitHub returned 422 for query. Skipping.")
            return None
        resp.raise_for_status()
        return resp.json()

    def search_repos(self, query, per_page=30, max_pages=4):
        repos = []
        for page in range(1, max_pages + 1):
            data = self.get("/search/repositories", params={
                "q": query,
                "sort": "updated",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            })
            if data is None:
                break
            items = data.get("items", [])
            if not items:
                break
            repos.extend(items)
            time.sleep(2)
        return repos

    def get_contributors(self, owner, repo):
        try:
            data = self.get(f"/repos/{owner}/{repo}/contributors",
                            params={"per_page": 30})
            if data is None:
                return []
            return data
        except Exception:
            return []

    def get_commit_count(self, owner, repo):
        try:
            url = f"{self.BASE}/repos/{owner}/{repo}/commits"
            resp = self.session.get(url, params={"per_page": 1})
            if resp.status_code != 200:
                return 0
            link = resp.headers.get("Link", "")
            match = re.search(r'page=(\d+)>; rel="last"', link)
            if match:
                return int(match.group(1))
            return len(resp.json())
        except Exception:
            return 0

    def check_is_fork_network(self, owner, repo):
        try:
            data = self.get(f"/repos/{owner}/{repo}")
            if data is None:
                return False, None
            is_fork = data.get("fork", False)
            parent = data.get("parent", {}).get("full_name") if is_fork else None
            return is_fork, parent
        except Exception:
            return False, None

def count_python_loc(repo_path):
    total_loc = 0
    file_count = 0
    for py_file in Path(repo_path).rglob("*.py"):
        path_str = str(py_file).lower()
        if any(excl in path_str for excl in EXCLUDE_PATTERNS):
            continue
        try:
            lines = py_file.read_text(errors="ignore").splitlines()
            code_lines = [l for l in lines
                          if l.strip() and not l.strip().startswith("#")]
            total_loc += len(code_lines)
            file_count += 1
        except Exception:
            continue
    return total_loc, file_count

def check_training_code(repo_path):
    hits = {pattern: False for pattern in TRAINING_INDICATORS}
    for py_file in Path(repo_path).rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
            for pattern in TRAINING_INDICATORS:
                if re.search(pattern, content):
                    hits[pattern] = True
        except Exception:
            continue
    return sum(hits.values()), len(hits)

def check_wrapper_usage(repo_path):
    for py_file in Path(repo_path).rglob("*.py"):
        try:
            content = py_file.read_text(errors="ignore")
            for pattern in WRAPPER_INDICATORS:
                if re.search(pattern, content):
                    return True
        except Exception:
            continue
    return False

def check_pytorch_dependency(repo_path):
    dep_files = ["requirements.txt", "setup.py", "setup.cfg",
                 "pyproject.toml", "environment.yml", "Pipfile"]
    for dep_file in dep_files:
        fp = Path(repo_path) / dep_file
        if fp.exists():
            try:
                content = fp.read_text(errors="ignore").lower()
                if "torch" in content or "pytorch" in content:
                    return True, dep_file
            except Exception:
                continue
    return False, None

def deep_check_repo(clone_url, repo_name):
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = os.path.join(tmpdir, repo_name)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "--quiet", clone_url, repo_path],
                timeout=120, capture_output=True, check=True
            )
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            return {"error": str(e)}

        loc, file_count = count_python_loc(repo_path)
        training_hits, training_total = check_training_code(repo_path)
        uses_wrapper = check_wrapper_usage(repo_path)
        has_pytorch_dep, dep_file = check_pytorch_dependency(repo_path)

        return {
            "python_loc": loc,
            "python_file_count": file_count,
            "training_indicators_found": training_hits,
            "training_indicators_total": training_total,
            "uses_high_level_wrapper": uses_wrapper,
            "has_pytorch_dependency": has_pytorch_dep,
            "pytorch_dep_file": dep_file or "",
        }

def deduplicate(repos):
    seen = set()
    unique = []
    for r in repos:
        name = r["full_name"]
        if name not in seen:
            seen.add(name)
            unique.append(r)
    return unique

def basic_metadata(repo, gh):
    owner = repo["owner"]["login"]
    name = repo["name"]
    full_name = repo["full_name"]

    contributors = gh.get_contributors(owner, name)
    contributor_count = len(contributors)
    contributor_logins = [c.get("login", "?") for c in contributors[:10]]

    commit_count = gh.get_commit_count(owner, name)

    is_fork, fork_parent = gh.check_is_fork_network(owner, name)

    return {
        "repo_url": repo["html_url"],
        "full_name": full_name,
        "owner": owner,
        "name": name,
        "description": (repo.get("description") or "")[:200],
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "size_kb": repo.get("size", 0),
        "default_branch": repo.get("default_branch", "main"),
        "created_at": repo.get("created_at", ""),
        "pushed_at": repo.get("pushed_at", ""),
        "contributor_count": contributor_count,
        "contributor_logins": ";".join(contributor_logins),
        "commit_count": commit_count,
        "is_fork": is_fork,
        "fork_parent": fork_parent or "",
        "license": (repo.get("license") or {}).get("spdx_id", ""),
        "topics": ";".join(repo.get("topics", [])),
        "clone_url": repo.get("clone_url", ""),
    }

def main():
    parser = argparse.ArgumentParser(
        description="GitHub mining for ML Clones Study — Phase 1 candidate discovery"
    )
    parser.add_argument("--output", default="candidates.csv",
                        help="Output CSV file path")
    parser.add_argument("--token", default=None,
                        help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    parser.add_argument("--deep-check", action="store_true",
                        help="Clone each candidate and run local LOC/training checks "
                             "(slower but more accurate)")
    parser.add_argument("--max-pages", type=int, default=4,
                        help="Max pages per search query (default: 4, i.e., ~120 results/query)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Warning: No GitHub token provided. Rate limit is 60 requests/hour.")
        print("  Set GITHUB_TOKEN env var or pass --token for 5,000/hour.\n")

    gh = GitHubAPI(token=token)

    print("=" * 60)
    print("PHASE 1A: GitHub Candidate Discovery")
    print("=" * 60)

    all_repos = []
    for i, query in enumerate(SEARCH_QUERIES, 1):
        print(f"\nQuery {i}/{len(SEARCH_QUERIES)}: {query[:80]}...")
        repos = gh.search_repos(query, max_pages=args.max_pages)
        print(f"  Found {len(repos)} results")
        all_repos.extend(repos)
        time.sleep(3)

    all_repos = deduplicate(all_repos)
    print(f"\nTotal unique candidates after dedup: {len(all_repos)}")

    print(f"\nCollecting metadata for {len(all_repos)} candidates...")
    rows = []
    for repo in tqdm(all_repos, desc="Fetching metadata"):
        row = basic_metadata(repo, gh)

        if row["is_fork"]:
            row["auto_reject"] = "fork"
        elif row["commit_count"] < 30:
            row["auto_reject"] = f"low_commits ({row['commit_count']})"
        else:
            row["auto_reject"] = ""

        rows.append(row)
        time.sleep(0.5)

    if args.deep_check:
        print(f"\nRunning deep checks (cloning repos for local analysis)...")
        non_rejected = [r for r in rows if not r.get("auto_reject")]
        print(f"  Deep-checking {len(non_rejected)} non-rejected candidates")

        for row in tqdm(non_rejected, desc="Deep checking"):
            result = deep_check_repo(row["clone_url"], row["name"])
            if "error" in result:
                row["deep_check_error"] = result["error"]
                continue
            row.update(result)

            loc = result.get("python_loc", 0)
            files = result.get("python_file_count", 0)
            if loc < 500:
                row["auto_reject"] = f"too_few_loc ({loc})"
            elif loc > 5000:
                row["auto_reject"] = f"too_many_loc ({loc})"
            elif files < 5:
                row["auto_reject"] = f"too_few_files ({files})"
            elif result.get("uses_high_level_wrapper"):
                row["auto_reject"] = "uses_wrapper (Lightning/fastai)"
            elif result.get("training_indicators_found", 0) < 3:
                row["auto_reject"] = f"weak_training_code ({result['training_indicators_found']}/{result['training_indicators_total']})"
            elif not result.get("has_pytorch_dependency"):
                row["auto_reject"] = "no_pytorch_in_deps"

    if not rows:
        print("No candidates found. Try broadening search queries.")
        sys.exit(1)

    fieldnames = list(rows[0].keys())
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    auto_rejected = sum(1 for r in rows if r.get("auto_reject"))
    remaining = total - auto_rejected

    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Total candidates found:    {total}")
    print(f"Auto-rejected:             {auto_rejected}")
    print(f"Remaining for screening:   {remaining}")
    print(f"Output written to:         {args.output}")
    print(f"\nNext step: Open {args.output} and manually screen remaining")
    print(f"candidates against the full inclusion criteria checklist.")
    print(f"Target: select 10–15 projects across 3 ML domains.")

if __name__ == "__main__":
    main()
