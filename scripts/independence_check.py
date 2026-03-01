#!/usr/bin/env python3

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from itertools import combinations

try:
    import requests
    from tqdm import tqdm
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install requests tqdm")
    sys.exit(1)

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
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

def parse_owner_repo(repo_url):
    match = re.search(r"github\.com/([^/]+)/([^/.]+)", repo_url)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Cannot parse GitHub URL: {repo_url}")

def get_contributors(gh, owner, repo):
    contributors = []
    page = 1
    while True:
        data = gh.get(f"/repos/{owner}/{repo}/contributors",
                      params={"per_page": 100, "page": page})
        if not data:
            break
        contributors.extend(data)
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.3)
    return contributors

def get_repo_info(gh, owner, repo):
    data = gh.get(f"/repos/{owner}/{repo}")
    if data is None:
        return {}
    return {
        "full_name": data.get("full_name", ""),
        "fork": data.get("fork", False),
        "parent": data.get("parent", {}).get("full_name") if data.get("parent") else None,
        "source": data.get("source", {}).get("full_name") if data.get("source") else None,
        "created_at": data.get("created_at", ""),
        "pushed_at": data.get("pushed_at", ""),
    }

def get_user_info(gh, login):
    data = gh.get(f"/users/{login}")
    if data is None:
        return {}
    return {
        "login": login,
        "name": data.get("name", ""),
        "company": (data.get("company") or "").strip().lstrip("@"),
        "location": data.get("location", ""),
        "bio": data.get("bio", ""),
        "blog": data.get("blog", ""),
    }

def check_shared_contributors(contribs_a, contribs_b):
    logins_a = {c["login"] for c in contribs_a}
    logins_b = {c["login"] for c in contribs_b}
    shared = logins_a & logins_b

    if not shared:
        return False, []

    details = []
    for login in shared:
        a_info = next((c for c in contribs_a if c["login"] == login), {})
        b_info = next((c for c in contribs_b if c["login"] == login), {})
        details.append({
            "login": login,
            "contributions_a": a_info.get("contributions", 0),
            "contributions_b": b_info.get("contributions", 0),
        })
    return True, details

def check_fork_relationship(info_a, info_b):
    if info_a.get("parent") == info_b.get("full_name"):
        return True, "A is fork of B"
    if info_b.get("parent") == info_a.get("full_name"):
        return True, "B is fork of A"

    if (info_a.get("source") and info_b.get("source") and
            info_a["source"] == info_b["source"]):
        return True, f"Common source: {info_a['source']}"

    if info_a.get("fork") or info_b.get("fork"):
        return False, "One or both are forks (but of different repos)"

    return False, "Neither is a fork"

def check_shared_org(user_profiles_a, user_profiles_b):
    orgs_a = set()
    orgs_b = set()

    for p in user_profiles_a:
        company = p.get("company", "").lower().strip()
        if company:
            orgs_a.add(company)

    for p in user_profiles_b:
        company = p.get("company", "").lower().strip()
        if company:
            orgs_b.add(company)

    shared = orgs_a & orgs_b
    generic = {"", "freelance", "self-employed", "independent", "none", "n/a"}
    shared = shared - generic

    if shared:
        return True, list(shared)
    return False, []

def creation_date_diff(info_a, info_b):
    try:
        date_a = datetime.fromisoformat(info_a["created_at"].replace("Z", "+00:00"))
        date_b = datetime.fromisoformat(info_b["created_at"].replace("Z", "+00:00"))
        diff = abs((date_a - date_b).days)
        return diff
    except (KeyError, ValueError):
        return -1

def main():
    parser = argparse.ArgumentParser(
        description="Independence verification for ML Clones Study"
    )
    parser.add_argument("--input", required=True,
                        help="CSV with selected projects (must have 'repo_url' column)")
    parser.add_argument("--output", default="independence_matrix.csv",
                        help="Output CSV for pairwise independence matrix")
    parser.add_argument("--output-profiles", default="contributor_profiles.json",
                        help="Output JSON with contributor profile details")
    parser.add_argument("--token", default=None,
                        help="GitHub personal access token (or set GITHUB_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Warning: No GitHub token. Rate limit is 60 requests/hour.")
        print("  A token is strongly recommended for this script.\n")

    gh = GitHubAPI(token=token)

    projects = []
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("repo_url", "").strip()
            if not url:
                continue
            owner, repo = parse_owner_repo(url)
            projects.append({
                "repo_url": url,
                "owner": owner,
                "repo": repo,
                "label": f"{owner}/{repo}",
            })

    n = len(projects)
    n_pairs = n * (n - 1) // 2
    print(f"Loaded {n} projects → {n_pairs} pairwise comparisons\n")

    if n < 2:
        print("Need at least 2 projects. Exiting.")
        sys.exit(1)

    print("Gathering repository metadata and contributor lists...")
    for proj in tqdm(projects, desc="Fetching repo data"):
        proj["info"] = get_repo_info(gh, proj["owner"], proj["repo"])
        proj["contributors"] = get_contributors(gh, proj["owner"], proj["repo"])
        time.sleep(0.3)

    all_logins = set()
    for proj in projects:
        for c in proj["contributors"]:
            all_logins.add(c["login"])

    print(f"\nFetching profiles for {len(all_logins)} unique contributors...")
    user_profiles = {}
    for login in tqdm(all_logins, desc="Fetching profiles"):
        user_profiles[login] = get_user_info(gh, login)
        time.sleep(0.3)

    with open(args.output_profiles, "w", encoding="utf-8") as f:
        json.dump(user_profiles, f, indent=2)
    print(f"  Contributor profiles saved to {args.output_profiles}")

    print(f"\nRunning {n_pairs} pairwise independence checks...")
    results = []
    flags = 0

    for proj_a, proj_b in tqdm(list(combinations(projects, 2)), desc="Checking pairs"):
        label_a = proj_a["label"]
        label_b = proj_b["label"]

        has_shared, shared_details = check_shared_contributors(
            proj_a["contributors"], proj_b["contributors"]
        )
        shared_logins = [d["login"] for d in shared_details] if has_shared else []

        is_fork_related, fork_note = check_fork_relationship(
            proj_a["info"], proj_b["info"]
        )

        profiles_a = [user_profiles.get(c["login"], {}) for c in proj_a["contributors"]]
        profiles_b = [user_profiles.get(c["login"], {}) for c in proj_b["contributors"]]
        has_shared_org, shared_orgs = check_shared_org(profiles_a, profiles_b)

        date_diff = creation_date_diff(proj_a["info"], proj_b["info"])

        issues = []
        if has_shared:
            issues.append(f"SHARED_CONTRIBUTORS: {shared_logins}")
        if is_fork_related:
            issues.append(f"FORK_RELATED: {fork_note}")
        if has_shared_org:
            issues.append(f"SHARED_ORG: {shared_orgs}")
        if 0 <= date_diff <= 7:
            issues.append(f"CLOSE_CREATION_DATES: {date_diff} days apart")

        if not issues:
            verdict = "PASS"
        elif any("FORK_RELATED" in i or "SHARED_CONTRIBUTORS" in i for i in issues):
            verdict = "FAIL"
            flags += 1
        else:
            verdict = "REVIEW"
            flags += 1

        results.append({
            "project_a": label_a,
            "project_b": label_b,
            "shared_contributors": ";".join(shared_logins) if shared_logins else "none",
            "shared_contributor_count": len(shared_logins),
            "fork_related": is_fork_related,
            "fork_note": fork_note,
            "shared_org": ";".join(shared_orgs) if shared_orgs else "none",
            "creation_date_diff_days": date_diff,
            "template_evidence": "MANUAL_CHECK_NEEDED",
            "cross_references": "MANUAL_CHECK_NEEDED",
            "issues": " | ".join(issues) if issues else "none",
            "verdict": verdict,
        })

    fieldnames = list(results[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    fail_count = sum(1 for r in results if r["verdict"] == "FAIL")
    review_count = sum(1 for r in results if r["verdict"] == "REVIEW")

    print(f"\n{'=' * 60}")
    print(f"INDEPENDENCE VERIFICATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Total pairs checked:  {len(results)}")
    print(f"  PASS:               {pass_count}")
    print(f"  FAIL:               {fail_count}")
    print(f"  NEEDS REVIEW:       {review_count}")
    print(f"\nOutput written to:    {args.output}")
    print(f"Profiles saved to:    {args.output_profiles}")

    if fail_count > 0:
        print(f"\n*** WARNING: {fail_count} pair(s) FAILED independence check. ***")
        print("These projects share contributors or are fork-related.")
        print("You must remove one project from each failing pair.\n")
        for r in results:
            if r["verdict"] == "FAIL":
                print(f"  FAIL: {r['project_a']} vs {r['project_b']}")
                print(f"        {r['issues']}")

    if review_count > 0:
        print(f"\n*** NOTE: {review_count} pair(s) need MANUAL REVIEW. ***")
        print("Check for shared org affiliation or close creation dates.\n")
        for r in results:
            if r["verdict"] == "REVIEW":
                print(f"  REVIEW: {r['project_a']} vs {r['project_b']}")
                print(f"          {r['issues']}")

    print(f"\n*** MANUAL STEPS STILL REQUIRED ***")
    print(f"For each pair, you must still manually verify:")
    print(f"  1. No template seeding (inspect first 3-5 commits of each repo)")
    print(f"  2. No cross-references between repos (search code/README/comments)")
    print(f"Update the 'template_evidence' and 'cross_references' columns in")
    print(f"  {args.output} after completing manual checks.")

if __name__ == "__main__":
    main()
