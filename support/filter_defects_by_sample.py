#!/usr/bin/env python3
"""
filter_defects_by_sample.py

Filter the defects/issues file so that it only keeps rows whose `repo`
is present in the current MCP host sample file.

Inputs (edit paths as needed):
  - SAMPLE_FILE: CSV with at least column `full_name`
  - DEFECTS_FILE: CSV with at least column `repo`

Output:
  - OUT_FILE: filtered defects CSV
"""

import pandas as pd

# --------------------------------------------------------------------
# Paths – tweak these to match your project layout
SAMPLE_FILE  = "Repo/mcp_hosts_sampleV0.csv"
DEFECTS_FILE = "data/defects.csv"
OUT_FILE     = "data/defects_hosts_sample.csv"
# --------------------------------------------------------------------

def main():
    # Load sample host repos
    sample = pd.read_csv(SAMPLE_FILE)
    if "full_name" not in sample.columns:
        raise SystemExit(f"`full_name` column not found in {SAMPLE_FILE}")

    sample_repos = (
        sample["full_name"]
        .astype(str)
        .str.strip()
        .unique()
    )
    repo_set = set(sample_repos)

    print(f"✅ Loaded {len(sample)} rows from {SAMPLE_FILE}")
    print(f"   Unique host repos in sample: {len(repo_set)}")

    # Load defects/issues
    defects = pd.read_csv(DEFECTS_FILE)
    if "repo" not in defects.columns:
        raise SystemExit(f"`repo` column not found in {DEFECTS_FILE}")

    print(f"✅ Loaded {len(defects)} defect rows from {DEFECTS_FILE}")

    # Filter defects to only those whose repo is in the sample
    filtered = defects[
        defects["repo"].astype(str).str.strip().isin(repo_set)
    ].copy()

    print(f"🎯 Defects after filtering: {len(filtered)}")

    # Save filtered defects
    filtered.to_csv(OUT_FILE, index=False, encoding="utf-8")
    print(f"💾 Saved filtered defects to {OUT_FILE}")

if __name__ == "__main__":
    main()
