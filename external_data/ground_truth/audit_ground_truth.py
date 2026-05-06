"""
Ground Truth Audit — Missing Value Analysis
============================================
Produces:
  1. Summary stats: % missing per field (printed + CSV)
  2. Row-level flags: one column per field, True = GT missing (CSV)

Usage:
    python audit_ground_truth.py --gt path/to/senate_ground_truth.csv
    python audit_ground_truth.py --gt path/to/senate_ground_truth.csv --out audit_results/
"""

import argparse
import os
import pandas as pd


# ── Fields to audit ──────────────────────────────────────────────────────────
# Maps column name in CSV → human-readable label
# Source: groundtruth.py scrape targets (Wikipedia, Ballotpedia, Pew)
GT_FIELDS = {
    "full_name":       "Full name       (Wikipedia h1)",
    "birthdate":       "Birthdate       (Wikipedia infobox)",
    "gender":          "Gender          (Wikipedia infobox / pronouns)",
    "race_ethnicity":  "Race/ethnicity  (Wikipedia infobox)",
    "religion":        "Religion        (Pew Research merge)",
    "committee_roles": "Committee roles (Ballotpedia 2025-2026)",
}

# Absence indicators consistent with evaluator.is_absence_indicator()
ABSENCE_TERMS = {
    "none", "null", "unknown", "not found", "not available",
    "n/a", "—", "??", "no info", "missing", "unavailable",
    "not provided", "not stated", "error",
}


def is_missing(val) -> bool:
    """Return True if value is NaN, empty, or an absence indicator string."""
    if pd.isna(val):
        return True
    s = str(val).strip()
    if s == "":
        return True
    if s.lower() in ABSENCE_TERMS:
        return True
    return False


def audit(gt_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(gt_path)
    n = len(df)
    print(f"\nLoaded {n} senators from: {gt_path}\n")

    # ── 1. Row-level flags ───────────────────────────────────────────────────
    flags = pd.DataFrame()

    # Carry identifying columns if present
    for id_col in ["senator_id", "full_name", "name", "state"]:
        if id_col in df.columns:
            flags[id_col] = df[id_col]

    present_fields = []
    for col, label in GT_FIELDS.items():
        if col in df.columns:
            flags[f"missing_{col}"] = df[col].apply(is_missing)
            present_fields.append((col, label))
        else:
            print(f"  [WARN] Column '{col}' not found in CSV — skipping")

    flags["missing_count"] = flags[[f"missing_{c}" for c, _ in present_fields]].sum(axis=1)

    flags_path = os.path.join(out_dir, "gt_missing_flags.csv")
    flags.to_csv(flags_path, index=False)
    print(f"Row-level flags saved → {flags_path}\n")

    # ── 2. Summary stats ─────────────────────────────────────────────────────
    summary_rows = []
    print(f"{'Field':<45} {'Missing':>8}  {'Present':>8}  {'% Missing':>10}")
    print("─" * 77)

    for col, label in present_fields:
        n_missing = int(flags[f"missing_{col}"].sum())
        n_present = n - n_missing
        pct = n_missing / n * 100
        summary_rows.append({
            "field": col,
            "source": label.split("(")[1].rstrip(")") if "(" in label else "",
            "n_total": n,
            "n_missing": n_missing,
            "n_present": n_present,
            "pct_missing": round(pct, 1),
        })
        print(f"{label:<45} {n_missing:>8}  {n_present:>8}  {pct:>9.1f}%")

    print("─" * 77)

    # Senators with ANY missing field
    any_missing = int((flags["missing_count"] > 0).sum())
    all_missing = int((flags["missing_count"] == len(present_fields)).sum())
    print(f"\nSenators with ≥1 missing GT field : {any_missing} / {n}  ({any_missing/n*100:.1f}%)")
    print(f"Senators with ALL fields missing   : {all_missing} / {n}")

    # Per-field breakdown of senators missing multiple fields
    print("\nMissing field co-occurrence (top combinations):")
    flag_cols = [f"missing_{c}" for c, _ in present_fields]
    combo_counts = flags[flag_cols].value_counts().head(10)
    for combo, count in combo_counts.items():
        missing_names = [present_fields[i][0] for i, v in enumerate(combo) if v]
        if missing_names:
            print(f"  {count:>3}x  missing: {', '.join(missing_names)}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "gt_missing_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary stats saved → {summary_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit ground truth CSV for missing values.")
    parser.add_argument("--gt",  required=True, help="Path to senate_ground_truth.csv")
    parser.add_argument("--out", default="audit_results/", help="Output directory (default: audit_results/)")
    args = parser.parse_args()
    audit(args.gt, args.out)
