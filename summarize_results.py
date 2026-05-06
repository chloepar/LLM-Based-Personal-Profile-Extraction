"""
Scan all completed result directories and produce a crosstab of evaluation scores.
Rows: (defense, prompt_type, icl_num, adaptive_attack)
Columns: per-info-category scores + mean

Usage:
  python summarize_results.py
  python summarize_results.py --result_dir ./result/groq_llama-3.1-8b-instant --output results_summary.csv
  python summarize_results.py --task_config ./configs/task_configs/synthetic.json --provider gpt
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import open_config, open_txt

DIR_RE = re.compile(
    r"^(?:\w+)_(.+?)_(direct|pseudocode|contextual|persona)_(\d+)_adaptive_attack_(yes|no)$"
)


def evaluate_config(defense_name, prompt_type, icl_num, adaptive_attack, res_dir, task_config_path, provider, info_cats):
    npz_path = f"{res_dir}/all_raw_responses.npz"
    if not os.path.exists(npz_path):
        return None

    raw = np.load(npz_path, allow_pickle=True)
    all_raw_responses = raw["res"].item()
    all_labels        = raw["label"].item()

    first_cat = info_cats[0]
    if first_cat not in all_raw_responses or len(all_raw_responses[first_cat]) == 0:
        return None  # empty result (email_only bug)

    defense   = PIE.create_defense(defense_name)
    evaluator = PIE.create_evaluator(provider, info_cats)

    task_config  = open_config(task_config_path)
    task_manager, _ = PIE.create_task(task_config)

    total = len(all_raw_responses[first_cat])
    for i in range(total):
        _, curr_label = task_manager[i]
        if isinstance(curr_label.get("education"), list):
            curr_label["education"] = ", ".join(
                f"{e.get('degree','')} from {e.get('institution','')}".strip()
                + (f" in {e['year']}" if e.get("year") else "")
                for e in curr_label["education"]
            )
        for info_cat in info_cats:
            try:
                response = all_raw_responses[info_cat][i]
            except (KeyError, IndexError):
                continue
            if not response or response.strip() == "":
                continue
            evaluator.update(response, curr_label, info_cat, defense, verbose=0)

    return evaluator.get_result(), evaluator.get_filtered_result(), evaluator.get_code_counts()


_CAT_DISPLAY = {
    "name": "Name",
    "birthdate": "Birthdate",
    "gender": "Gender",
    "race_ethnicity": "Race/Ethnicity",
    "committee_roles": "Committee Roles",
    "religion": "Religion",
    "education": "Education",
}

# Attributes and display names for the combined paper comparison table,
# matching the row order in Liu et al. (USENIX Security '25).
_PAPER_ATTRS = ["name", "gender", "religion", "birthdate", "education", "committee_roles"]
_PAPER_ATTR_DISPLAY = {
    "name":            "Full Name",
    "gender":          "Gender",
    "religion":        "Religion",
    "birthdate":       "Birth Year",
    "education":       "Education",
    "committee_roles": "Committee Roles",
}


# ── Combined-table helpers ────────────────────────────────────────────────────

def load_baseline_results(csv_path):
    """Return {method: {attr: score}} from baseline_results.csv, or None."""
    if not os.path.exists(csv_path):
        return None
    bdf = pd.read_csv(csv_path, index_col="attribute")
    return {col: bdf[col].to_dict() for col in bdf.columns}


def compute_gt_coverage(task_config_path, attrs):
    """Return {attr: int pct} from the dataset's labels.json."""
    import json
    task_config = open_config(task_config_path)
    labels = json.load(open(task_config["dataset_info"]["label_path"]))
    n = len(labels)
    return {
        cat: round(100 * sum(
            1 for d in labels.values()
            if d.get(cat) not in (None, "", [], "none")
        ) / n)
        for cat in attrs
    }


def _select_llm_col(df, prompt_type, icl_num, attrs):
    """Return {attr: raw_score} for defense=no, adaptive_attack=no, or None."""
    mask = (
        (df["defense"] == "no") &
        (df["prompt_type"] == prompt_type) &
        (df["icl_num"] == icl_num) &
        (df["adaptive_attack"] == "no")
    )
    if not mask.any():
        return None
    row = df[mask].iloc[0]
    return {cat: row[f"{cat}_raw"] for cat in attrs}


def _pct(scores, cat):
    """Format a score as an integer percentage string, or '—' if unavailable."""
    if scores is None or cat not in scores or scores[cat] is None:
        return "—"
    return f"{int(round(float(scores[cat]) * 100))}%"


def build_combined_html_table(df, baseline, gt_coverage, paper_attrs):
    """HTML table with merged column-group headers matching the paper image."""
    llm_cols = [
        ("direct", _select_llm_col(df, "direct",     0, paper_attrs)),
        ("ICL",    _select_llm_col(df, "direct",     5, paper_attrs)),
        ("pseudo", _select_llm_col(df, "pseudocode", 0, paper_attrs)),
    ]
    base_methods = ["regex", "keyword", "spacy", "bert"]
    n_llm  = len(llm_cols)
    n_base = len(base_methods)

    L = []
    L.append("<table>")
    L.append("  <thead>")
    L.append("    <tr>")
    L.append('      <th rowspan="2"></th>')
    L.append(f'      <th colspan="{n_llm}">LLM PIE</th>')
    L.append(f'      <th colspan="{n_base}">Baseline PIE</th>')
    L.append('      <th rowspan="2">Ground Truth</th>')
    L.append("    </tr>")
    L.append("    <tr>")
    for name, _ in llm_cols:
        L.append(f"      <th>{name}</th>")
    for m in base_methods:
        L.append(f"      <th>{m}</th>")
    L.append("    </tr>")
    L.append("  </thead>")
    L.append("  <tbody>")
    for cat in paper_attrs:
        display = _PAPER_ATTR_DISPLAY.get(cat, cat)
        L.append("    <tr>")
        L.append(f"      <td><b>{display}</b></td>")
        for _, scores in llm_cols:
            L.append(f"      <td>{_pct(scores, cat)}</td>")
        for m in base_methods:
            b = baseline.get(m) if baseline else None
            L.append(f"      <td>{_pct(b, cat)}</td>")
        L.append(f"      <td>{gt_coverage.get(cat, '—')}%</td>")
        L.append("    </tr>")
    L.append("  </tbody>")
    L.append("</table>")
    return "\n".join(L)


def build_combined_latex_table(df, baseline, gt_coverage, paper_attrs):
    """LaTeX booktabs table with multicolumn group headers."""
    llm_cols = [
        ("direct", _select_llm_col(df, "direct",     0, paper_attrs)),
        ("ICL",    _select_llm_col(df, "direct",     5, paper_attrs)),
        ("pseudo", _select_llm_col(df, "pseudocode", 0, paper_attrs)),
    ]
    base_methods = ["regex", "keyword", "spacy", "bert"]
    n_llm  = len(llm_cols)
    n_base = len(base_methods)

    def val(scores, cat):
        if scores is None or cat not in scores:
            return "{--}"
        return str(int(round(float(scores[cat]) * 100)))

    col_spec = "l" + "r" * n_llm + "r" * n_base + "r"
    L = []
    L.append(f"\\begin{{tabular}}{{{col_spec}}}")
    L.append("\\toprule")
    L.append(
        f" & \\multicolumn{{{n_llm}}}{{c}}{{LLM PIE}}"
        f" & \\multicolumn{{{n_base}}}{{c}}{{Baseline PIE}} & \\\\"
    )
    L.append(
        f"\\cmidrule(lr){{2-{1 + n_llm}}}"
        f"\\cmidrule(lr){{{2 + n_llm}-{1 + n_llm + n_base}}}"
    )
    sub_headers = [""] + [n for n, _ in llm_cols] + base_methods + ["GT"]
    L.append(" & ".join(sub_headers) + " \\\\")
    L.append("\\midrule")
    for cat in paper_attrs:
        display = _PAPER_ATTR_DISPLAY.get(cat, cat)
        cells = [display]
        for _, scores in llm_cols:
            cells.append(val(scores, cat))
        for m in base_methods:
            b = baseline.get(m) if baseline else None
            cells.append(val(b, cat))
        cells.append(str(gt_coverage.get(cat, 0)))
        L.append(" & ".join(cells) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    return "\n".join(L)


def build_paper_table(df, info_cats):
    include_icl = df["icl_num"].nunique() > 1 or (df["icl_num"] != 0).any()
    include_adaptive = df["adaptive_attack"].nunique() > 1 or (df["adaptive_attack"] != "no").any()

    header_cols = ["Defense", "Prompt"]
    if include_icl:
        header_cols.append("ICL")
    if include_adaptive:
        header_cols.append("Adaptive")
    score_headers = [_CAT_DISPLAY.get(c, c) for c in info_cats] + ["Mean"]
    all_headers = header_cols + score_headers

    sep = "| " + " | ".join("---" for _ in all_headers) + " |"
    header_row = "| " + " | ".join(all_headers) + " |"

    lines = [header_row, sep]
    for _, row in df.iterrows():
        defense_label = "None" if row["defense"] == "no" else row["defense"]
        prompt_label = row["prompt_type"].capitalize()
        cells = [defense_label, prompt_label]
        if include_icl:
            cells.append(str(int(row["icl_num"])))
        if include_adaptive:
            cells.append(row["adaptive_attack"])
        for cat in info_cats:
            pct = int(round(row[f"{cat}_raw"] * 100))
            cells.append(f"{pct}%")
        mean_pct = int(round(row["mean_raw"] * 100))
        cells.append(f"{mean_pct}%")
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Summarize PIE extraction results into a CSV crosstab.")
    parser.add_argument("--result_dir", default="./result/groq_llama-3.1-8b-instant",
                        help="Directory containing per-config result subdirectories")
    parser.add_argument("--task_config", default="./configs/task_configs/senator.json",
                        help="Task config JSON (controls dataset and label paths)")
    parser.add_argument("--info_cats", default="./data/system_prompts/info_category.txt",
                        help="File listing information categories, one per line")
    parser.add_argument("--provider", default="groq",
                        help="Model provider name (used to select evaluation logic)")
    parser.add_argument("--output", default="./results_summary.csv",
                        help="Output CSV path")
    parser.add_argument("--baseline_csv", default="./baseline_results.csv",
                        help="CSV produced by run_baselines.py")
    args = parser.parse_args()

    info_cats = open_txt(args.info_cats)
    # Derive dataset prefix from the task config filename (e.g. "senator" from senator.json)
    dataset = os.path.splitext(os.path.basename(args.task_config))[0]
    dir_re = re.compile(
        rf"^{dataset}_(.+?)_(direct|pseudocode|contextual|persona)_(\d+)_adaptive_attack_(yes|no)$"
    )

    rows = []
    for dirname in sorted(os.listdir(args.result_dir)):
        m = dir_re.match(dirname)
        if not m:
            continue
        defense, prompt_type, icl_num, adaptive_attack = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        res_dir = f"{args.result_dir}/{dirname}"

        print(f"Evaluating {dirname} ...", end=" ", flush=True)
        result = evaluate_config(defense, prompt_type, icl_num, adaptive_attack, res_dir,
                                 args.task_config, args.provider, info_cats)
        if result is None:
            print("skipped (no data)")
            continue
        print("done")

        raw_scores, filtered_scores, code_counts = result
        index = {"defense": defense, "prompt_type": prompt_type,
                 "icl_num": icl_num, "adaptive_attack": adaptive_attack}
        row = {
            **index,
            **{f"{cat}_raw": round(raw_scores[cat], 4) for cat in info_cats},
            **{f"{cat}_filtered": round(filtered_scores[cat], 4) for cat in info_cats},
            **{f"{cat}_code_pct": round(100.0 * code_counts[cat][0] / code_counts[cat][1], 1)
               if code_counts[cat][1] > 0 else 0.0
               for cat in info_cats},
        }
        row["mean_raw"] = round(sum(raw_scores.values()) / len(raw_scores), 4)
        row["mean_filtered"] = round(sum(filtered_scores.values()) / len(filtered_scores), 4)
        rows.append(row)

    if not rows:
        print("No completed results found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["defense", "prompt_type", "icl_num", "adaptive_attack"])

    index_cols = ["defense", "prompt_type", "icl_num", "adaptive_attack"]
    raw_cols = [f"{cat}_raw" for cat in info_cats] + ["mean_raw"]
    filtered_cols = [f"{cat}_filtered" for cat in info_cats] + ["mean_filtered"]
    code_cols = [f"{cat}_code_pct" for cat in info_cats]

    print("\n" + "=" * 80)
    print("RAW SCORES")
    print(df[index_cols + raw_cols].to_string(index=False))
    print("\nFILTERED SCORES (code-shaped responses zeroed)")
    print(df[index_cols + filtered_cols].to_string(index=False))
    print("\nCODE-SHAPED RESPONSE RATE (%)")
    print(df[index_cols + code_cols].to_string(index=False))
    print("=" * 80)

    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    paper_table = build_paper_table(df, info_cats)
    table_path = os.path.join(os.path.dirname(args.output), "results_table.md")
    with open(table_path, "w") as f:
        f.write(paper_table + "\n")
    print(f"\nPaper table:\n")
    print(paper_table)
    print(f"\nSaved to {table_path}")

    # ── Combined LLM + baseline paper table ──────────────────────────────────
    baseline = load_baseline_results(args.baseline_csv)
    if baseline is None:
        print(f"\n(Skipping combined table — {args.baseline_csv} not found. Run run_baselines.py first.)")
    else:
        gt_coverage = compute_gt_coverage(args.task_config, _PAPER_ATTRS)

        html_table  = build_combined_html_table(df, baseline, gt_coverage, _PAPER_ATTRS)
        latex_table = build_combined_latex_table(df, baseline, gt_coverage, _PAPER_ATTRS)

        out_dir = os.path.dirname(args.output) or "."
        html_path  = os.path.join(out_dir, "results_table_combined.md")
        latex_path = os.path.join(out_dir, "results_table.tex")

        with open(html_path, "w") as f:
            f.write(html_table + "\n")
        with open(latex_path, "w") as f:
            f.write(latex_table + "\n")

        print("\n" + "=" * 80)
        print("COMBINED PAPER TABLE (LLM PIE + Baseline PIE + Ground Truth)")
        print("=" * 80)
        print(html_table)
        print(f"\nSaved HTML  → {html_path}")
        print(f"Saved LaTeX → {latex_path}")


if __name__ == "__main__":
    main()
