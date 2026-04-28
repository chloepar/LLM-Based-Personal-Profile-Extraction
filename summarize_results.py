"""
Scan all completed result directories and produce a crosstab of evaluation scores.
Rows: (defense, prompt_type, icl_num, adaptive_attack)
Columns: per-info-category scores + mean
"""

import os
import re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import open_config, open_txt

RESULT_BASE = "./result/groq_llama-3.1-8b-instant"
TASK_CONFIG  = "./configs/task_configs/senator.json"
INFO_CATS    = open_txt("./data/system_prompts/info_category.txt")

DIR_RE = re.compile(
    r"^senator_(.+?)_(direct|pseudocode|contextual|persona)_(\d+)_adaptive_attack_(yes|no)$"
)


def evaluate_config(defense_name, prompt_type, icl_num, adaptive_attack, res_dir):
    npz_path = f"{res_dir}/all_raw_responses.npz"
    if not os.path.exists(npz_path):
        return None

    raw = np.load(npz_path, allow_pickle=True)
    all_raw_responses = raw["res"].item()
    all_labels        = raw["label"].item()

    first_cat = INFO_CATS[0]
    if first_cat not in all_raw_responses or len(all_raw_responses[first_cat]) == 0:
        return None  # empty result (email_only bug)

    defense   = PIE.create_defense(defense_name)
    evaluator = PIE.create_evaluator("groq", INFO_CATS)

    task_config  = open_config(TASK_CONFIG)
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
        for info_cat in INFO_CATS:
            try:
                response = all_raw_responses[info_cat][i]
            except (KeyError, IndexError):
                continue
            if not response or response.strip() == "":
                continue
            evaluator.update(response, curr_label, info_cat, defense, verbose=0)

    return evaluator.get_result()


def main():
    rows = []
    for dirname in sorted(os.listdir(RESULT_BASE)):
        m = DIR_RE.match(dirname)
        if not m:
            continue
        defense, prompt_type, icl_num, adaptive_attack = m.group(1), m.group(2), int(m.group(3)), m.group(4)
        res_dir = f"{RESULT_BASE}/{dirname}"

        print(f"Evaluating {dirname} ...", end=" ", flush=True)
        scores = evaluate_config(defense, prompt_type, icl_num, adaptive_attack, res_dir)
        if scores is None:
            print("skipped (no data)")
            continue
        print("done")

        row = {
            "defense": defense,
            "prompt_type": prompt_type,
            "icl_num": icl_num,
            "adaptive_attack": adaptive_attack,
            **{cat: round(v, 4) for cat, v in scores.items()},
        }
        row["mean"] = round(sum(scores.values()) / len(scores), 4)
        rows.append(row)

    if not rows:
        print("No completed results found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["defense", "prompt_type", "icl_num", "adaptive_attack"])

    index_cols = ["defense", "prompt_type", "icl_num", "adaptive_attack"]
    score_cols = INFO_CATS + ["mean"]
    print("\n" + "=" * 80)
    print(df[index_cols + score_cols].to_string(index=False))
    print("=" * 80)

    df.to_csv("./results_summary.csv", index=False)
    print("\nSaved to results_summary.csv")


if __name__ == "__main__":
    main()
