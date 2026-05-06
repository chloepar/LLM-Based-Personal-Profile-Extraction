"""
Baseline PIE evaluation for the senator dataset.
Implements regex, keyword, spacy, and bert (NER) extractors — matching
the four Baseline PIE columns from Liu et al. (USENIX Security '25).

Usage:
  python run_baselines.py
  python run_baselines.py --output baseline_results.csv
"""

import re
import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
import LLMPersonalInfoExtraction as PIE
from LLMPersonalInfoExtraction.utils import open_config, open_txt, get_parser, parsed_data_to_string
from LLMPersonalInfoExtraction.evaluator.Evaluator import Evaluator

# ── Known-value lists ─────────────────────────────────────────────────────────

RELIGIONS = [
    "catholic", "protestant", "jewish", "muslim", "islam", "christian",
    "baptist", "methodist", "presbyterian", "episcopalian", "lutheran",
    "mormon", "latter-day saints", "lds", "buddhist", "hindu", "unitarian",
    "congregationalist", "quaker", "evangelical", "pentecostal", "orthodox",
    "jehovah", "seventh-day adventist", "disciples of christ", "reform jewish",
    "reformed church", "church of christ",
]

DEGREE_RE = re.compile(
    r"\b(B\.?A\.?|B\.?S\.?|M\.?A\.?|M\.?S\.?|Ph\.?D\.?|J\.?D\.?|M\.?D\.?"
    r"|M\.?B\.?A\.?|LL\.?B\.?|LL\.?M\.?|Ed\.?D\.?|B\.?Ed\.?)\b",
    re.IGNORECASE,
)
EDU_INSTITUTION_RE = re.compile(
    r"\b([A-Z][a-zA-Z\s]+(?:University|College|Institute|School)\b[^.\n]{0,60})"
)
COMMITTEE_RE = re.compile(
    r"(?:(?:Senate|House|Joint|Select|Special)\s+)?Committee\s+"
    r"(?:on|of|for|to|on the)\s+[^\n;]{3,120}",
    re.IGNORECASE,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _extend_to_word(text, start, end):
    """Expand an entity's end index to the next whitespace/punctuation boundary."""
    while end < len(text) and text[end] not in " \n\t.,;:()'\"":
        end += 1
    return text[start:end].strip()


def _pronoun_gender(text):
    tl = text.lower()
    he  = len(re.findall(r"\b(he|him|his)\b", tl))
    she = len(re.findall(r"\b(she|her|hers)\b", tl))
    if he > she:  return "Male"
    if she > he:  return "Female"
    return "none"


def _religion_keyword(text):
    tl = text.lower()
    for r in RELIGIONS:
        if re.search(rf"\b{re.escape(r)}\b", tl):
            return r.title()
    return "none"


# ── Regex extractor ───────────────────────────────────────────────────────────

def regex_extract(info_cat, text):
    if info_cat == "name":
        # "Senator\n{First Last}" (multi-line layout)
        m = re.search(r"Senator\n([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)", text)
        if m: return m.group(1)
        # "{First Last} U.S. Senator" (name before title)
        m = re.search(r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+U\.S\.\s+Senator", text, re.MULTILINE)
        if m: return m.group(1)
        # "U.S. Senator {First Last}" or "Senator {First Last}"
        m = re.search(r"(?:U\.S\.\s+)?Senator\s+([A-Z][a-z]+(?:[ -][A-Z][a-z]+)+)", text)
        if m: return m.group(1)
        # Name followed by a bio verb in first 2000 chars — handles "Bernie Sanders is serving..."
        m = re.search(
            r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s+"
            r"(?:was born|is serving|has represented|represents|graduated|grew up|"
            r"was elected|has served|joined the|was sworn|has been|is a)",
            text[:2000],
        )
        if m: return m.group(1)
        return "none"

    if info_cat == "gender":
        return _pronoun_gender(text)

    if info_cat == "religion":
        return _religion_keyword(text)

    if info_cat == "birthdate":
        m = re.search(
            r"born\b.{0,80}"
            r"((?:January|February|March|April|May|June|July|August|"
            r"September|October|November|December)\s+\d{1,2},?\s+1[89]\d\d)",
            text, re.IGNORECASE,
        )
        if m: return m.group(1)
        m = re.search(r"born\b.{0,100}\b(1[89]\d\d)\b", text, re.IGNORECASE)
        if m: return m.group(1)
        return "none"

    if info_cat == "education":
        results = []
        for dm in DEGREE_RE.finditer(text):
            ctx = text[max(0, dm.start()-300):min(len(text), dm.end()+300)]
            im = EDU_INSTITUTION_RE.search(ctx)
            if im:
                results.append(f"{dm.group()} from {im.group(1).strip()}")
        if results:
            return ", ".join(list(dict.fromkeys(results))[:4])
        unis = EDU_INSTITUTION_RE.findall(text)
        return ", ".join(u.strip() for u in unis[:4]) if unis else "none"

    if info_cat == "committee_roles":
        found = [m.group().strip() for m in COMMITTEE_RE.finditer(text)]
        found += re.findall(r"[A-Z][a-zA-Z\s]+ Committee\b", text)
        found = list(dict.fromkeys(found))
        return "|".join(found[:15]) if found else "none"

    if info_cat == "race_ethnicity":
        races = ["white", "black", "african american", "hispanic", "latino",
                 "asian american", "asian", "native american", "pacific islander"]
        tl = text.lower()
        for race in races:
            if re.search(rf"\b{re.escape(race)}\b", tl):
                return race.title()
        return "none"

    return "none"


# ── Keyword extractor ─────────────────────────────────────────────────────────

def keyword_extract(info_cat, text):
    if info_cat == "name":
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == "Senator":
                candidates = [l.strip() for l in lines[i+1:i+4] if l.strip()]
                if candidates and re.match(r"^[A-Z][a-z]", candidates[0]):
                    return candidates[0]
        # Fall back: find first "Senator X Y" pattern
        return regex_extract("name", text)

    if info_cat in ("gender", "race_ethnicity"):
        return regex_extract(info_cat, text)

    if info_cat == "religion":
        return _religion_keyword(text)

    if info_cat == "birthdate":
        for line in text.split("\n"):
            if "born" in line.lower():
                m = re.search(r"1[89]\d\d", line)
                if m: return m.group()
        return "none"

    if info_cat == "education":
        edu_kws = {"university", "college", "institute", "school",
                   "graduated", "bachelor", "master", "doctorate", "degree"}
        sentences = re.split(r"[.\n]", text)
        hits = [s.strip() for s in sentences
                if any(kw in s.lower() for kw in edu_kws) and len(s.strip()) > 15]
        return ". ".join(hits[:3]) if hits else "none"

    if info_cat == "committee_roles":
        lines = text.split("\n")
        hits = [l.strip() for l in lines
                if "committee" in l.lower() and len(l.strip()) > 5]
        return "|".join(hits[:15]) if hits else "none"

    return "none"


# ── SpaCy extractor ───────────────────────────────────────────────────────────

def spacy_extract(nlp, info_cat, text):
    if info_cat == "name":
        doc = nlp(text[:3000])
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                return ent.text
        return "none"

    if info_cat in ("gender", "religion", "race_ethnicity"):
        return regex_extract(info_cat, text)

    if info_cat == "birthdate":
        tl = text.lower()
        born_idx = tl.find("born")
        if born_idx >= 0:
            window = text[max(0, born_idx-10):born_idx+200]
            for ent in nlp(window).ents:
                if ent.label_ == "DATE":
                    return ent.text
        for ent in nlp(text[:5000]).ents:
            if ent.label_ == "DATE" and re.search(r"1[89]\d\d", ent.text):
                return ent.text
        return "none"

    if info_cat == "education":
        edu_kws = {"university", "college", "institute", "school"}
        doc = nlp(text[:6000])
        orgs = [ent.text for ent in doc.ents
                if ent.label_ == "ORG" and any(kw in ent.text.lower() for kw in edu_kws)]
        return ", ".join(list(dict.fromkeys(orgs))[:4]) if orgs else "none"

    if info_cat == "committee_roles":
        doc = nlp(text)
        committees = [ent.text for ent in doc.ents if "committee" in ent.text.lower()]
        regex_hits = [m.group().strip() for m in COMMITTEE_RE.finditer(text)]
        merged = list(dict.fromkeys(committees + regex_hits))
        return "|".join(merged[:15]) if merged else "none"

    return "none"


# ── BERT NER extractor ────────────────────────────────────────────────────────

def _ner_chunks(text, chunk_chars=2000, overlap=200):
    """Yield overlapping character-level chunks with their start offset."""
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        yield start, text[start:end]
        if end == len(text):
            break
        start += chunk_chars - overlap


def bert_extract(ner_pipeline, info_cat, text):
    if info_cat == "name":
        for chunk_start, chunk in _ner_chunks(text, chunk_chars=1000):
            ents = ner_pipeline(chunk)
            for e in ents:
                if e["entity_group"] == "PER" and e["score"] > 0.9:
                    return _extend_to_word(chunk, e["start"], e["end"])
        return "none"

    if info_cat in ("gender", "religion", "race_ethnicity"):
        return regex_extract(info_cat, text)

    if info_cat == "birthdate":
        tl = text.lower()
        born_idx = tl.find("born")
        if born_idx >= 0:
            window = text[max(0, born_idx-10):born_idx+300]
            m = re.search(r"1[89]\d\d", window)
            if m: return m.group()
        return "none"

    if info_cat == "education":
        edu_kws = {"university", "college", "institute", "school"}
        orgs = []
        for chunk_start, chunk in _ner_chunks(text):
            for e in ner_pipeline(chunk):
                if e["entity_group"] == "ORG":
                    span = _extend_to_word(chunk, e["start"], e["end"])
                    if any(kw in span.lower() for kw in edu_kws):
                        orgs.append(span)
        orgs = list(dict.fromkeys(orgs))
        return ", ".join(orgs[:4]) if orgs else "none"

    if info_cat == "committee_roles":
        orgs = []
        for chunk_start, chunk in _ner_chunks(text):
            for e in ner_pipeline(chunk):
                if e["entity_group"] in ("ORG", "MISC"):
                    span = _extend_to_word(chunk, e["start"], e["end"])
                    if "committee" in span.lower():
                        orgs.append(span)
        regex_hits = [m.group().strip() for m in COMMITTEE_RE.finditer(text)]
        merged = list(dict.fromkeys(orgs + regex_hits))
        return "|".join(merged[:15]) if merged else "none"

    return "none"


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_baseline(extractor_fn, info_cats, task_manager, provider="groq"):
    defense = PIE.create_defense("no")
    evaluator = Evaluator(provider, info_cats)

    for i in range(len(task_manager)):
        raw_list, curr_label = task_manager[i]

        # Parse HTML to text (same pipeline as LLM PIE)
        raw = "\n".join(raw_list)
        parser = get_parser(task_manager.dataset, include_link=False)
        parser.feed(raw)
        text = parsed_data_to_string(task_manager.dataset, parser.data)
        text = text.replace("href\n#\n", "")

        # Flatten education list label
        if isinstance(curr_label.get("education"), list):
            curr_label["education"] = ", ".join(
                f"{e.get('degree','')} from {e.get('institution','')}".strip()
                + (f" in {e['year']}" if e.get("year") else "")
                for e in curr_label["education"]
            )

        for info_cat in info_cats:
            response = extractor_fn(info_cat, text)
            evaluator.update(response, curr_label, info_cat, defense, verbose=0)

    return evaluator.get_result()


def evaluate_baseline_spacy(nlp, info_cats, task_manager, provider="groq"):
    def fn(info_cat, text):
        return spacy_extract(nlp, info_cat, text)
    return evaluate_baseline(fn, info_cats, task_manager, provider)


def evaluate_baseline_bert(ner_pipeline, info_cats, task_manager, provider="groq"):
    def fn(info_cat, text):
        return bert_extract(ner_pipeline, info_cat, text)
    return evaluate_baseline(fn, info_cats, task_manager, provider)


# ── Table formatting ──────────────────────────────────────────────────────────

_ATTR_DISPLAY = {
    "name":             "Full Name",
    "gender":           "Gender",
    "religion":         "Religion",
    "birthdate":        "Birth Year",
    "education":        "Education",
    "committee_roles":  "Committee Roles",
    "race_ethnicity":   "Race/Ethnicity",
}

def print_table(results_by_method, info_cats):
    methods = list(results_by_method.keys())
    col_w = 10

    header = f"{'Attribute':<20}" + "".join(f"{m:>{col_w}}" for m in methods)
    print("\n" + "=" * len(header))
    print("BASELINE PIE RESULTS")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cat in info_cats:
        label = _ATTR_DISPLAY.get(cat, cat)
        row = f"{label:<20}"
        for m in methods:
            pct = int(round(results_by_method[m].get(cat, 0) * 100))
            row += f"{pct:>{col_w-1}}%"
        print(row)
    print("=" * len(header))


def save_csv(results_by_method, info_cats, path):
    import csv
    methods = list(results_by_method.keys())
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["attribute"] + methods)
        for cat in info_cats:
            row = [cat]
            for m in methods:
                row.append(round(results_by_method[m].get(cat, 0), 4))
            writer.writerow(row)
    print(f"Saved to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_config",  default="./configs/task_configs/senator.json")
    parser.add_argument("--info_cats",    default="./data/system_prompts/info_category.txt")
    parser.add_argument("--provider",     default="groq")
    parser.add_argument("--output",       default="./baseline_results.csv")
    parser.add_argument("--skip_bert",    action="store_true",
                        help="Skip BERT NER (faster, for testing)")
    args = parser.parse_args()

    info_cats = open_txt(args.info_cats)
    task_config = open_config(args.task_config)
    task_manager, _ = PIE.create_task(task_config)
    print(f"Evaluating {len(task_manager)} senator profiles on {len(info_cats)} attributes.\n")

    results = {}

    print("Running regex baseline...")
    results["regex"] = evaluate_baseline(regex_extract, info_cats, task_manager, args.provider)
    print("Done.\n")

    print("Running keyword baseline...")
    results["keyword"] = evaluate_baseline(keyword_extract, info_cats, task_manager, args.provider)
    print("Done.\n")

    print("Running spacy baseline...")
    import spacy
    nlp = spacy.load("en_core_web_sm")
    results["spacy"] = evaluate_baseline_spacy(nlp, info_cats, task_manager, args.provider)
    print("Done.\n")

    if not args.skip_bert:
        print("Running bert baseline (dslim/bert-base-NER)...")
        from transformers import pipeline as hf_pipeline
        ner = hf_pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
        results["bert"] = evaluate_baseline_bert(ner, info_cats, task_manager, args.provider)
        print("Done.\n")

    print_table(results, info_cats)
    save_csv(results, info_cats, args.output)


if __name__ == "__main__":
    main()
