"""Score clause pairs and run reproducible pilot or human-label evaluation."""
import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"
PATTERNS = {
    "en": (r"\bmust not\b", r"\bmay\b", r"\bunless\b"),
    "hi": (r"मना है", r"की अनुमति है", r"लिखित छूट"),
}


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def tokens(text):
    return re.sub(r"[^\w\u0900-\u0963]+", " ", text).split()


def cosine(left, right):
    a, b = Counter(left), Counter(right)
    length = math.sqrt(sum(x*x for x in a.values()) * sum(x*x for x in b.values()))
    return sum(a[key] * b[key] for key in a.keys() & b.keys()) / length if length else 0.0


def scores(original, simplified, language):
    a, b = normalize(original), normalize(simplified)
    word = cosine(tokens(a), tokens(b))
    chars = cosine((a[i:i+3] for i in range(len(a)-2)),
                   (b[i:i+3] for i in range(len(b)-2)))
    explanations = []
    if re.findall(r"\d+", a) != re.findall(r"\d+", b):
        explanations.append(("The numeric expressions differ.", 3))
    for pattern, reason, penalty in zip(PATTERNS[language], (
        "The prohibition marker changes.", "The permission marker changes.",
        "The exception marker changes.",
    ), (6, 4, 3)):
        if bool(re.search(pattern, a)) != bool(re.search(pattern, b)):
            explanations.append((reason, penalty))
    hybrid = max(1.0, 1 + 9 * (word + chars) / 2 - sum(p for _, p in explanations))
    if a == b and a:
        hybrid = 10.0
    return {"word_cosine": 1 + 9 * word, "trigram_cosine": 1 + 9 * chars,
            "lexical_diagnostics": hybrid}, [reason for reason, _ in explanations]


def decision(score):
    if score >= 8:
        return "Preserved"
    if score >= 5:
        return "Partially Preserved"
    return "Not Preserved"


def load_rows(annotation_files=()):
    rows = [json.loads(line) for line in (DATA / "pilot.jsonl").read_text(encoding="utf-8").splitlines()]
    if not annotation_files:
        return rows
    key = json.loads((DATA / "annotation_key.json").read_text())
    ratings = defaultdict(dict)
    for path in annotation_files:
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for item in csv.DictReader(handle):
                if not item["score"].strip():
                    continue
                pair, annotator = item["pair_id"], item["annotator_id"].strip()
                score = float(item["score"])
                if pair not in key or not annotator or not 1 <= score <= 10:
                    raise ValueError(f"Invalid annotation for {pair}")
                if annotator in ratings[key[pair]]:
                    raise ValueError(f"Duplicate annotation for {pair} by {annotator}")
                ratings[key[pair]][annotator] = score
    missing = [row["pair_id"] for row in rows if len(ratings[row["pair_id"]]) < 2]
    if missing:
        raise ValueError(f"Two ratings per pair required. Pairs missing ratings: {len(missing)}")
    return [{**row, "reference": statistics.mean(ratings[row["pair_id"]].values())} for row in rows]


def metrics(rows, values):
    labels = [row["reference"] for row in rows]
    errors = [p-y for p, y in zip(values, labels, strict=True)]
    pearson = statistics.correlation(labels, values) if len(set(labels)) > 1 and len(set(values)) > 1 else None
    return dict(n=len(labels), pearson=pearson,
                rmse=math.sqrt(statistics.mean(e*e for e in errors)),
                mae=statistics.mean(abs(e) for e in errors))


def report(rows, predictions):
    slices = {"all": rows}
    for field in ("language", "domain"):
        for value in sorted({row[field] for row in rows}):
            slices[value] = [row for row in rows if row[field] == value]
    methods = next(iter(predictions.values()))
    return {name: {method: metrics(group, [predictions[r["pair_id"]][method] for r in group])
                   for method in methods} for name, group in slices.items()}


def pilot(rows):
    predictions = {}
    cases = []
    for row in rows:
        results, reasons = scores(row["original"], row["simplified"], row["language"])
        predictions[row["pair_id"]] = results
        if row["category"] != "paraphrase":
            cases.append({**row, **results, "decision": decision(results["lexical_diagnostics"]),
                          "signals": reasons})
    # Sample two held-out employment families across both languages and all changes.
    cases = [r for r in cases if r["group"] in {"family-07", "family-09"}
             and r["pair_id"].split("-")[1] == "07"]
    sanity = []
    for lang, unrelated in (("en", "Penguins dive."), ("hi", "तारे चमकते हैं।")):
        original = next(row["original"] for row in rows if row["language"] == lang)
        for name, text in (("identical", original), ("unrelated", unrelated)):
            check, _ = scores(original, text, lang)
            sanity.append(dict(language=lang, check=name, **check))
    return dict(metrics=report(rows, predictions), sanity=sanity, reviewed_cases=cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--annotations", nargs="+", default=[])
    evaluate.add_argument("--output", type=Path)
    score = sub.add_parser("score")
    score.add_argument("original")
    score.add_argument("simplified")
    score.add_argument("--language", choices=PATTERNS, required=True)
    args = parser.parse_args()
    if args.command == "score":
        values, reasons = scores(args.original, args.simplified, args.language)
        result = dict(score=values["lexical_diagnostics"], decision=decision(values["lexical_diagnostics"]),
                      signals=reasons, baselines=values)
        output = None
    else:
        result = dict(label_source="human" if args.annotations else "synthetic", **pilot(load_rows(args.annotations)))
        output = args.output
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output:
        output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
