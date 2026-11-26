"""Generate controlled pairs for a pilot and independent human annotation."""
import csv
import json
import random
from pathlib import Path

# Each family provides an English action, a Hindi action, and a deadline trigger.
FAMILIES = [
    ("tenancy", "pay the rent", "किराया चुकाना", "the start of the month", "महीना शुरू होने"),
    ("tenancy", "return the keys", "चाबियाँ लौटाना", "the end of the lease", "पट्टा समाप्त होने"),
    ("tenancy", "report water damage", "पानी से हुए नुकसान की सूचना देना", "discovering a leak", "रिसाव का पता चलने"),
    ("tenancy", "remove personal belongings", "अपना सामान हटाना", "receiving the notice", "नोटिस मिलने"),
    ("tenancy", "provide a forwarding address", "नया पत्राचार पता बताना", "vacating the property", "संपत्ति खाली करने"),
    ("employment", "submit an expense report", "खर्च का विवरण जमा करना", "returning from a work trip", "कार्य यात्रा से लौटने"),
    ("employment", "return the laptop", "लैपटॉप लौटाना", "the end of employment", "नौकरी समाप्त होने"),
    ("employment", "report a workplace injury", "कार्यस्थल पर लगी चोट की सूचना देना", "the accident", "दुर्घटना होने"),
    ("employment", "complete the safety training", "सुरक्षा प्रशिक्षण पूरा करना", "receiving the assignment", "निर्देश मिलने"),
    ("employment", "provide a medical certificate", "चिकित्सा प्रमाणपत्र देना", "resuming work", "दोबारा काम शुरू करने"),
]

VARIANTS = [
    ("paraphrase", 10, "The duty, deadline, trigger and exception are retained."),
    ("permission", 4, "A duty is converted into permission."),
    ("deadline", 6, "The allowed time is doubled."),
    ("exception", 5, "The written waiver exception is omitted."),
    ("contradiction", 2, "The duty is replaced by a prohibition."),
]


def pair_text(domain, lang, action, trigger, days):
    if lang == "en":
        actor = "tenant" if domain == "tenancy" else "employee"
        duty = f"The {actor} must {action} within {days} days of {trigger}"
        waiver = ", unless granted a written waiver."
        return duty + waiver, [
            f"Unless given a written waiver, the {actor} must {action} within {days} days of {trigger}.",
            duty.replace(" must ", " may ", 1) + waiver,
            duty.replace(f" {days} days", f" {2 * days} days", 1) + waiver,
            duty + ".",
            duty.replace(" must ", " must not ", 1) + waiver,
        ]
    actor = "किरायेदार" if domain == "tenancy" else "कर्मचारी"
    deadline = f"{trigger} के {days} दिनों के भीतर"
    waiver = "लिखित छूट न मिलने पर"
    original = f"{waiver}, {actor} को {deadline} {action} होगा।"
    return original, [
        f"{actor} को {deadline} {action} होगा, यदि लिखित छूट नहीं मिली है।",
        f"{waiver}, {actor} को {deadline} {action[:-2]}ने की अनुमति है।",
        original.replace(f" {days} दिनों", f" {2 * days} दिनों", 1),
        f"{actor} को {deadline} {action} होगा।",
        f"{waiver}, {actor} के लिए {deadline} {action} मना है।",
    ]


def generate():
    rows = []
    for i, (domain, en, hi, trigger_en, trigger_hi) in enumerate(FAMILIES, 1):
        for days in (3, 7, 15):
            for language, action, trigger in (("en", en, trigger_en), ("hi", hi, trigger_hi)):
                original, candidates = pair_text(domain, language, action, trigger, days)
                for (category, score, rationale), simplified in zip(VARIANTS, candidates):
                    rows.append(dict(pair_id=f"{i:02d}-{days:02d}-{language}-{category}",
                                     group=f"family-{i:02d}", domain=domain, language=language,
                                     original=original, simplified=simplified, category=category,
                                     reference=score, rationale=rationale))
    return rows


def main():
    root = Path(__file__).resolve().parent / "data"
    root.mkdir(exist_ok=True)
    rows = generate()
    (root / "pilot.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    fields = ["pair_id", "language", "original", "simplified", "annotator_id", "score", "notes"]
    random.Random(346).shuffle(rows)
    with (root / "annotation.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, 1):
            writer.writerow({**row, "pair_id": f"P{index:03d}"})
    (root / "annotation_key.json").write_text(
        json.dumps({f"P{i:03d}": row["pair_id"] for i, row in enumerate(rows, 1)}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
