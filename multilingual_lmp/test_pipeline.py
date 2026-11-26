"""Regression tests for benchmark isolation, annotation import, and scoring."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from .benchmark import generate
from .evaluate import DATA, decision, load_rows, metrics, scores
from .transformer import TEST_TOPICS, TRAIN_TOPICS, control_pairs, split


class BenchmarkTests(unittest.TestCase):
    def test_control_augmentation_keeps_sources_and_topics_disjoint(self):
        train, test = split(generate())
        anchors = list(control_pairs(train, TRAIN_TOPICS, "anchor"))
        probes = list(control_pairs(test, TEST_TOPICS, "sanity"))
        self.assertEqual((len(anchors), len(probes)), (48, 72))
        self.assertFalse({r["original"] for r in anchors} & {r["original"] for r in probes})
        for lang in TRAIN_TOPICS:
            self.assertFalse(set(TRAIN_TOPICS[lang]) & set(TEST_TOPICS[lang]))
        for row in anchors + probes:
            identical = row["check"] == "identical"
            self.assertEqual(row["original"] == row["simplified"], identical)
            self.assertEqual(row["reference"], 10 if identical else 1)

    def test_paired_rows_and_split(self):
        rows = generate()
        self.assertEqual(rows, load_rows())
        self.assertEqual(len({r["pair_id"] for r in rows}), 300)
        train, test = split(rows)
        self.assertEqual((len(train), len(test)), (120, 180))
        self.assertEqual({r["domain"] for r in train}, {"tenancy"})
        self.assertEqual(sum(r["domain"] == "employment" for r in test), 150)
        self.assertFalse({r["group"] for r in train} & {r["group"] for r in test})
        for partition in (train, test):
            self.assertEqual(sum(r["language"] == "hi" for r in partition), len(partition) // 2)

    def test_annotations_hide_labels_and_replace_references(self):
        key = json.loads((DATA / "annotation_key.json").read_text())
        with (DATA / "annotation.csv").open(encoding="utf-8-sig", newline="") as handle:
            template = csv.DictReader(handle)
            self.assertNotIn("reference", template.fieldnames)
            self.assertNotIn("category", template.fieldnames)
            blind = {row["pair_id"]: row for row in template}
        originals = {r["pair_id"]: r for r in generate()}
        self.assertEqual(len(set(key.values())), len(blind))
        for blind_id, pair_id in key.items():
            self.assertEqual(blind[blind_id]["original"], originals[pair_id]["original"])
            self.assertEqual(blind[blind_id]["simplified"], originals[pair_id]["simplified"])
        with tempfile.TemporaryDirectory() as temp:
            files = []
            for annotator, score in (("reader-a", 8), ("reader-b", 5)):
                path = Path(temp) / f"{annotator}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, ["pair_id", "annotator_id", "score"])
                    writer.writeheader()
                    writer.writerows(dict(pair_id=pair, annotator_id=annotator, score=score) for pair in key)
                files.append(path)
            with self.assertRaisesRegex(ValueError, "Two ratings per pair"):
                load_rows(files[:1])
            self.assertTrue(all(r["reference"] == 6.5 for r in load_rows(files)))
            with self.assertRaisesRegex(ValueError, "Duplicate annotation"):
                load_rows(files + files[:1])

    def test_score_boundaries_and_changed_number(self):
        original = "किरायेदार को 3 दिनों के भीतर किराया चुकाना होगा।"
        same, reasons = scores(original, original, "hi")
        self.assertEqual(same["lexical_diagnostics"], 10)
        self.assertFalse(reasons)
        changed, reasons = scores(original, original.replace("3", "6"), "hi")
        self.assertEqual(decision(changed["lexical_diagnostics"]), "Partially Preserved")
        self.assertEqual(reasons, ["The numeric expressions differ."])
        self.assertEqual([decision(n) for n in (1, 5, 8, 10)], [
            "Not Preserved", "Partially Preserved", "Preserved", "Preserved",
        ])

    def test_constant_predictions_report_undefined_pearson(self):
        result = metrics([{"reference": 1}, {"reference": 3}], [2, 2])
        self.assertEqual(result, dict(n=2, pearson=None, rmse=1, mae=1))


if __name__ == "__main__":
    unittest.main()
