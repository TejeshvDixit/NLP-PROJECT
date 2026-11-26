# Legal meaning preservation in English and Hindi

This CSET 346 extension applies the scalar pair-regression approach from
[JUDGEBERT](https://aclanthology.org/2025.emnlp-main.5/) to bilingual tenancy
and employment clauses. It includes a reproducible **synthetic pilot** and a
workflow for replacing synthetic targets with independent human judgments.

## Quick start

Run all commands from the repository root. The dependency-free commands
use Python 3.10 or newer.

```sh
python3 -m multilingual_lmp.benchmark
python3 -m unittest multilingual_lmp.test_pipeline
python3 -m multilingual_lmp.evaluate evaluate --output /tmp/lmp-lexical.json
python3 -m multilingual_lmp.evaluate score \
  'The tenant must pay within 7 days.' \
  'The tenant may pay within 7 days.' --language en
```

The scoring command reports the 1–10 score, the decision and the detected
surface cues. The decision thresholds are fixed at 8 and 5. Scores of 8–10 are
Preserved, scores of 5 to below 8 are Partially Preserved, and scores below 5
are Not Preserved. The cues describe observable token changes in the lexical
scorer. They can miss synonymous forms, scope shifts and changes involving
multiple interacting clauses.

## Transformer evaluation

Use an isolated environment. The pilot was run on Python 3.12 with PyTorch
2.15.0 and Transformers 4.57.6 on an Apple Silicon CPU. The checkpoint is a
public download of approximately 450 MB.

```sh
python3.12 -m venv /tmp/venv-lmp
source /tmp/venv-lmp/bin/activate
python -m pip install -r multilingual_lmp/requirements.txt
python -m multilingual_lmp.transformer train \
  --revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
  --output /tmp/lmp-transformer.json --save-model /tmp/lmp-model
python -m multilingual_lmp.transformer score /tmp/lmp-model \
  'The tenant must pay within 7 days.' \
  'The tenant may pay within 7 days.'
```

The regressor jointly encodes the original and rewrite with multilingual
MiniLM, fine-tunes a one-output head with mean squared error, and maps
predictions to the bounded 1–10 scale. The two generic semantic baselines
are mean-pooled sentence cosine and symmetric best-token alignment. They
encode each sentence independently using frozen MiniLM. Word and character
trigram cosine are additional overlap references. The rule-based scorer
combines both overlap measures and subtracts penalties for visible changes.

Four tenancy clause families supply 120 training pairs. The tenancy test
has 30 pairs from a fifth family. Employment contributes 150 test pairs
from five families and supplies zero training pairs. English and Hindi
versions, deadline variants and rewrites stay together in each family.
Every slice reports Pearson correlation, RMSE and MAE on the same 1–10
scale. Pearson is null when either series is constant. Identical and
unrelated controls are supplementary and excluded from the metrics.

## Dataset and evidence

| File | Contents |
| --- | --- |
| [data/pilot.jsonl](data/pilot.jsonl) | 300 pairs with synthetic targets and rationales |
| [data/annotation.csv](data/annotation.csv) | Blind IDs, sentence pairs and blank rating fields |
| [data/annotation_key.json](data/annotation_key.json) | Coordinator-only mapping from blind IDs to original records |
| [data/pilot_results.json](data/pilot_results.json) | Full-set lexical metrics, controls and 16 sampled cases |
| [data/transformer_results.json](data/transformer_results.json) | Held-out pilot metrics, predictions and controls |
| [case_analysis.md](case_analysis.md) | Interpretation of 16 held-out employment examples |

Ten clause families each cover three deadlines, two languages and five
rewrite types. The clauses are original fictional examples. The synthetic
targets are assigned by perturbation type and deliberately fixed within
each type. The diagnostic rules use the same cue types as the generator,
so their performance is a check of the controlled setup. Domain holdout
changes the actor and action vocabulary while both domains share a duty,
deadline and written-waiver structure. Human evidence and authentic,
structurally varied clauses are required for a stronger generalization
claim.

## Human evaluation

Follow the [annotation guide](annotation_guide.md), and pass both completed
copies to the evaluation command. The loader checks blind IDs, rejects
repeated ratings from the same reviewer and requires at least two
independent scores per pair. It uses the mean human score per pair.

```sh
python -m multilingual_lmp.transformer train \
  --revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
  --annotations /path/to/rater-a.csv /path/to/rater-b.csv \
  --output /tmp/lmp-human-results.json
```

Keep human ratings separate from the shared project files until the
reviewers agree to release their de-identified responses.

## Sanity control follow-up

The original eight-epoch pilot remains in `data/transformer_results.json`.
A follow-up adds 48 synthetic anchors: an exact-copy pair and an unrelated
pair for each unique training clause. The revised run uses 12 epochs, the
same 180 benchmark test pairs, and 72 separate control pairs drawn from
test families with unrelated sentences absent from training.

```sh
python -m multilingual_lmp.transformer train \
  --revision e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
  --sanity-anchors --epochs 12 \
  --output /tmp/lmp-anchored-results.json --save-model /tmp/lmp-anchored-model
```

All 36 identical controls were Preserved and all 36 unrelated controls were
Not Preserved under the existing score bands. Identical scores ranged from
9.86 to 10.00. Unrelated scores ranged from 2.49 to 4.15 against a target
of 1. Benchmark RMSE rose from 0.302 to 0.379. The runs also differ in epoch
count, so this comparison does not isolate the effect of augmentation.
See [the follow-up results](data/anchored_results.json) for per-pair scores.
