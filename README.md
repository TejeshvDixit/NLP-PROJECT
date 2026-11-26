# Legal Meaning Preservation in English and Hindi

Team Nexus · Tejeshv Dixit · CSET 346 Natural Language Processing

A JUDGEBERT-inspired synthetic benchmark and multilingual MiniLM pair regressor for tenancy and employment clauses. The pilot contains 300 labeled pairs. The follow-up placed all 72 held-out controls in their expected decision bands. Independent human ratings are pending.

Clone or download this repository, then run these commands from its root with Python 3.10 or later:

```sh
python3 -m unittest multilingual_lmp.test_pipeline
python3 -m multilingual_lmp.evaluate score 'The tenant must pay within 7 days.' 'The tenant may pay within 7 days.' --language en
python3 -m multilingual_lmp.evaluate evaluate --output /tmp/lmp-lexical.json
```

See [the project guide](multilingual_lmp/README.md) for model training, reproduction and human-rating instructions.

Upstream: [GRAAL-Research/JUDGEBERT](https://github.com/GRAAL-Research/JUDGEBERT).
