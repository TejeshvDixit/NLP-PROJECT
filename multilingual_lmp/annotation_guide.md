# Human annotation guide

For each pair, judge how much of the original clause's legal effect the
rewrite preserves. Compare the actor, action, duty or permission, timing,
scope, conditions and exceptions. Read each pair on its own. Use the facts
stated in the pair and leave a note if a missing fact makes the legal effect
ambiguous.

| Score | Anchor |
| --- | --- |
| 10 | Equivalent legal effect across the whole clause |
| 8–9 | Substantially equivalent, with minor ambiguity or a small loss of precision |
| 5–7 | The main duty, permission or entitlement survives, with a material change to a condition, exception, timing or scope |
| 3–4 | A major duty, permission or entitlement changes, even though the topic remains the same |
| 1–2 | The clause reverses the original effect or has almost no legal meaning in common |

Choose the exact integer within the range that fits the practical consequence.
Small word changes can deserve low scores. For example, replacing “must” with
“may” changes an obligation into permission. When one reading of a clause
changes the result substantially, record the issue in `notes`.

## Reviewer workflow

1. Each reviewer receives a separate copy of `data/annotation.csv` and this
   guide. Reviewers should be fluent in the language they rate and comfortable
   reading legal clauses. The coordinator keeps the ID key and pilot outputs.
2. Calibrate using separate practice examples. Each reviewer then completes
   the benchmark independently. Fill `annotator_id` with the same anonymous
   reviewer code on every rated row, enter a score from 1 to 10 and add a
   short note where the reason may be ambiguous. Save cleaned CSV as UTF-8.
3. Collect at least two independent judgments per pair before revealing
   other scores. Review large disagreements and categorize their causes.
   Preserve the original judgments and report the number of ratings, mean
   absolute disagreement by language and any post-review adjudication
   separately.
4. Use the untouched mean rating per pair as the primary evaluation target.
   Keep the fixed family split and score thresholds for comparability. Report
   the reviewer qualifications, per-language coverage and limitations in
   the paper before drawing conclusions about human alignment.
