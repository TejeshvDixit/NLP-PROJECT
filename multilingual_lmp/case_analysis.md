# Analysis of 16 held-out employment pairs

The subset uses the laptop-return and safety-training families at the seven-day setting. It covers all four material changes in both languages. Each target is assigned by the synthetic benchmark. See the full clauses and unrounded scores in [transformer_results.json](data/transformer_results.json).

| Pair | Target | Pooled cosine | Token alignment | Pair regressor | Lexical diagnostics | Interpretation |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 07-07-en-permission | 4 | 9.92 | 9.84 | 4.22 | 5.68 | A departing employee can choose whether to return the laptop under the rewrite. The lexical score understates the change in obligation. |
| 07-07-en-deadline | 6 | 9.31 | 9.50 | 5.66 | 6.72 | The deadline for returning the laptop moves from day 7 to day 14. The return duty survives under both clauses. |
| 07-07-en-exception | 5 | 9.59 | 9.19 | 5.54 | 5.94 | The employer’s written waiver can no longer release the return obligation. Both clauses still mention returning the laptop. |
| 07-07-en-contradiction | 2 | 9.48 | 9.52 | 2.09 | 3.85 | The employee is forbidden to return the laptop in the original seven-day window. A single inserted negator reverses the practical instruction. |
| 07-07-hi-permission | 4 | 9.92 | 9.81 | 3.68 | 4.89 | The permission phrase allows laptop return without requiring it. Both contextual methods score the changed clause above 9.8. |
| 07-07-hi-deadline | 6 | 9.18 | 9.42 | 5.60 | 6.62 | Replacing 7 with 14 lengthens the time to return the laptop. The pair regressor keeps the prediction in the partial band. |
| 07-07-hi-exception | 5 | 9.77 | 9.47 | 5.53 | 5.86 | Deleting the initial written-waiver phrase extends the duty to cases where a waiver was granted. |
| 07-07-hi-contradiction | 2 | 9.67 | 9.56 | 2.05 | 3.03 | The Hindi prohibition replaces the duty to return the laptop. The shared action and trigger keep pooled cosine close to 10. |
| 09-07-en-permission | 4 | 9.90 | 9.83 | 4.10 | 5.67 | The rewrite gives the employee a choice about safety training. The lexical scorer puts this major change in the partial band. |
| 09-07-en-deadline | 6 | 9.09 | 9.46 | 5.61 | 6.71 | An employee can complete the training seven days later under the rewrite. The assignment remains the event that starts the period. |
| 09-07-en-exception | 5 | 9.52 | 9.19 | 5.45 | 5.92 | The rewrite requires training even when the employee has written permission to skip it. The training action is unchanged. |
| 09-07-en-contradiction | 2 | 9.30 | 9.43 | 2.16 | 3.85 | The rewrite bars safety training during the original compliance window. Pooled cosine still predicts above 9. |
| 09-07-hi-permission | 4 | 9.90 | 9.79 | 3.80 | 5.05 | The Hindi wording converts assigned training to an optional activity. The regressor gives a low score while lexical diagnostics just cross the partial threshold. |
| 09-07-hi-deadline | 6 | 9.13 | 9.42 | 5.54 | 6.67 | The employee gains an extra seven days for safety training. The duty and written-waiver condition both survive. |
| 09-07-hi-exception | 5 | 9.63 | 9.42 | 5.43 | 6.05 | The Hindi rewrite removes the written waiver from the training duty. The error affects the employees who receive a waiver. |
| 09-07-hi-contradiction | 2 | 9.33 | 9.32 | 2.14 | 3.15 | The Hindi rewrite prohibits the very training originally required. The regressor gives a low score despite similar contextual representations. |

Both contextual similarity methods classify all 16 examples as Preserved at the threshold of 8. The regressor matches the synthetic decision bands on all 16. Several obligation-to-permission edits exceed the partial-preservation threshold for lexical diagnostics. All four kinds of edits have the same structure across the two families, which limits how broadly these outcomes can be interpreted.
