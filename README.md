# Ranking Lab

A small, reproducible offline ranking experiment on the original MovieLens 100K ratings. It compares an interaction-popularity baseline with an item-item collaborative filter. It uses Python 3.10+ and only the standard library.

```sh
python3 ranking_lab.py download
python3 -m unittest discover -s tests -v
python3 ranking_lab.py run --output results.json
```

The download uses the [official GroupLens archive](https://files.grouplens.org/datasets/movielens/ml-100k.zip), checks its MD5 (`0e33842e24a9c977be4e0107933c0723`), and extracts only `u.data`, `u.item`, and `README` to a new directory. The checksum detects changes; MD5 is not an authenticity guarantee. The archive and extracted data are not included in this repository. The full results file contains per-user scores and remains local by default.

**Data terms:** The archive's README permits research use under conditions: acknowledge the data in publications, do not imply GroupLens or University of Minnesota endorsement, do not redistribute without permission, and obtain GroupLens faculty permission before commercial or revenue-bearing use. Read the downloaded README for the complete terms. Cite [Harper and Konstan, The MovieLens Datasets: History and Context (2015)](https://doi.org/10.1145/2827872). This 1997–98 sample is old and filtered to users with at least 20 ratings and complete demographics.

## Evaluation design

```text
all rating timestamps ── sort ── 80% cutoff ── 90% cutoff ── end
                           train    validation        test
                                      │                │
                              choose on NDCG     refit and evaluate
```

- The cutoff is the timestamp at the zero-indexed 80% or 90% event position. Ties always go together into the later window. Actual window sizes can differ from 80/10/10. Every model sees only records strictly earlier than its window.
- A rating of at least 4 is positive. For each user, the held-out window is one basket of novel positive items. All prior ratings, including negative ratings, are removed from that user's candidates. There are no within-window updates or sequential predictions. Users with no novel positive target in the window are excluded and counted.
- The full candidate catalog consists of **all items observed in ratings before the window**, not sampled negatives. This conservative availability proxy does not prove an item was actually available. Held-out positives for items never observed before the cutoff cannot be recommended; they remain in the metric denominator and are counted. New users are evaluated and counted. Item CF has no personal signal for a new user and breaks zero-score ties by item ID.
- Popularity counts prior positive ratings. Item CF sums cosine similarities of binary positive co-occurrences; its shrinkage grid is 0, 10, 100. All scoring ties use ascending item ID. Select one of the fixed four settings by validation NDCG@10, with the grid order as tie-break; refit on train plus validation. Report the fixed popularity baseline and selected setting on test. If popularity wins, the single test computation is reused. The test is not used for tuning.
- Paired validation comparisons resample eligible users with replacement, preserving each user’s difference from popularity. The 95% percentile intervals use 2,000 draws and seed 42. They are descriptive for this fixed validation window and selection grid; they do not correct for model selection or multiple comparisons. Resampling users does not account for shared time effects or other correlations.
- Hit-rate@10 is the fraction of eligible users with at least one relevant item in their top 10. NDCG@10 is binary gain, normalized by the ideal ordering of all eligible held-out positives, including unavailable items. Coverage is the fraction of the pre-window catalog recommended at least once across eligible users. The JSON includes each user's hit, NDCG, target count, unavailable count, and cold-user indicator.

## Measured run

A local run of the command above on the verified official archive produced the following. These results use one split and one fixed grid; the bootstrap intervals below do not estimate variation across time windows.

| Setting | Validation NDCG@10 | Validation hit-rate@10 |
|---|---:|---:|
| Popularity | 0.2365 | 0.6505 |
| Item CF, shrink 0 | 0.0915 | 0.3978 |
| Item CF, shrink 10 | 0.0903 | 0.3925 |
| Item CF, shrink 100 | 0.0878 | 0.3925 |

For validation NDCG@10, the paired item-CF-minus-popularity differences were −0.1450 (95% percentile interval −0.1785 to −0.1120) for shrink 0; −0.1462 (−0.1794 to −0.1143) for shrink 10; and −0.1487 (−0.1810 to −0.1167) for shrink 100. These intervals describe user resampling on the same validation window used for selection; they are not independent confirmation and cannot account for correlated users or future time periods. The JSON also reports paired hit-rate intervals.

Popularity was selected. Its held-out test NDCG@10 was **0.2445**, hit-rate@10 **0.6536**, and catalog coverage **0.0458**. The test had 153 eligible users, of whom 76 were new to the training history; 53 of 5,629 positive target items were unavailable under the catalog rule. There were 13 window users without a novel positive target. The split used 79,999 / 10,001 / 10,000 ratings; validation and test catalogs had 1,616 and 1,637 observed items respectively. The fixed cutoff timestamps were 889237269 and 891382309.

The test results by history are user-averaged:

| Test group | Eligible users | Hit-rate@10 | NDCG@10 | Positive targets | Unavailable positives |
|---|---:|---:|---:|---:|---:|
| Cold: no pre-test ratings | 76 | 0.8684 | 0.3324 | 4,076 | 15 |
| Warm: at least one pre-test rating | 77 | 0.4416 | 0.1578 | 1,553 | 38 |

These groups have different numbers of targets per user, which affects hit rate and NDCG. Small groups and shared time effects limit the comparison. Catalog coverage is reported only for the full eligible population; it is a union of recommendations and group coverages cannot be averaged into the overall result.

This retrospective benchmark measures ranking against observed ratings, not causal preference, online engagement, or business impact. A user can have many positive ratings in one window; hit rate is therefore easier to achieve than with a single target. The data has exposure and selection bias, no explicit impression logs, and many cold users. A real deployment needs time-valid item availability, appropriate negative/exposure data, user privacy controls, stronger baselines, repeated time windows, and prospective evaluation.
