# Part B: evaluation and leakage

## Part B: evaluation and leakage
| Strategy | Train RMSE | Test RMSE |
|---|---:|---:|---|
| B1 random split | 0.0000 | 1.5579 |
| B1 random split baseline (training mean) | 1.2671 | 1.2426 |
| B1 chronological split | 0.0000 | 2.0676 |
| B1 chronological split baseline (training mean) | 1.2685 | 1.2362 |
| B2 duplicates retained | 0.0000 | 1.5906 |
| B2 duplicate rows removed | 0.0000 | 1.7591 |
| B3 scaler fit on all rows | 1.1001 | 1.1858 |
| B3 scaler inside training pipeline | 1.0931 | 1.1904 |

Random splitting mixes dates, so it can hide temporal distribution shift and usually gives an optimistic estimate for deployment on later submissions. In B.2, 180 copies were injected and 47 answer-identical records cross the chronological boundary when timestamps are ignored. The retained-duplicate evaluation can reuse answers across train and test, making test performance optimistic; deduplication before splitting removes these repeated responses. In B.3, whole-data scaling lets the test distribution influence learned means and variances; the gap can be small, but imputation and scaling must be fitted on training data to keep evaluation honest.


