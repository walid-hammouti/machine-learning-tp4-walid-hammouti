# Part C: diagnosing overfitting

## Part C: diagnosing overfitting
| Tree depth | Train RMSE | Test RMSE | Gap |
|---|---:|---:|---:|---|
| 1 | 1.2231 | 1.1744 | -0.0487 |
| 2 | 1.1849 | 1.1458 | -0.0391 |
| 3 | 1.1553 | 1.1341 | -0.0213 |
| 5 | 1.0760 | 1.1523 | 0.0763 |
| 10 | 0.6975 | 1.9967 | 1.2992 |
| 20 | 0.0516 | 2.0559 | 2.0044 |
| None | 0.0000 | 2.0676 | 2.0676 |

The tree train error falls as depth increases, while test error is minimized at depth 3; overfitting becomes evident as the gap expands and test error stops improving. See part_c_tree_depth.png.

| Polynomial degree | 5-fold temporal CV RMSE |
|---:|---:|
| 1 | 1.1850 |
| 2 | 1.1849 |
| 3 | 1.1861 |
| 5 | 1.1925 |
| 9 | 1.2001 |
| 15 | 1.2001 |

Polynomial degree changes a one-dimensional smooth feature representation, so its effect is constrained compared with a tree that partitions a dozen features and can isolate individual observations. See part_c_poly_degree.png.


