# Final discussion

## Final discussion
1. Best generalization on the untouched test set: Linear Regression (RMSE 1.1016).
2. Largest generalization gap: Gradient Boosting (68 trees) (0.0534 CGPA points).
3. Limiting tree depth, selecting complexity with temporal validation, regularization, and leakage-safe preprocessing are the main controls; their effect is visible in the tables above.
4. Training error rewards fitting known examples and cannot measure performance on later unseen responses; validation compares candidate choices on data withheld from fitting.
5. Repeated test-guided choices gradually fit the test set too, turning it into another training signal and biasing the final estimate.
6. The best test R² is 0.204; modest explanatory power indicates the survey captures only part of the determinants of CGPA. A high score can also result from leakage or an unrealistic split, so evaluation design matters as much as the number.

Figures: part_a_cgpa_time.png, part_c_tree_depth.png, part_c_poly_degree.png.
