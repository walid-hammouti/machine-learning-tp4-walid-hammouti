# Part D: generalization improvements

## Part D: generalization improvements
### D.1 Cohort encoding (5-fold time-series validation)
| Treatment | CV RMSE |
|---|---:|
| remove | 1.3705 |
| rare_group | 1.3719 |
| target_encode | 1.3640 |

The lowest validation RMSE indicates the preferred treatment. A small improvement over removal means cohort_code adds little reliable predictive information; target encoding is cross-fitted to avoid using each row's target in its own encoding.

### D.2 Submission day
| Model | Feature choice | CV RMSE |
|---|---|---:|
| Linear Regression | with day | 1.1688 |
| Linear Regression | without day | 1.1629 |
| Decision Tree depth 5 | with day | 1.3705 |
| Decision Tree depth 5 | without day | 1.3106 |
| Gradient Boosting | with day | 1.1790 |
| Gradient Boosting | without day | 1.1600 |

All three models had lower validation RMSE without submission_day, so it is excluded from the final feature matrix. Since all future test dates lie past the training range, a linear day effect can extrapolate poorly and tree models cannot extrapolate the trend.

### D.3 Regularization
| Model | Selected alpha | CV RMSE | Lasso zero coefficients |
|---|---:|---:|---:|
| Linear | none | 1.1925 | - |
| Ridge | 10 | 1.1855 | - |
| Lasso | 0.01 | 1.1848 | 5 |

Ridge shrinks all polynomial coefficients continuously; Lasso can set coefficients exactly to zero. The unregularized degree-5 model is more sensitive to noisy curvature.

### D.4 Final untouched test evaluation
Feature set for final comparison: study_hours, sleep_hours, attendance_rate, study_consistency, tasks_on_time, assignments_on_time, revision_frequency, career_goal_clarity, daily_productivity, stress_level, age plus cohort_code with cross-fitted target encoding. Submission day is excluded because all three D.2 comparisons favored exclusion. Tree depth chosen by temporal CV: 2; Ridge alpha: 100; boosting trees selected on the chronological validation block: 68.

| Model | Train RMSE | Test RMSE | Test R² |
|---|---:|---:|---:|---|
| Baseline | 1.2685 | 1.2362 | -0.0029 |
| Linear Regression | 1.1252 | 1.1016 | 0.2036 |
| Ridge alpha=100 | 1.1297 | 1.1026 | 0.2022 |
| Decision Tree depth=2 | 1.1849 | 1.1458 | 0.1384 |
| Gradient Boosting (68 trees) | 1.0516 | 1.1050 | 0.1987 |


