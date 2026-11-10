# Part A: inspection

## Part A: inspection
- Input row order chronological: False; exact duplicates: 0; excluding student_id: 0; ignoring timestamp: 0.
- Cohort categories with fewer than 10 rows: 12 of 42.
- Missing values (columns with any): daily_productivity (34, 2.8%), revision_frequency (34, 2.8%), focus_duration (43, 3.6%), screen_time_non_study (32, 2.7%), energy_level (28, 2.3%), stress_level (35, 2.9%), routine_rating (41, 3.4%), sleep_hours (55, 4.6%), online_courses (36, 3.0%), programming_foundation (41, 3.4%), events_participation (58, 4.8%), external_resources (45, 3.8%), external_pressure (29, 2.4%).
- Numerical association with target, strongest positive: attendance_rate (0.525); weakest magnitude: sleep_hours (-0.010).
- Risk label vs CGPA category cross-tab:

cgpa_category           5.0 – 6.9  7.0 – 8.4  8.5 – 9.4  9.5 – 10.0
performance_risk_level                                             
High Risk                     188        221         39           4
Low Risk                        0          4         89          83
Moderate Risk                  48        195        234          95

The risk label must not be used: it is a derived label closely tied to the same self-reported academic outcome and can directly reveal target information. The CGPA time plot is in part_a_cgpa_time.png; its 7-day mean stays broadly stable near 8.0, with only a modest early-April rise and no abrupt drift.

Risks: (1) target-derived performance_risk_level can leak CGPA; (2) duplicates can place the same response in both partitions; (3) random splits can mix earlier and later survey periods; (4) rare cohort categories encourage memorization; (5) flexible trees/polynomials can fit noise in a small, noisy survey.


