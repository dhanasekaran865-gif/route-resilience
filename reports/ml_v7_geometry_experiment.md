# ML V7 — Collinear Continuity + Local Grid Geometry Experiment Report

## 1. Frozen Baseline Reproduction

- Candidates Generated: 3760
- Validated Candidates: 664
- TP: 20 | FP: 11 | FN: 32
- Precision: 0.645 | Recall: 0.385 | F1: 0.482

## 2. Primary Metrics Comparison Table

| Variant | Feats | TP | FP | FN | Precision | Recall | F1 | Overall Rec | ML-Stage Rec | Non-Disc Rec | Disc Rec |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V7-A (V4-C Baseline)** | 20 | 20 | 11 | 32 | 0.645 | 0.385 | **0.482** | 20/52 (38.5%) | 20/37 (54.1%) | 7/22 (31.8%) | 13/15 (86.7%) |
| **V7-B (+ Continuity u/v)** | 22 | 19 | 14 | 33 | 0.576 | 0.365 | **0.447** | 19/52 (36.5%) | 19/37 (51.4%) | 7/22 (31.8%) | 12/15 (80.0%) |
| **V7-C (+ Straight & Collinear)** | 22 | 19 | 12 | 33 | 0.613 | 0.365 | **0.458** | 19/52 (36.5%) | 19/37 (51.4%) | 7/22 (31.8%) | 12/15 (80.0%) |
| **V7-D (+ Stub Align & Grid Consistency)** | 22 | 18 | 11 | 34 | 0.621 | 0.346 | **0.444** | 18/52 (34.6%) | 18/37 (48.6%) | 6/22 (27.3%) | 12/15 (80.0%) |
| **V7-E (+ All V7 Geometry Features)** | 30 | 18 | 7 | 34 | 0.720 | 0.346 | **0.468** | 18/52 (34.6%) | 18/37 (48.6%) | 6/22 (27.3%) | 12/15 (80.0%) |

## 3. Redundant-Grid Subgroup Recovery (`is_disconnected = 0`)

| Variant | Overall Non-Disc | Detour < 6 | Detour >= 6 | Deficit <= 0 | Deficit > 0 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **V7-A (V4-C Baseline)** | 7/22 (31.8%) | 0/7 (0.0%) | 7/15 (46.7%) | 0/9 (0.0%) | 7/13 (53.8%) |
| **V7-B (+ Continuity u/v)** | 7/22 (31.8%) | 0/7 (0.0%) | 7/15 (46.7%) | 0/9 (0.0%) | 7/13 (53.8%) |
| **V7-C (+ Straight & Collinear)** | 7/22 (31.8%) | 0/7 (0.0%) | 7/15 (46.7%) | 0/9 (0.0%) | 7/13 (53.8%) |
| **V7-D (+ Stub Align & Grid Consistency)** | 6/22 (27.3%) | 0/7 (0.0%) | 6/15 (40.0%) | 0/9 (0.0%) | 6/13 (46.2%) |
| **V7-E (+ All V7 Geometry Features)** | 6/22 (27.3%) | 0/7 (0.0%) | 6/15 (40.0%) | 0/9 (0.0%) | 6/13 (46.2%) |

## 4. Length Subgroup Recovery (Edges Reaching ML Stage)

| Variant | <25m (n=13) | 25-50m (n=11) | 50-75m (n=9) | 75-90m (n=4) |
| :--- | :---: | :---: | :---: | :---: |
| **V7-A (V4-C Baseline)** | 11/13 (84.6%) | 4/11 (36.4%) | 5/9 (55.6%) | 0/4 (0.0%) |
| **V7-B (+ Continuity u/v)** | 11/13 (84.6%) | 4/11 (36.4%) | 4/9 (44.4%) | 0/4 (0.0%) |
| **V7-C (+ Straight & Collinear)** | 11/13 (84.6%) | 4/11 (36.4%) | 4/9 (44.4%) | 0/4 (0.0%) |
| **V7-D (+ Stub Align & Grid Consistency)** | 11/13 (84.6%) | 4/11 (36.4%) | 3/9 (33.3%) | 0/4 (0.0%) |
| **V7-E (+ All V7 Geometry Features)** | 11/13 (84.6%) | 4/11 (36.4%) | 3/9 (33.3%) | 0/4 (0.0%) |

## 5. Feature Importance in V7-E

| Rank | Feature | Type | Importance |
| :---: | :--- | :---: | :---: |
| 1 | `parallel_grid_alignment` | New Geometry | 0.1411 |
| 2 | `detour_ratio` | V4-C Base | 0.1292 |
| 3 | `has_reverse_edge` | V4-C Base | 0.0923 |
| 4 | `degree_deficit` | V4-C Base | 0.0814 |
| 5 | `is_disconnected` | V4-C Base | 0.0737 |
| 6 | `v1_structural_score` | V4-C Base | 0.0595 |
| 7 | `shortest_path_dist` | V4-C Base | 0.0556 |
| 8 | `mean_bearing_diff` | V4-C Base | 0.0522 |
| 9 | `bearing_diff_u` | V4-C Base | 0.0491 |
| 10 | `candidate_length` | V4-C Base | 0.0448 |
| 11 | `bearing_diff_v` | V4-C Base | 0.0308 |
| 12 | `local_angle_residual` | New Geometry | 0.0191 |
| 13 | `degree_sum` | V4-C Base | 0.0190 |
| 14 | `endpoint_heading_continuity_v` | New Geometry | 0.0138 |
| 15 | `common_neighbors` | V4-C Base | 0.0135 |
| 16 | `opposite_heading_continuity` | New Geometry | 0.0132 |
| 17 | `stub_alignment_score` | New Geometry | 0.0114 |
| 18 | `jaccard_coeff` | V4-C Base | 0.0113 |
| 19 | `endpoint_heading_continuity_u` | New Geometry | 0.0110 |
| 20 | `street_count_v` | V4-C Base | 0.0104 |
| 21 | `straight_continuation_score` | New Geometry | 0.0103 |
| 22 | `street_count_u` | V4-C Base | 0.0098 |
| 23 | `collinearity_score` | New Geometry | 0.0093 |
| 24 | `degree_u` | V4-C Base | 0.0092 |
| 25 | `local_direction_consistency` | New Geometry | 0.0073 |
| 26 | `street_count_diff` | V4-C Base | 0.0057 |
| 27 | `degree_v` | V4-C Base | 0.0056 |
| 28 | `endpoint_continuation_count` | New Geometry | 0.0048 |
| 29 | `is_deadend_u` | V4-C Base | 0.0035 |
| 30 | `is_deadend_v` | V4-C Base | 0.0021 |

## 6. Tracking the 17 V6 ML False Negatives in V7-E

| # | Edge | Length | V6 Prob | V7-E Prob | Diff | Disc | Detour | Deficit | Outcome |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | `256794571 -> 256794570` | 25.2m | 0.724 | 0.659 | -0.065 | 0 | 3.4 | +2.0 | Still Missed |
| 02 | `10616173 -> 9476442` | 48.6m | 0.710 | 0.696 | -0.014 | 0 | 5.4 | +1.0 | Still Missed |
| 03 | `333650005 -> 107317` | 48.6m | 0.685 | 0.717 | +0.031 | 0 | 5.4 | +2.0 | Still Missed |
| 04 | `26652117 -> 26652119` | 50.3m | 0.685 | 0.677 | -0.008 | 1 | 9.9 | +0.0 | Still Missed |
| 05 | `33141178 -> 6863550501` | 54.9m | 0.670 | 0.687 | +0.017 | 0 | 24.1 | +0.0 | Still Missed |
| 06 | `5630569824 -> 14791190` | 12.5m | 0.627 | 0.630 | +0.003 | 0 | 73.5 | +0.0 | Still Missed |
| 07 | `107737 -> 6277683849` | 28.9m | 0.607 | 0.627 | +0.020 | 0 | 14.4 | +1.0 | Still Missed |
| 08 | `25504199 -> 25504193` | 76.6m | 0.516 | 0.398 | -0.118 | 1 | 6.5 | -1.0 | Still Missed |
| 09 | `4973447303 -> 4973447307` | 28.7m | 0.504 | 0.430 | -0.075 | 0 | 27.1 | +0.0 | Still Missed |
| 10 | `2725915874 -> 2725915891` | 9.1m | 0.503 | 0.437 | -0.066 | 0 | 6.6 | +1.0 | Still Missed |
| 11 | `255578 -> 496812960` | 84.2m | 0.480 | 0.551 | +0.071 | 0 | 5.9 | +1.0 | Still Missed |
| 12 | `4973447307 -> 4973447303` | 30.6m | 0.422 | 0.401 | -0.021 | 0 | 40.1 | +0.0 | Still Missed |
| 13 | `21704021 -> 21990641` | 77.8m | 0.374 | 0.254 | -0.120 | 0 | 11.4 | -2.0 | Still Missed |
| 14 | `21665713 -> 9512917` | 66.3m | 0.360 | 0.397 | +0.037 | 0 | 12.2 | +0.0 | Still Missed |
| 15 | `21990641 -> 21704021` | 77.8m | 0.260 | 0.160 | -0.100 | 0 | 3.0 | -2.0 | Still Missed |
| 16 | `9512917 -> 21665713` | 66.3m | 0.239 | 0.249 | +0.010 | 0 | 2.5 | +0.0 | Still Missed |
| 17 | `1645813888 -> 1139318692` | 28.3m | 0.142 | 0.180 | +0.038 | 0 | 1.6 | +0.0 | Still Missed |

## 7. Answers to Experiment Questions

**A. Did geometric continuity improve overall F1?**
- Baseline V7-A: F1 = 0.482 (P=0.645, R=0.385)
- Full V7-E: F1 = 0.468 (P=0.720, R=0.346)

**B. Did it improve non-disconnecting/redundant-grid recovery?**
- Baseline non-disconnecting recovery: 7/22 (31.8%)
- V7-E non-disconnecting recovery: 6/22 (27.3%)

**C. Which geometric feature contributed the most measurable signal?**
- `parallel_grid_alignment` ranked #1 with importance 0.1411.

**D. How many of the 17 V6 ML-stage FNs were recovered?**
- Exactly 0 of 17 edges recovered.

**E. Did false positives increase?**
- Baseline FPs: 11 -> V7-E FPs: 7 (Delta: -4).

**F. Which error class remains after adding geometry?**
- Low detour ratio (< 6) grid edges with degree deficit <= 0, and edges with length >= 75m.

**G. What should the next controlled experiment be?**
- Systematic analysis of Decision Threshold and Ensembling / Probability Calibration on the validated space.
