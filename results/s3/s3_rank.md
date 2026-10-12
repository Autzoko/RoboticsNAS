# S3: SR@10k vs SR@30k (standalone, s2-h10, search-val, 400 paired episodes, n=15)

Kendall -0.020 [95% CI -0.202, 0.295] | Spearman -0.050 | 30k top-3 within 10k top-5: False | mean SR 10k 0.733 vs 30k 0.836
Reference: test-retest Kendall of the 30k ranking ~0.56; supernet V0..V4 0.08 to -0.03.
**Pre-registered gate (tau >= 0.40 and top-3 in top-5): FAIL**

| arch | SR@10k | SR@30k | supernet V0 | ms/step | deploy MB |
|---|---|---|---|---|---|
| v12-e4-stretch-f1-t16-s2-h10 | 0.718 | 0.880 | 0.440 | 6.9 | 616 |
| v8-e16-stretch-f0.75-t16-s2-h10 | 0.782 | 0.880 | 0.682 | 11.3 | 654 |
| v24-e16-top-f0.5-t64-s2-h10 | 0.733 | 0.863 | 0.718 | 14.7 | 925 |
| v16-e8-top-f1-t16-s2-h10 | 0.755 | 0.863 | 0.705 | 9.4 | 786 |
| v24-e16-stretch-f0.5-t16-s2-h10 | 0.752 | 0.853 | 0.713 | 14.7 | 920 |
| v12-e4-top-f0.75-t16-s2-h10 | 0.730 | 0.848 | 0.580 | 7.0 | 580 |
| v16-e16-stretch-f1-t64-s2-h10 | 0.760 | 0.843 | 0.833 | 12.7 | 980 |
| v24-e12-top-f1-t64-s2-h10 | 0.705 | 0.840 | 0.610 | 12.8 | 1036 |
| v8-e4-stretch-f0.5-t64-s2-h10 | 0.690 | 0.833 | 0.735 | 6.1 | 497 |
| v24-e8-stretch-f1-t16-s2-h10 | 0.745 | 0.833 | 0.642 | 11.2 | 937 |
| v16-e12-top-f0.75-t16-s2-h10 | 0.765 | 0.823 | 0.752 | 11.2 | 752 |
| v8-e12-stretch-f0.75-t64-s2-h10 | 0.792 | 0.823 | 0.605 | 9.5 | 603 |
| v12-e8-top-f0.75-t64-s2-h10 | 0.760 | 0.818 | 0.718 | 8.7 | 628 |
| v20-e8-top-f1-t64-s2-h10 | 0.762 | 0.790 | 0.613 | 10.3 | 865 |
| v24-e4-top-f0.75-t64-s2-h10 | 0.540 | 0.752 | 0.147 | 9.5 | 806 |

Selection rule 'cheapest (ms/step) arch tied with the best' applied to each ranking source; the pick is scored by its 30k SR:

| ranking source | pick | tied set size | pick SR@30k | regret vs 30k best | ms/step | deploy MB |
|---|---|---|---|---|---|---|
| 30k standalone (oracle, 3x cost) | v12-e4-stretch-f1-t16-s2-h10 | 6 | 0.880 | 0.000 | 6.9 | 616 |
| 10k standalone (two-stage) | v12-e8-top-f0.75-t64-s2-h10 | 7 | 0.818 | 0.062 | 8.7 | 628 |
| supernet V0 (weight sharing) | v16-e16-stretch-f1-t64-s2-h10 | 1 | 0.843 | 0.037 | 12.7 | 980 |
