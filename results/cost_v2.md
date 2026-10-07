# Cost table v2 — latency and memory (A100-PCIE-40GB, batch 1, eager bf16)

Source: `results/cost/a100_v2.jsonl` (`rnas/latency.py`, job s2_cost). weights_MB = bf16 size of parameters the subnet
actually uses (vision encoder + connector, embeddings, VLM layers read, sliced expert, projections); act. peak MB =
measured peak extra GPU memory during one policy call; deploy MB = weights + activation peak. v1 latencies were
measured on A100-80GB (e.g. published config 374 ms there vs 396 ms here).

## E4 official-test candidates

| candidate | arch | official SR | ms/call | ms/control step | active params (M) | VLM (M) | expert (M) | weights MB (bf16) | act. peak MB | deploy MB |
|---|---|---|---|---|---|---|---|---|---|---|
| smolvla_default_published | `v16-e16-stretch-f1-t64-s10-h50` | 0.674 | 396 | 7.9 | 403 | 205 | 99.9 | 768 | 212 | 980 |
| smolvla_default_tuned_schedule | `v16-e16-stretch-f1-t64-s2-h10` | 0.810 | 127 | 12.7 | 403 | 205 | 99.9 | 768 | 212 | 980 |
| ours_proxySH_kd_cap | `v16-e16-stretch-f1-t64-s2-h10` | 0.821 | 127 | 12.7 | 403 | 205 | 99.9 | 768 | 212 | 980 |
| random_search_cap_ft5k | `v16-e16-stretch-f1-t64-s10-h50` | 0.662 | 396 | 7.9 | 403 | 205 | 99.9 | 768 | 212 | 980 |
| nas_valloss_top1_cap | `v16-e8-stretch-f1-t16-s4-h50` | 0.701 | 130 | 2.6 | 354 | 205 | 50.8 | 674 | 111 | 786 |
| standalone_race_best_small | `v12-e4-stretch-f1-t16-s4-h5` | 0.823 | 88 | 17.7 | 290 | 165 | 26.2 | 553 | 64 | 616 |

## 16 S1 networks, standalone, schedule s2-h10 (search-val SR, 400 eps; '-' = not yet trained/evaluated)

| arch | search-val SR | ms/call | ms/control step | active params (M) | VLM (M) | expert (M) | weights MB (bf16) | act. peak MB | deploy MB |
|---|---|---|---|---|---|---|---|---|---|
| `v12-e4-stretch-f1-t16-s2-h10` | 0.880 | 69 | 6.9 | 290 | 165 | 26.2 | 553 | 64 | 616 |
| `v16-e12-top-f0.5-t16-s2-h10` | 0.875 | 112 | 11.2 | 352 | 205 | 48.8 | 671 | 56 | 726 |
| `v16-e8-top-f1-t16-s2-h10` | 0.863 | 94 | 9.4 | 354 | 205 | 50.8 | 674 | 111 | 786 |
| `v24-e16-top-f0.5-t64-s2-h10` | 0.863 | 147 | 14.7 | 446 | 283 | 64.5 | 851 | 74 | 925 |
| `v12-e4-top-f0.75-t16-s2-h10` | 0.848 | 70 | 7.0 | 285 | 165 | 21.8 | 544 | 36 | 580 |
| `v16-e16-stretch-f1-t64-s2-h10` | 0.843 | 127 | 12.7 | 403 | 205 | 99.9 | 768 | 212 | 980 |
| `v8-e4-stretch-f0.5-t64-s2-h10` | 0.833 | 61 | 6.1 | 242 | 126 | 17.3 | 461 | 36 | 497 |
| `v24-e8-stretch-f1-t16-s2-h10` | 0.833 | 112 | 11.2 | 432 | 283 | 50.8 | 824 | 112 | 937 |
| `v8-e12-stretch-f0.75-t64-s2-h10` | 0.823 | 95 | 9.5 | 286 | 126 | 62.0 | 546 | 57 | 603 |
| `v12-e8-top-f0.75-t64-s2-h10` | 0.818 | 87 | 8.7 | 305 | 165 | 41.9 | 583 | 45 | 628 |
| `v24-e4-top-f0.75-t64-s2-h10` | 0.752 | 95 | 9.5 | 403 | 283 | 21.8 | 769 | 36 | 806 |
| `v20-e8-top-f1-t64-s2-h10` | - | 103 | 10.3 | 393 | 244 | 50.8 | 749 | 116 | 865 |
| `v24-e12-top-f1-t64-s2-h10` | - | 128 | 12.8 | 457 | 283 | 75.3 | 871 | 165 | 1036 |
| `v16-e12-top-f0.75-t16-s2-h10` | - | 112 | 11.2 | 365 | 205 | 62.0 | 696 | 56 | 752 |
| `v24-e16-stretch-f0.5-t16-s2-h10` | - | 147 | 14.7 | 446 | 283 | 64.5 | 851 | 70 | 920 |
| `v8-e16-stretch-f0.75-t16-s2-h10` | - | 113 | 11.3 | 306 | 126 | 82.2 | 584 | 69 | 654 |

## Key comparison (search-val SR; official test not yet run for the s2-h10 small net)

| | published v16-e16 s10-h50 | default net s2-h10 | **v12-e4-stretch-f1-t16 s2-h10** |
|---|---|---|---|
| search-val SR (standalone) | 0.713 | 0.843 | **0.880** |
| ms per call | 396 | 127 | **69** (5.7x) |
| ms per control step | 7.9 | 12.7 | **6.9** |
| active params | 403M | 403M | **290M** (-28%) |
| expert params | 99.9M | 99.9M | **26.2M** (-74%) |
| activation peak | 212 MB | 212 MB | **64 MB** (-70%) |
| deploy memory | 980 MB | 980 MB | **616 MB** (-37%) |
