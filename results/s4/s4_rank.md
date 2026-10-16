# S4: teacher-imitation @10k vs plain @30k (n=8, 400 paired search-val episodes, s2-h10)

| ranking source | Kendall vs plain@30k [95% CI] | Spearman | mean SR | SR range | SR std across nets |
|---|---|---|---|---|---|
| distill@10k (S4) | 0.222 [-0.255, 0.619] | 0.452 | 0.785 | 0.760-0.812 | 0.016 |
| plain@10k (S3) | 0.109 [-0.113, 0.500] | 0.168 | 0.725 | 0.540-0.792 | 0.077 |
| plain@30k (S1, target) | 1 | 1 | 0.833 | 0.752-0.880 | 0.042 |

Kendall difference distill - plain@10k: +0.113 [95% CI -0.503, +0.486]. Per-net binomial SE at 400 eps ~0.018.
**Pre-registered gate (tau >= 0.40 and >= plain@10k + 0.2): FAIL**

| arch | distill@10k | plain@10k | plain@30k | ms/step |
|---|---|---|---|---|
| v12-e4-stretch-f1-t16-s2-h10 | 0.777 | 0.718 | 0.880 | 6.9 |
| v8-e16-stretch-f0.75-t16-s2-h10 | 0.792 | 0.782 | 0.880 | 11.3 |
| v16-e8-top-f1-t16-s2-h10 | 0.797 | 0.755 | 0.863 | 9.4 |
| v16-e16-stretch-f1-t64-s2-h10 | 0.812 | 0.760 | 0.843 | 12.7 |
| v8-e4-stretch-f0.5-t64-s2-h10 | 0.760 | 0.690 | 0.833 | 6.1 |
| v8-e12-stretch-f0.75-t64-s2-h10 | 0.792 | 0.792 | 0.823 | 9.5 |
| v20-e8-top-f1-t64-s2-h10 | 0.770 | 0.762 | 0.790 | 10.3 |
| v24-e4-top-f0.75-t64-s2-h10 | 0.775 | 0.540 | 0.752 | 9.5 |

Selection rule 'cheapest (ms/step) tied with the best', scored by plain@30k SR:

| ranking source | pick | tied set | pick SR@30k | regret | ms/step |
|---|---|---|---|---|---|
| plain@30k (oracle) | v12-e4-stretch-f1-t16-s2-h10 | 3 | 0.880 | 0.000 | 6.9 |
| distill@10k | v12-e4-stretch-f1-t16-s2-h10 | 6 | 0.880 | 0.000 | 6.9 |
| plain@10k | v16-e8-top-f1-t16-s2-h10 | 5 | 0.863 | 0.017 | 9.4 |
