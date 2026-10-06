# E2 proxy analysis (bench_sn)

archs with complete grid: 111; cells/arch: 400 (search 200, eval 200)
SR: mean 0.573, min 0.085, max 0.845
split-half reliability (search vs eval SR): Spearman 0.962, Kendall 0.840

| signal | Kendall tau vs SR_eval [95% CI] | Spearman | top-10 overlap |
|---|---|---|---|
| sr_search | 0.840 [0.804, 0.873] | 0.962 | 8/10 |
| cl_2ep | 0.819 [0.769, 0.860] | 0.945 | 6/10 |
| cl_1ep | 0.775 [0.726, 0.818] | 0.922 | 6/10 |
| kd_offline | 0.756 [0.697, 0.811] | 0.915 | 8/10 |
| kd_onpolicy | 0.738 [0.670, 0.796] | 0.902 | 8/10 |
| act_l1_exec | 0.667 [0.605, 0.726] | 0.858 | 4/10 |
| act_l1 | 0.538 [0.445, 0.621] | 0.729 | 0/10 |
| self_cons | 0.427 [0.320, 0.535] | 0.580 | 3/10 |
| grip_err | 0.169 [0.046, 0.299] | 0.225 | 0/10 |
| fm_loss | 0.165 [0.005, 0.330] | 0.137 | 0/10 |
| ms_per_step | -0.061 [-0.178, 0.060] | -0.083 | 0/10 |
| call_ms | -0.132 [-0.258, -0.007] | -0.218 | 0/10 |
| expert_params_M | -0.516 [-0.612, -0.408] | -0.684 | 0/10 |
