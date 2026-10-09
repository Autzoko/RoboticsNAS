
### cap=deploy_MB:0.5 (n=58), budget=400 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[kd_onpolicy]  |     0.770 |    0.000 |       0.000 |           0.000 |
| bound_race(M1+M3)        |     0.770 |    0.000 |       0.000 |         400.000 |
| proxy_sh[zc_naswot]_n8   |     0.759 |    0.011 |       0.001 |         384.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.743 |    0.027 |       0.002 |         384.000 |
| proxy_sh[act_l1_exec]_n8 |     0.740 |    0.030 |       0.002 |         384.000 |
| sh_n16                   |     0.729 |    0.041 |       0.002 |         384.000 |
| proxy_sh[fm_loss]_n8     |     0.715 |    0.055 |       0.002 |         384.000 |
| predictor_ridge          |     0.712 |    0.058 |       0.004 |         400.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.065 |       0.000 |           0.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.080 |       0.000 |           0.000 |
| random_full              |     0.614 |    0.156 |       0.007 |         400.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.548 |    0.221 |       0.003 |         384.000 |
| proxy_sh[zc_snip]_n8     |     0.531 |    0.239 |       0.002 |         384.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.520 |    0.250 |       0.000 |           0.000 |

### cap=deploy_MB:0.5 (n=58), budget=800 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[kd_onpolicy]  |     0.770 |    0.000 |       0.000 |           0.000 |
| bound_race(M1+M3)        |     0.770 |    0.000 |       0.000 |         800.000 |
| proxy_sh[zc_naswot]_n8   |     0.766 |    0.004 |       0.001 |         792.000 |
| proxy_sh[act_l1_exec]_n8 |     0.748 |    0.022 |       0.002 |         792.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.738 |    0.032 |       0.002 |         792.000 |
| sh_n16                   |     0.731 |    0.039 |       0.002 |         768.000 |
| predictor_ridge          |     0.724 |    0.046 |       0.002 |         800.000 |
| proxy_sh[fm_loss]_n8     |     0.712 |    0.058 |       0.001 |         792.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.065 |       0.000 |           0.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.080 |       0.000 |           0.000 |
| random_full              |     0.678 |    0.092 |       0.005 |         800.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.545 |    0.225 |       0.003 |         792.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.524 |    0.246 |       0.001 |         792.000 |
| proxy_top1[zc_naswot]    |     0.520 |    0.250 |       0.000 |           0.000 |

### cap=deploy_MB:0.5 (n=58), budget=1600 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[kd_onpolicy]  |     0.770 |    0.000 |       0.000 |           0.000 |
| proxy_sh[zc_naswot]_n8   |     0.770 |    0.000 |       0.000 |        1456.000 |
| bound_race(M1+M3)        |     0.770 |    0.000 |       0.000 |        1600.000 |
| proxy_sh[act_l1_exec]_n8 |     0.758 |    0.012 |       0.001 |        1456.000 |
| sh_n16                   |     0.732 |    0.038 |       0.002 |        1600.000 |
| predictor_ridge          |     0.729 |    0.041 |       0.002 |        1600.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.728 |    0.042 |       0.002 |        1456.000 |
| random_full              |     0.721 |    0.049 |       0.003 |        1600.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.065 |       0.000 |           0.000 |
| proxy_sh[fm_loss]_n8     |     0.697 |    0.073 |       0.001 |        1456.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.080 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.530 |    0.240 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.523 |    0.247 |       0.001 |        1456.000 |
| proxy_sh[zc_snip]_n8     |     0.520 |    0.250 |       0.000 |        1456.000 |
| proxy_top1[zc_naswot]    |     0.520 |    0.250 |       0.000 |           0.000 |

### cap=ms_per_step:0.25 (n=28), budget=400 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_sh[act_l1_exec]_n8 |     0.715 |    0.040 |       0.001 |         384.000 |
| proxy_sh[fm_loss]_n8     |     0.714 |    0.041 |       0.001 |         384.000 |
| proxy_sh[zc_naswot]_n8   |     0.710 |    0.045 |       0.001 |         384.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.710 |    0.045 |       0.001 |         384.000 |
| sh_n16                   |     0.706 |    0.049 |       0.002 |         384.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.050 |       0.000 |           0.000 |
| predictor_ridge          |     0.702 |    0.053 |       0.002 |         400.000 |
| proxy_top1[zc_naswot]    |     0.700 |    0.055 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.696 |    0.059 |       0.001 |         384.000 |
| bound_race(M1+M3)        |     0.687 |    0.068 |       0.002 |         401.000 |
| proxy_top1[kd_onpolicy]  |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.657 |    0.098 |       0.004 |         384.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.120 |       0.000 |           0.000 |
| random_full              |     0.624 |    0.131 |       0.006 |         400.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.200 |       0.000 |           0.000 |

### cap=ms_per_step:0.25 (n=28), budget=800 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_sh[fm_loss]_n8     |     0.711 |    0.044 |       0.001 |         792.000 |
| proxy_sh[act_l1_exec]_n8 |     0.711 |    0.044 |       0.001 |         792.000 |
| sh_n16                   |     0.710 |    0.045 |       0.001 |         768.000 |
| proxy_sh[zc_naswot]_n8   |     0.708 |    0.047 |       0.001 |         792.000 |
| predictor_ridge          |     0.707 |    0.048 |       0.001 |         800.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.706 |    0.049 |       0.001 |         792.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.050 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.700 |    0.055 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.700 |    0.055 |       0.000 |         792.000 |
| bound_race(M1+M3)        |     0.699 |    0.056 |       0.000 |         800.000 |
| random_full              |     0.687 |    0.068 |       0.003 |         800.000 |
| proxy_top1[kd_onpolicy]  |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.679 |    0.076 |       0.001 |         792.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.120 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.200 |       0.000 |           0.000 |

### cap=ms_per_step:0.25 (n=28), budget=1600 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| predictor_ridge          |     0.708 |    0.047 |       0.001 |        1120.000 |
| sh_n16                   |     0.706 |    0.049 |       0.001 |        1600.000 |
| proxy_sh[fm_loss]_n8     |     0.705 |    0.050 |       0.000 |        1456.000 |
| proxy_top1[act_l1_exec]  |     0.705 |    0.050 |       0.000 |           0.000 |
| proxy_sh[act_l1_exec]_n8 |     0.705 |    0.050 |       0.000 |        1456.000 |
| random_full              |     0.705 |    0.050 |       0.002 |        1600.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.702 |    0.053 |       0.000 |        1456.000 |
| proxy_sh[zc_naswot]_n8   |     0.700 |    0.055 |       0.000 |        1456.000 |
| proxy_sh[zc_snip]_n8     |     0.700 |    0.055 |       0.000 |        1456.000 |
| proxy_top1[zc_naswot]    |     0.700 |    0.055 |       0.000 |           0.000 |
| bound_race(M1+M3)        |     0.700 |    0.055 |       0.000 |        1603.000 |
| proxy_top1[kd_onpolicy]  |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.680 |    0.075 |       0.000 |        1456.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.075 |       0.000 |           0.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.120 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.200 |       0.000 |           0.000 |

### cap=ms_per_step:0.5 (n=56), budget=400 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[act_l1_exec]  |     0.845 |    0.000 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.826 |    0.019 |       0.003 |         384.000 |
| proxy_sh[zc_naswot]_n8   |     0.821 |    0.024 |       0.003 |         384.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.821 |    0.024 |       0.002 |         384.000 |
| proxy_sh[act_l1_exec]_n8 |     0.810 |    0.035 |       0.003 |         384.000 |
| bound_race(M1+M3)        |     0.770 |    0.075 |       0.000 |         400.000 |
| sh_n16                   |     0.765 |    0.080 |       0.004 |         384.000 |
| predictor_ridge          |     0.747 |    0.098 |       0.004 |         400.000 |
| proxy_sh[fm_loss]_n8     |     0.733 |    0.112 |       0.002 |         384.000 |
| proxy_top1[kd_onpolicy]  |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.673 |    0.172 |       0.001 |         384.000 |
| random_full              |     0.667 |    0.178 |       0.006 |         400.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.210 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.290 |       0.000 |           0.000 |

### cap=ms_per_step:0.5 (n=56), budget=800 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[act_l1_exec]  |     0.845 |    0.000 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.845 |    0.000 |       0.000 |         792.000 |
| proxy_sh[zc_naswot]_n8   |     0.836 |    0.009 |       0.001 |         792.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.836 |    0.009 |       0.001 |         792.000 |
| proxy_sh[act_l1_exec]_n8 |     0.832 |    0.013 |       0.002 |         792.000 |
| predictor_ridge          |     0.794 |    0.051 |       0.003 |         800.000 |
| sh_n16                   |     0.781 |    0.064 |       0.003 |         768.000 |
| bound_race(M1+M3)        |     0.770 |    0.075 |       0.000 |         800.000 |
| proxy_sh[fm_loss]_n8     |     0.748 |    0.097 |       0.002 |         792.000 |
| random_full              |     0.721 |    0.123 |       0.005 |         800.000 |
| proxy_top1[kd_onpolicy]  |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.680 |    0.165 |       0.000 |         792.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.210 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.290 |       0.000 |           0.000 |

### cap=ms_per_step:0.5 (n=56), budget=1600 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_top1[act_l1_exec]  |     0.845 |    0.000 |       0.000 |           0.000 |
| proxy_sh[zc_snip]_n8     |     0.845 |    0.000 |       0.000 |        1456.000 |
| proxy_sh[zc_naswot]_n8   |     0.845 |    0.000 |       0.000 |        1456.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.844 |    0.001 |       0.000 |        1456.000 |
| proxy_sh[act_l1_exec]_n8 |     0.844 |    0.001 |       0.001 |        1456.000 |
| sh_n16                   |     0.793 |    0.052 |       0.003 |        1600.000 |
| predictor_ridge          |     0.791 |    0.054 |       0.003 |        1600.000 |
| bound_race(M1+M3)        |     0.772 |    0.073 |       0.001 |        1600.000 |
| proxy_sh[fm_loss]_n8     |     0.767 |    0.078 |       0.001 |        1456.000 |
| random_full              |     0.759 |    0.086 |       0.004 |        1600.000 |
| proxy_top1[kd_onpolicy]  |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.685 |    0.160 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.680 |    0.165 |       0.000 |        1456.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| proxy_top1[fm_loss]      |     0.635 |    0.210 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.555 |    0.290 |       0.000 |           0.000 |

### cap=none (n=111), budget=400 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_sh[zc_naswot]_n8   |     0.811 |    0.034 |       0.002 |         384.000 |
| proxy_top1[kd_onpolicy]  |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_sh[act_l1_exec]_n8 |     0.794 |    0.051 |       0.002 |         384.000 |
| proxy_top1[act_l1_exec]  |     0.790 |    0.055 |       0.000 |           0.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.786 |    0.058 |       0.001 |         384.000 |
| bound_race(M1+M3)        |     0.783 |    0.062 |       0.001 |         400.000 |
| sh_n16                   |     0.756 |    0.089 |       0.003 |         384.000 |
| predictor_ridge          |     0.753 |    0.092 |       0.003 |         400.000 |
| proxy_sh[fm_loss]_n8     |     0.715 |    0.130 |       0.002 |         384.000 |
| proxy_sh[zc_snip]_n8     |     0.702 |    0.143 |       0.001 |         384.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.155 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| random_full              |     0.664 |    0.181 |       0.007 |         400.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.656 |    0.189 |       0.002 |         384.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.315 |       0.000 |           0.000 |

### cap=none (n=111), budget=800 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_sh[zc_naswot]_n8   |     0.819 |    0.026 |       0.002 |         792.000 |
| proxy_top1[kd_onpolicy]  |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_top1[act_l1_exec]  |     0.790 |    0.055 |       0.000 |           0.000 |
| proxy_sh[act_l1_exec]_n8 |     0.789 |    0.056 |       0.001 |         792.000 |
| bound_race(M1+M3)        |     0.786 |    0.059 |       0.001 |         800.000 |
| predictor_ridge          |     0.783 |    0.062 |       0.003 |         800.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.780 |    0.065 |       0.001 |         792.000 |
| sh_n16                   |     0.774 |    0.071 |       0.003 |         768.000 |
| random_full              |     0.724 |    0.121 |       0.004 |         800.000 |
| proxy_sh[fm_loss]_n8     |     0.712 |    0.133 |       0.001 |         792.000 |
| proxy_sh[zc_snip]_n8     |     0.705 |    0.140 |       0.000 |         792.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.155 |       0.000 |           0.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.674 |    0.171 |       0.001 |         792.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.315 |       0.000 |           0.000 |

### cap=none (n=111), budget=1600 episodes

| method                   |   sr_eval |   regret |   regret_se |   episodes_used |
|:-------------------------|----------:|---------:|------------:|----------------:|
| proxy_sh[zc_naswot]_n8   |     0.821 |    0.024 |       0.002 |        1456.000 |
| proxy_top1[kd_onpolicy]  |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_top1[zc_naswot]    |     0.800 |    0.045 |       0.000 |           0.000 |
| proxy_top1[act_l1_exec]  |     0.790 |    0.055 |       0.000 |           0.000 |
| bound_race(M1+M3)        |     0.784 |    0.061 |       0.001 |        1600.000 |
| predictor_ridge          |     0.784 |    0.061 |       0.002 |        1600.000 |
| sh_n16                   |     0.779 |    0.066 |       0.003 |        1600.000 |
| proxy_sh[act_l1_exec]_n8 |     0.778 |    0.067 |       0.001 |        1456.000 |
| proxy_sh[kd_onpolicy]_n8 |     0.776 |    0.069 |       0.000 |        1456.000 |
| random_full              |     0.754 |    0.091 |       0.003 |        1600.000 |
| proxy_sh[zc_snip]_n8     |     0.705 |    0.140 |       0.000 |        1456.000 |
| proxy_sh[fm_loss]_n8     |     0.697 |    0.148 |       0.001 |        1456.000 |
| proxy_top1[fm_loss]      |     0.690 |    0.155 |       0.000 |           0.000 |
| proxy_sh[zc_gradnorm]_n8 |     0.680 |    0.165 |       0.000 |        1456.000 |
| proxy_top1[zc_snip]      |     0.680 |    0.165 |       0.000 |           0.000 |
| proxy_top1[zc_gradnorm]  |     0.530 |    0.315 |       0.000 |           0.000 |
