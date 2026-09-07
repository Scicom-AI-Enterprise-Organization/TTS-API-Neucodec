<!-- n=80 paragraphs, langs=['en', 'ms'] -->

### Seam behaviour (median over paragraphs; joins vs interior points of the same clip)

| Condition | joins/clip | f0 step at join (st) | interior (st) | **f0 excess** | level step join (dB) | interior (dB) | **level excess** | gap at join (s) | interior (s) | **gap excess** |
|---|---|---|---|---|---|---|---|---|---|---|
| A single | 5.5 | 4.08 | 1.70 | 2.16 | 5.29 | 2.08 | 3.45 | 0.040 | 0.000 | 0.040 |
| B interleave | 5.5 | 4.76 | 1.60 | 3.12 | 6.18 | 1.96 | 4.46 | 0.068 | 0.000 | 0.068 |
| C cold | 5.5 | 4.50 | 1.75 | 2.62 | 4.09 | 2.09 | 2.12 | 0.030 | 0.000 | 0.030 |

### Direction of the step at a join, and what each chunk ends on (mean over paragraphs)

| Condition | signed f0 step at join (st) | same at interior points (st) | joins that step UP >1 st | signed level step at join (dB) | pitch movement over last 200 ms of a NON-FINAL chunk (st) | silence padding per boundary (s) |
|---|---|---|---|---|---|
| A single | 0.291 | -0.800 | 45% | 5.960 | 0.421 | 0.025 |
| B interleave | 0.113 | -0.905 | 53% | 4.970 | 1.086 | 0.022 |
| C cold | 0.711 | -0.670 | 58% | 3.185 | 0.123 | 0.008 |

### Whole-paragraph prosody (median)

| Condition | duration s | dur vs A | words/s | f0 median Hz | f0 IQR st | declination st/s | active level dB | silence frac | chunk f0 spread st | chunk level SD dB |
|---|---|---|---|---|---|---|---|---|---|---|
| A single | 14.98 | 1.000 | 2.89 | 184.4 | 3.91 | -0.161 | -17.1 | 0.025 | 3.77 | 0.99 |
| B interleave | 15.58 | 1.017 | 2.92 | 186.0 | 3.94 | -0.137 | -17.4 | 0.036 | 3.00 | 0.96 |
| C cold | 15.06 | 1.028 | 2.83 | 192.0 | 4.10 | -0.082 | -16.9 | 0.026 | 3.62 | 1.20 |

### Intelligibility and naturalness

| Condition | CER % | WER % | UTMOSv2 | UTMOSv2 level-matched |
|---|---|---|---|---|
| A single | 0.50 | 2.25 | 3.181 | 2.817 |
| B interleave | 0.44 | 2.13 | 3.180 | 2.776 |
| C cold | 0.58 | 2.11 | 3.226 | 2.835 |

### Paired differences (same paragraph, same chunking)

| Metric | better | B-A median [95% CI] | B wins | C-A median [95% CI] | C wins | B-C median [95% CI] | B wins vs C |
|---|---|---|---|---|---|---|---|
| `excess_f0_st` | lower | 0.738 [0.189, 1.400] | 38% (80) | 0.661 [0.100, 1.226] | 36% (80) | 0.574 [-0.175, 1.073] | 45% (80) |
| `excess_db` | lower | -0.418 [-2.558, 1.385] | 52% (80) | -1.709 [-3.322, -0.407] | 62% (80) | 1.969 [1.421, 2.807] | 22% (80) |
| `excess_gap_s` | lower | 0.0300 [0.0200, 0.0450] | 22% (80) | 0.0000 [-0.0100, 0.0125] | 48% (80) | 0.0300 [0.0200, 0.0450] | 15% (80) |
| `join_f0_st_signed` | lower | -0.108 [-1.358, 1.119] | 51% (80) | -0.020 [-0.600, 0.923] | 50% (80) | -0.706 [-1.899, 0.675] | 55% (80) |
| `join_reset` | lower | 0.100 [-0.083, 0.200] | 42% (80) | 0.100 [0.000, 0.167] | 32% (80) | 0.000 [-0.095, 0.000] | 46% (80) |
| `nonfinal_pad_s` | lower | 0.0000 [-0.0048, 0.0050] | 49% (80) | -0.0123 [-0.0170, -0.0085] | 79% (80) | 0.0104 [0.0069, 0.0140] | 21% (80) |
| `dur_ratio` | lower | 0.0172 [0.0010, 0.0354] | 38% (80) | 0.0279 [0.0106, 0.0424] | 36% (80) | 0.0051 [-0.0163, 0.0149] | 45% (80) |
| `seg_f0_spread_st` | lower | -0.650 [-1.000, -0.150] | 65% (80) | 0.175 [-0.375, 0.500] | 44% (80) | -0.575 [-0.950, -0.200] | 65% (80) |
| `seg_db_std` | lower | -0.119 [-0.189, 0.069] | 55% (80) | 0.292 [-0.012, 0.348] | 40% (80) | -0.207 [-0.447, -0.065] | 71% (80) |
| `f0_slope_st_per_s` | lower | 0.0129 [-0.0022, 0.0556] | 39% (80) | 0.0645 [0.0327, 0.0802] | 28% (80) | -0.0623 [-0.0826, -0.0109] | 64% (80) |
| `cer` | lower | 0.0000 [0.0000, 0.0000] | 28% (80) | 0.0000 [0.0000, 0.0000] | 25% (80) | 0.0000 [0.0000, 0.0000] | 30% (80) |
| `mos_levelmatched` | higher | -0.101 [-0.161, -0.020] | 38% (80) | 0.023 [-0.060, 0.112] | 54% (80) | -0.048 [-0.216, 0.033] | 42% (80) |
| `mos` | higher | -0.024 [-0.057, 0.023] | 45% (80) | 0.040 [-0.028, 0.078] | 57% (80) | -0.052 [-0.086, 0.002] | 40% (80) |

### Chunk N+1 against chunk N (whole-chunk medians)

| Condition | signed register step (st) | \|register step\| (st) | signed level step (dB) | \|level step\| (dB) | steps up >1 st | n |
|---|---|---|---|---|---|---|
| A single | -0.372 | 1.604 | -0.235 | 1.348 | 24% | 438 |
| B interleave | -0.274 | 1.241 | -0.222 | 1.157 | 20% | 438 |
| C cold | -0.182 | 1.648 | -0.159 | 1.608 | 28% | 438 |

| Metric | pair | signed diff [95% CI] | \|step\| diff [95% CI] | n |
|---|---|---|---|---|
| register step (st) | B-C | -0.091 [-0.298, 0.115] | -0.407 [-0.556, -0.261] | 438 |
| register step (st) | B-A | 0.098 [-0.088, 0.279] | -0.363 [-0.501, -0.226] | 438 |
| register step (st) | C-A | 0.190 [-0.024, 0.395] | 0.044 [-0.100, 0.189] | 438 |
| level step (dB) | B-C | -0.063 [-0.263, 0.139] | -0.451 [-0.586, -0.320] | 438 |
| level step (dB) | B-A | 0.013 [-0.150, 0.176] | -0.191 [-0.309, -0.069] | 438 |
| level step (dB) | C-A | 0.076 [-0.150, 0.296] | 0.260 [0.119, 0.407] | 438 |

### Per-join, pooled over every chunk boundary

| Metric | A single | B interleave | C cold | B-C paired mean [95% CI] | B closer to A |
|---|---|---|---|---|---|
| signed f0 step (st) | 0.426 | -0.353 | 0.529 | -0.923 [-1.585, -0.283] | no |
| |f0 step| (st) | 4.357 | 5.253 | 4.737 | +0.558 [+0.115, +1.008] | no |
| steps up >1 st | 0.449 | 0.501 | 0.563 | -0.065 [-0.120, -0.012] | **yes** |
| signed level step (dB) | 5.812 | 4.896 | 3.023 | +1.873 [+1.059, +2.698] | **yes** |
| |level step| (dB) | 9.439 | 7.563 | 4.947 | +2.616 [+2.021, +3.217] | **yes** |
| silence at the join (s) | 0.0510 | 0.0731 | 0.0427 | +0.0304 [+0.0245, +0.0363] | no |

438 matched B/C joins; A has 438 inferred ones.

### LM behaviour and cost

| Condition | requests | tokens/word | collapsed chunks | finish=length | median prompt tok | median LM latency s |
|---|---|---|---|---|---|---|
| A single | 80 | 17.3 | 0 (0.0%) | 0 | 80 | 2.762 |
| B interleave | 518 | 18.1 | 0 (0.0%) | 0 | 418 | 0.423 |
| C cold | 518 | 18.2 | 0 (0.0%) | 0 | 19 | 0.427 |

### Per language (median excess at joins / chunk level SD / CER %)

| Lang | A single | B interleave | C cold |
|---|---|---|---|
| en f0 excess st | 1.550 | 3.149 | 2.924 |
| en gap excess s | 0.0400 | 0.0800 | 0.0300 |
| en chunk level SD dB | 0.955 | 0.973 | 1.331 |
| en CER % | 0.25 | 0.00 | 0.00 |
| ms f0 excess st | 2.827 | 3.048 | 2.238 |
| ms gap excess s | 0.0325 | 0.0575 | 0.0350 |
| ms chunk level SD dB | 1.066 | 0.954 | 1.136 |
| ms CER % | 0.68 | 0.79 | 0.70 |

