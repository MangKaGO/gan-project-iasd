Collapsed runs (final mean D(fake) < 0.01): baseline_seed1, ttur_seed1

## Training lever, converged runs only (epoch 100, vanilla sampling, mean ± std)

| Config | Runs | Collapsed | FID ↓ | Precision ↑ | Recall ↑ | Train time / run |
|---|---|---|---|---|---|---|
| Baseline | 2 | 1/3 | 29.5 ± 2.8 | 0.491 ± 0.012 | 0.239 ± 0.005 | 8.3 min |
| Spectral norm | 3 | 0/3 | 176.5 ± 6.2 | 0.063 ± 0.037 | 0.002 ± 0.002 | 16.0 min |
| TTUR | 2 | 1/3 | 133.2 ± 115.2 | 0.305 ± 0.315 | 0.091 ± 0.129 | 8.5 min |
| SN + TTUR | 3 | 0/3 | 122.5 ± 2.8 | 0.218 ± 0.013 | 0.017 ± 0.009 | 16.7 min |

## Training lever, all runs (epoch 100, vanilla sampling, mean ± std)

| Config | Runs | Collapsed | FID ↓ | Precision ↑ | Recall ↑ | Train time / run |
|---|---|---|---|---|---|---|
| Baseline | 3 | 1/3 | 143.0 ± 196.7 | 0.327 ± 0.284 | 0.160 ± 0.138 | 8.3 min |
| Spectral norm | 3 | 0/3 | 176.5 ± 6.2 | 0.063 ± 0.037 | 0.002 ± 0.002 | 16.0 min |
| TTUR | 3 | 1/3 | 195.6 ± 135.3 | 0.203 ± 0.284 | 0.061 ± 0.105 | 8.5 min |
| SN + TTUR | 3 | 0/3 | 122.5 ± 2.8 | 0.218 ± 0.013 | 0.017 ± 0.009 | 16.7 min |

## Inference lever, converged runs only (final checkpoints, mean ± std over seeds)

| Generator | Sampler | D | K | ψ | FID ↓ | Precision ↑ | Recall ↑ | Acceptance | Time (s) |
|---|---|---|---|---|---|---|---|---|---|
| Baseline | drs | calibrated | 2 |  | 23.5 ± 0.1 | 0.527 ± 0.009 | 0.228 ± 0.005 | 0.493 | 0.0 |
| Baseline | drs | calibrated | 5 |  | 22.9 ± 2.0 | 0.542 ± 0.015 | 0.237 ± 0.008 | 0.195 | 0.1 |
| Baseline | drs | calibrated | 10 |  | 23.4 ± 2.2 | 0.554 ± 0.022 | 0.238 ± 0.009 | 0.097 | 0.1 |
| Baseline | drs | final | 2 |  | 28.0 ± 1.6 | 0.504 ± 0.010 | 0.235 ± 0.031 | 0.506 | 0.2 |
| Baseline | drs | final | 5 |  | 26.6 ± 1.5 | 0.502 ± 0.014 | 0.243 ± 0.029 | 0.198 | 0.1 |
| Baseline | drs | final | 10 |  | 26.2 ± 1.8 | 0.508 ± 0.015 | 0.235 ± 0.033 | 0.100 | 0.1 |
| Baseline | truncation | none | 1 | 0.5 | 71.7 ± 7.0 | 0.799 ± 0.028 | 0.023 ± 0.001 | 1.000 | 0.0 |
| Baseline | truncation | none | 1 | 1.0 | 28.4 ± 0.1 | 0.672 ± 0.007 | 0.134 ± 0.011 | 1.000 | 0.0 |
| Baseline | truncation | none | 1 | 1.5 | 22.8 ± 1.7 | 0.595 ± 0.010 | 0.218 ± 0.024 | 1.000 | 0.1 |
| Baseline | vanilla | none | 1 |  | 29.5 ± 2.8 | 0.491 ± 0.012 | 0.239 ± 0.005 | 1.000 | 0.0 |
| SN + TTUR | drs | calibrated | 2 |  | 118.9 ± 1.8 | 0.231 ± 0.009 | 0.017 ± 0.011 | 0.508 | 0.0 |
| SN + TTUR | drs | calibrated | 5 |  | 117.5 ± 2.1 | 0.231 ± 0.010 | 0.017 ± 0.009 | 0.201 | 0.1 |
| SN + TTUR | drs | calibrated | 10 |  | 117.2 ± 1.5 | 0.230 ± 0.011 | 0.020 ± 0.006 | 0.101 | 0.1 |
| SN + TTUR | drs | final | 2 |  | 121.4 ± 3.3 | 0.223 ± 0.015 | 0.016 ± 0.010 | 0.511 | 0.0 |
| SN + TTUR | drs | final | 5 |  | 121.4 ± 3.4 | 0.223 ± 0.018 | 0.017 ± 0.008 | 0.200 | 0.1 |
| SN + TTUR | drs | final | 10 |  | 121.2 ± 3.1 | 0.222 ± 0.020 | 0.019 ± 0.012 | 0.101 | 0.1 |
| SN + TTUR | truncation | none | 1 | 0.5 | 169.7 ± 9.6 | 0.555 ± 0.250 | 0.000 ± 0.000 | 1.000 | 0.0 |
| SN + TTUR | truncation | none | 1 | 1.0 | 128.3 ± 15.2 | 0.403 ± 0.029 | 0.012 ± 0.008 | 1.000 | 0.0 |
| SN + TTUR | truncation | none | 1 | 1.5 | 119.4 ± 6.9 | 0.317 ± 0.022 | 0.018 ± 0.010 | 1.000 | 0.0 |
| SN + TTUR | vanilla | none | 1 |  | 122.5 ± 2.8 | 0.218 ± 0.013 | 0.017 ± 0.009 | 1.000 | 0.0 |

