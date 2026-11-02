# Training ablation (vanilla sampling)

Mean ± std over the non-collapsed seeds (3 seeds per configuration). FID on 10 000 images (Inception), precision and recall on 9 984 images (VGG-16, k = 5), as on the course platform.

| Training | G updates | Time (min) | Collapsed | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|
| Starter baseline (BCE, β1 = 0.9) | 94k | 8 | 1/3 | 29.5 ± 2.8 | 0.491 ± 0.012 | 0.239 ± 0.005 |
| Paper setting (hinge, β1 = 0.5) | 94k | 9 | 0/3 | 28.1 ± 3.1 | 0.514 ± 0.017 | 0.288 ± 0.024 |
| + SN | 94k | 16 | 0/3 | 52.7 ± 9.4 | 0.461 ± 0.013 | 0.172 ± 0.088 |
| + SN, 5 D steps (equal time) | 19k | 12 | 0/3 | 60.7 ± 2.3 | 0.399 ± 0.007 | 0.162 ± 0.017 |
| + SN, 5 D steps (equal G updates) | 94k | 57 | 0/3 | 27.6 ± 1.7 | 0.516 ± 0.002 | 0.293 ± 0.017 |

Starter setting (same budget as the starter baseline):

| Training | G updates | Time (min) | Collapsed | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|
| Starter setting + SN | 94k | 16 | 0/3 | 176.5 ± 6.2 | 0.063 ± 0.037 | 0.002 ± 0.002 |
| Starter setting + TTUR | 94k | 9 | 2/3 | 51.7 | 0.528 | 0.182 |
| Starter setting + SN + TTUR | 94k | 17 | 0/3 | 122.5 ± 2.8 | 0.218 ± 0.013 | 0.017 ± 0.009 |

# Inference (same generator, 10 000 images)

Acceptance = accepted / proposed samples (cost = 1 / acceptance generator passes). The recalibrated D is retrained for 3 epochs with G frozen.

| Model | Sampling | Acceptance | Time / 10k (s) | FID ↓ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|
| Paper setting, no SN | Vanilla | 1.00 | 0.01 | 28.1 ± 3.1 | 0.514 ± 0.017 | 0.288 ± 0.024 |
| Paper setting, no SN | DRS final D (K = 5) | 0.20 | 0.11 | 27.7 ± 2.6 | 0.509 ± 0.006 | 0.253 ± 0.008 |
| Paper setting, no SN | DRS recal. D (K = 5) | 0.20 | 0.05 | 25.3 ± 4.5 | 0.543 ± 0.012 | 0.252 ± 0.025 |
| Paper setting, no SN | DRS recal. D (K = 10) | 0.10 | 0.08 | 25.5 ± 5.4 | 0.543 ± 0.018 | 0.257 ± 0.024 |
| Paper setting, no SN | Trunc. ψ = 1.5 | 1.00 | 0.04 | 23.7 ± 3.2 | 0.586 ± 0.011 | 0.247 ± 0.001 |
| Paper setting, no SN | Trunc. ψ = 1.0 | 1.00 | 0.02 | 24.8 ± 2.5 | 0.646 ± 0.024 | 0.206 ± 0.012 |
| SN, 5 D steps | Vanilla | 1.00 | 0.01 | 27.6 ± 1.7 | 0.516 ± 0.002 | 0.293 ± 0.017 |
| SN, 5 D steps | DRS final D (K = 5) | 0.20 | 0.06 | 27.9 ± 4.3 | 0.522 ± 0.008 | 0.291 ± 0.033 |
| SN, 5 D steps | DRS recal. D (K = 5) | 0.20 | 0.05 | 26.0 ± 0.7 | 0.518 ± 0.006 | 0.296 ± 0.026 |
| SN, 5 D steps | DRS recal. D (K = 10) | 0.10 | 0.08 | 25.6 ± 0.9 | 0.515 ± 0.009 | 0.288 ± 0.018 |
| SN, 5 D steps | Trunc. ψ = 1.5 | 1.00 | 0.01 | 21.1 ± 2.3 | 0.583 ± 0.004 | 0.303 ± 0.024 |
| SN, 5 D steps | Trunc. ψ = 1.0 | 1.00 | 0.02 | 20.5 ± 3.0 | 0.653 ± 0.005 | 0.288 ± 0.021 |
