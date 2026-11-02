# Speaker notes (7 slides, 5 min: about 40 s each)

**1. Title (10 s).** We kept the generator fixed and studied one training change, spectral normalization, and one inference change, latent truncation, which we compare with rejection sampling.

**2. Baseline (40 s).** Our baseline is the starter code, unchanged. Every configuration runs with three seeds, and we compute the metrics as the platform does; on the same checkpoint we get 31.4, the platform 30.6. The baseline already has a problem: one seed out of three collapses at epoch 2. The two others reach FID 29.5, precision 0.49, recall 0.24.

**3. Methods (50 s).** Spectral normalization divides every layer of the discriminator by its largest singular value. We use it as in the paper: hinge loss, Adam beta1 0.5, five D steps for each G step. Our hypothesis: more stable training and better recall. At inference, truncation resamples the latent coordinates larger than psi, for free. DRS keeps a sample with a probability given by the discriminator; with K = 5 it costs five times more samples. Hypothesis: both raise precision, only truncation loses recall.

**4. Training (60 s).** One bar per configuration, one dot per seed. The paper's optimisation setting alone already removes the collapse and raises recall from 0.24 to 0.29. Adding SN with one D step makes things worse. With five D steps it catches up, 27.6, but for the same number of G steps it takes 57 minutes instead of 9; at equal time it is far behind. With the starter loss, SN does not learn at all. So H1: stability yes, but SN adds no recall.

**5. Inference (60 s).** Same two generators, only the sampling changes. Truncation with psi = 1 raises precision from 0.52 to 0.65 for both. Without SN, recall drops to 0.21. With SN, it stays at 0.29, on all three seeds. DRS gives a small gain for five times the samples. So H2 is rejected here: free truncation beats rejection.

**6. Result and limits (40 s).** Overall, from the starter to SN plus truncation: FID 29.5 to 20.5, precision 0.49 to 0.65, recall 0.24 to 0.29. The recall comes from the training setting, the precision from truncation, which SN makes safe. Limits: three seeds, psi chosen on the reported metrics, five D steps never tested without SN, and no explanation yet of why SN helps truncation.

**7. Next steps (30 s).** First the missing ablation without SN. Then TTUR instead of five D steps, OBRS against truncation at equal cost, and a proper selection of psi.
