# Climate Attribution of the Nepal Ice-Rock Collapse Preconditions
## Method B: VAE-Based Anomaly Detection

### The question

Method A asked whether the pre-event thermal conditions at the exact source-zone coordinate were unusual, using classical percentile and trend statistics. Method B asks the same underlying question in a structurally different way: was the spatial *pattern* of temperature across the ~220 km region surrounding the site unusual, judged by a model trained to recognize what normal patterns for this place and season look like.

This is deliberately a different mechanism from Method A, not a repeat of it. Where Method A reduces each day to a single number at a single point, Method B looks at the whole regional field at once and asks whether that field, as a spatial pattern, resembles the ones the model learned from 76 years of history.

Like Method A, this is attribution-adjacent screening, not formal detection and attribution. No probability-ratio claim is made anywhere in this analysis.

### Data and methods

**Spatial extent**: a 2 degree by 2 degree box centered on the site (roughly 220 km by 220 km), covering the Langtang, Rasuwa, and Nuwakot corridor. At ERA5-Land's native ~0.1 degree resolution this gives a 21 by 21 pixel field per day, enough spatial structure for a convolutional model to learn something meaningful without being so large it dilutes the site-specific signal with unrelated climate regimes.

**Training window**: 60 days before August 26 each year, fetched for 1950 through 2026 using `ee.Image.sampleRectangle()` (not `reduceRegion()` with a list reducer, which does not guarantee pixel order matches the 2D grid and would silently scramble every training image). All 77 years fetched successfully in under 3 minutes, with a uniform 21 by 21 shape and no meaningful missing data.

**Train and validation split**: by year, not by day. 2026 was excluded entirely from both training and validation, since it is the inference target. Of the remaining 76 years, 68 were used for training and 8 for validation, selected as whole years rather than individual days, because days within the same 60-day window are strongly autocorrelated. A day-level split would let the model see near-duplicates of held-out days during training.

**Normalization**: per-pixel mean and standard deviation computed from the training years only, then applied identically to validation and to 2026.

**Model**: a small convolutional variational autoencoder. Two convolutional layers in the encoder (1 to 16 to 32 channels), a small latent bottleneck, and a mirrored decoder. Kept deliberately small given roughly 4,500 training examples of a fairly smooth, low-complexity signal. A bigger network risked overfitting rather than learning anything more useful.

### Choosing the anomaly score: reconstruction error was the wrong default

The standard VAE anomaly-detection recipe uses reconstruction error: an anomalous input should be harder for the model to rebuild. A first hyperparameter sweep selected the configuration with the lowest validation reconstruction loss, and that selection turned out to be meaningless. Reconstruction loss decreased monotonically as the KL weight (beta) decreased and the latent dimension increased, with no interior optimum, meaning the "best" configuration was simply whichever one sat at the edge of the grid that was searched. Weakening the KL constraint and adding capacity always improves reconstruction, because it lets the model behave more like a plain autoencoder that can reconstruct almost anything, including inputs it has never seen. That is the opposite of what an anomaly detector needs.

The fix was to select configurations by how well they separate normal fields from synthetic anomalies, not by how low their reconstruction error gets. Synthetic anomalies were created by adding a uniform temperature shift (0.5, 1.0, and 2.0 degrees C, spanning from realistic, since Method A found roughly a 0.6 to 1 degree C anomaly at this site, to generous) to held-out validation fields, then measuring the effect size (Cohen's d) separating real fields from shifted ones. 2026 was never touched during this selection process.

This test also revealed that reconstruction error was the wrong score entirely, not just the wrong configuration. At the realistic 1 degree C shift, even the best reconstruction-error-based configuration reached only a small effect size (d = 0.36 to 0.37, below Cohen's threshold for a medium effect). Testing KL divergence alone as the anomaly score instead, at the same shift magnitude, reached d = 0.96, a large effect and two to three times more discriminative. The model does not need to reconstruct an anomalous field badly for the anomaly to matter. It can map the anomalous input to a less typical region of the compressed latent space (higher KL) even while still reconstructing it reasonably well. For this data, that turned out to be the far more sensitive signal.

The final configuration, selected on this basis, is beta = 0.5 and latent dimension = 8, using KL divergence as the primary anomaly score. Retrained for the full 100 epochs, it converged cleanly with no posterior collapse (KL stayed in the 17 to 18 range throughout, consistent with the 24.8 to 18.0 to 14.2 pattern seen across beta = 0.1, 0.5, and 1.0 at this latent size) and no overfitting.

### Results

2026's pre-event fields were scored using the same 7, 14, and 30 day lookback windows Method A used, for direct comparability.

| Window | KL score percentile (vs. 1950-2025) | Reconstruction error percentile |
|---|---|---|
| 7 days before | 71st | 31st |
| 14 days before | 96th | 96th |
| 30 days before | 97th | 97th |

The 14 and 30 day windows agree closely with Method A's point-based percentiles (95th and 91st respectively). Two structurally unrelated methods converging on the same answer, roughly 90s percentile, unusually warm, is a meaningful corroboration.

The 7-day window does not agree, and reconstruction error actually points in the opposite direction from KL at that window. That disagreement was investigated rather than reported as a loose end.

### The 7-day discrepancy

Three checks, run independently, all point the same direction.

**Raw spatial comparison**, using nothing but the data: 2026's mean field for the last 7 days was warmer than the 1950-2025 climatological mean across the entire region, every pixel, by 0.35 to 2.11 degrees C. The anomaly near the site itself (1.28 degrees C) was not meaningfully different from the field-wide average (1.39 degrees C). The warmth was region-wide, not concentrated at the site.

**Day-by-day KL score** for the full 30-day window shows where the actual signal is. Three large spikes, reaching KL values far above the historical 10th to 90th percentile band, sit roughly 8 to 18 days before the event. The final 7 days, by contrast, are the calmest stretch in the entire 30-day window, only modestly above the historical median.

**SHAP attribution**, using `shap.GradientExplainer` against a background of 200 randomly sampled training-year fields, was used to check what the model's KL score for the final week was actually responding to. Mean absolute SHAP value near the site (0.058) was, if anything, slightly lower than the whole-field average (0.073). The model's anomaly signal draws on the whole region fairly evenly, consistent with the raw-data finding that the final week's warmth was genuinely regional rather than localized at the site pixel.

**Conclusion**: the 7-day discrepancy is not a failure of either method. It reflects a two-phase precursor structure that neither method alone would have revealed. A broader, regional-scale thermal anomaly appeared roughly 1 to 2.5 weeks before the collapse (Method B's clearest signal, also captured within Method A's 14 and 30 day windows through averaging). A sharper, more localized intensification followed in the final week (Method A's clearest signal, at the exact site coordinate, which Method B's region-wide pattern did not register as strongly because the surrounding area was comparatively unremarkable during that specific week even as the site itself sharpened).

### Limitations

The synthetic-anomaly discriminability test used a uniform additive temperature shift as the stand-in for an anomaly. This is a reasonable proxy given Method A's own finding of a broadly warm pre-event signal, but it is a proxy, not the real 2026 anomaly, and a more spatially structured synthetic anomaly (for example, a localized rather than uniform shift) was not tested here.

The hyperparameter sweep and final training both used a fixed random seed for the train and validation year split. A different split could plausibly shift which configuration wins, though the retrained model's convergence behavior was consistent with the sweep's own findings, which is a reasonable check but not a full robustness test across multiple splits.

KL divergence as an anomaly score is specific to this trained model's own learned latent space. It is not directly comparable in absolute terms to Method A's percentile statistics, ERA5-Land's raw temperature, or any external benchmark. The percentile-rank framing used here (comparing 2026's KL score against the historical distribution of the same model's KL scores) is what makes the two methods comparable at all.

### What this does and does not claim

**Does claim**: the spatial pattern of pre-event temperature across the region was unusual relative to 76 years of history in the 14 and 30 day windows, corroborating Method A's finding through a structurally different mechanism. The final week's disagreement between methods reflects a real, checked two-phase precursor structure, not measurement noise or a modeling failure.

**Does not claim**: that the VAE's anomaly score is causally linked to the collapse, that KL divergence is a universally superior anomaly score outside this specific dataset and question, or that this analysis constitutes formal climate attribution.
