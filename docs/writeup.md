# Climate Attribution of the Nepal Ice-Rock Collapse Preconditions
## Combined Findings

### The question

On August 26, 2026, an ice-rock avalanche collapsed from the north face of Langtang Lirung on the Nepal-Tibet border, triggering a flood down the Lehende Khola that killed hundreds of people. This project asks whether the thermal conditions in the weeks before the collapse were unusual for this exact location, and whether that fits a real long-term warming trend at the site.

It is screening, not attribution. There is no probability-ratio claim anywhere in this project, and no claim that climate change caused this specific collapse. That kind of formal detection-and-attribution work requires a full counterfactual large-ensemble study, well beyond the scope of this analysis. What follows is a smaller, more honestly bounded question, answered two structurally different ways so that agreement between them means something and disagreement gets investigated rather than hidden.

Full detail on each method is in [`method_a_writeup.md`](method_a_writeup.md) and [`method_b_writeup.md`](method_b_writeup.md). This document is the synthesis.

### Two methods, briefly

**Method A** is classical statistics on ERA5-Land reanalysis at the exact site coordinate: percentile rank of the pre-event window against 76 years of history, and a Mann-Kendall and Theil-Sen trend, chosen over ordinary least squares after OLS residuals showed autocorrelation that would have overstated the trend's significance. Checked for spatial robustness across a five-point surrounding grid and cross-checked against an independent satellite sensor.

**Method B** is a convolutional variational autoencoder trained on 76 years of ERA5-Land temperature fields across the ~220 km region around the site, learning what a normal spatial pattern looks like for this place and season. The anomaly score is KL divergence, not the more standard reconstruction error, a choice made after testing showed KL divergence discriminated synthetic anomalies two to three times better for this data.

### Where they agree

| Window | Method A percentile | Method B percentile (KL) |
|---|---|---|
| 7 days before | 99th | 71st |
| 14 days before | 95th | 96th |
| 30 days before | 91st | 97th |

At 14 and 30 days, two methods built on different data representations, different mathematics, and different implicit assumptions about what an anomaly looks like, converge on the same answer: the pre-event period was unusually warm, roughly in the 90s percentile against 76 years of history. That kind of agreement between structurally unrelated methods is a stronger basis for the finding than either method alone would provide.

The long-term trend adds context rather than a separate finding: +0.070 degrees C per decade (95% confidence interval 0.02 to 0.12, Mann-Kendall p = 0.0043), real and statistically significant but modest, explaining only a fraction of year-to-year variance. 2026 was not the warmest year on record at this site. Six years since 1950 were warmer, the closest being 1960.

### Where they disagree, and why that matters more than a clean agreement would

The 7-day window is where the two methods genuinely diverge (99th percentile versus 71st), and that disagreement was chased down rather than smoothed over or quietly dropped from the writeup.

Three independent checks, a raw spatial comparison against climatology, the full 30-day day-by-day KL trajectory, and SHAP attribution of the model's own anomaly score, all point to the same explanation. The most extreme regional signal in the entire 30-day window actually occurred 8 to 18 days before the collapse, not in the final week. The final week itself was a real but comparatively mild region-wide warm anomaly, while Method A's point-based measurement at the exact site coordinate found its sharpest signal specifically in that final week.

Read together, this suggests a two-phase precursor structure: a broader regional thermal anomaly roughly one to two and a half weeks before the collapse, followed by a sharper, more localized intensification in the final days. Neither method alone would have revealed this. Method A has no spatial resolution to see the regional pattern building beforehand. Method B's window-averaging smoothed over how sharp the final week's localized signal was at the exact site. The disagreement between the two methods is not a flaw in either one. It is the two methods being sensitive to genuinely different things, and that difference turned out to be informative once it was taken seriously instead of resolved by picking whichever number was more convenient.

### What the data cannot tell us

No permafrost monitoring station exists anywhere near this site. This was checked directly against the Global Terrestrial Network for Permafrost's own published CALM and TSP station metadata, not assumed. Zero of 242 CALM sites and zero of 1,076 TSP boreholes fall within the regional bounding box. Since the actual destabilizing mechanism at a site like this is almost certainly permafrost or ground-ice temperature rather than air temperature, both methods here are limited to atmospheric and surface proxies. That is a real, structural limitation of what this kind of analysis can say, not a detail to gloss over.

A single ERA5-Land grid cell's absolute temperature is not trustworthy in this terrain. Demonstrated three separate times: a 30-meter DEM sample within 200 meters of the site varied by over 150 meters in elevation, the five-point spatial grid used for Method A's robustness check showed absolute temperatures varying by more than 10 degrees C over just 11 km due to real elevation heterogeneity at that grid spacing, and one grid point (directly east of the site) was 6.3 degrees C colder than the center point despite sitting at essentially the same elevation, a discrepancy that land-cover and snow-cover checks ruled out as a glacier or snow effect without fully resolving. For this reason, every comparison across locations in this project uses relative statistics (percentile rank, trend) rather than absolute temperature.

### What this project does and does not claim

**Does claim**: the thermal and spatial-pattern conditions preceding the August 26, 2026 collapse were anomalous relative to 76 years of record at this location, most clearly in the 14 and 30 day windows where two independent methods agree, and this occurred against a real, spatially consistent, though modest, long-term warming trend at the site. The final week shows a more complex, two-phase pattern that emerged from investigating a disagreement between methods rather than picking one result and ignoring the other.

**Does not claim**: that climate change caused this collapse, that the warming trend measured here matches the true local rate (reanalysis resolution is a real limit on that), or that the destabilizing mechanism has been directly measured. No permafrost data exists at this site, and none of the methods used here are a substitute for it.
