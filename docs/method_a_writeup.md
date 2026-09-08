# Climate Attribution of the Nepal Ice-Rock Collapse Preconditions
## Method A: Trend-Based Screening

### The question

On August 26, 2026, an ice-rock avalanche collapsed from the north face of
Langtang Lirung (28.285 N, 85.525 E, source zone ~5100-5200 m), triggering a
catastrophic flood down the Lehende Khola. This analysis asks a narrow,
deliberately bounded question: **were the thermal conditions in the weeks
before the collapse anomalous relative to the historical record for this
exact location, and is that anomaly consistent with long-term regional
warming?**

This is *not* a formal climate attribution study. It does not estimate a
probability ratio, and it does not claim climate change caused this
collapse. That would require a full counterfactual large-ensemble study —
the kind of analysis Sarah Sparrow's climateprediction.net platform at
Oxford performs for precipitation-triggered landslides in Brazil and the
Andes. This project asks a narrower, attribution-adjacent question with a
different, complementary methodology (AI/ML- and statistics-based
screening, applied to a thermal/permafrost destabilization mechanism rather
than precipitation).

### Data and methods

**Primary data**: ERA5-Land 2m air temperature, accessed via Google Earth
Engine, 1950-2026. Verified accessible before any analysis began, including
confirming the finalized product's actual latency (initially assumed to lag
2-3 months based on ECMWF documentation; empirically found to be current to
within 1 day of the event for this dataset).

**Historical baseline**: for each year 1950-2025, the mean 2m temperature
was computed over three lookback windows (7, 14, and 30 days) ending
immediately before that year's August 26. 2026 was computed identically and
compared against the 1950-2025 distribution using a percentile-rank
statistic (plotting-position formula, n+1 denominator, to avoid ever
reporting an artificially exact 0th/100th percentile from a finite sample).

**Trend estimation**: Mann-Kendall significance testing with Theil-Sen slope
estimation, not ordinary least squares. OLS was tested first and found to
have autocorrelated residuals (Durbin-Watson = 1.26, lag-1 autocorrelation
= 0.365), meaning its p-value would overstate confidence. Mann-Kendall/
Theil-Sen is distribution-free, robust to outliers, and is the standard
method in the comparable published High Mountain Asia literature.

### Results

**The pre-event window was anomalously warm, with the anomaly sharpest in
the most recent days before collapse:**

| Lookback window | 2026 value | Percentile rank (vs. 1950-2025) |
|---|---|---|
| 7 days | 9.4°C | 99th |
| 14 days | 9.4°C | 95th |
| 30 days | 9.3°C | 91st |

The percentile rank *decreases* as the window widens — consistent with a
short, sharp thermal spike immediately preceding the collapse, rather than
a month-long warm spell that happened to end in a disaster. This is a more
specific and more informative finding than "it was warm."

2026 was not the warmest year on record at this site: six years exceeded
it (1960, 1966, 1982, 2020, 2023, 2024), the closest being 1960 (9.56°C).
The 91st-percentile framing is deliberately reported instead of "hottest
ever," since the latter would not be true.

**Long-term trend**: +0.070°C/decade (95% CI: 0.02 to 0.12), Mann-Kendall
p = 0.0043. Statistically significant but modest — the trend is real but
does not explain most of the year-to-year variance, which is dominated by
interannual noise.

**Context for the trend magnitude**: this is lower than a directly
comparable ERA5-Land basin-scale study in the region (+0.15°C/decade), and
below commonly-cited High Mountain Asia elevation-dependent-warming rates.
Two plausible, non-exclusive explanations, both grounded in the published
literature rather than asserted: (1) reanalysis products are known to
underestimate warming trends at high elevations relative to station
observations; (2) elevation-dependent warming is not monotonic in this
region — cooling has been documented above 6000 m in the Himalaya during
2002-2017, and some Tibetan Plateau studies find warming *decreasing* with
altitude rather than amplifying.

### Robustness checks

**Spatial**: the finding was re-tested at 4 additional ERA5-Land grid
points (~11 km spacing) surrounding the site. Absolute temperatures varied
enormously across points (3.0°C to 13.2°C) — traced to genuine elevation
differences at this grid spacing in this terrain (confirmed via DEM
sampling), not reanalysis noise. Because of this, percentile rank and
trend — not raw temperature — are the valid cross-cell comparison
statistics. On both: **all 5 grid points ranked 2026 in the warm half of
their own history (76.6-94.8 percentile), and all 5 showed a positive,
statistically significant warming trend (0.067-0.103°C/decade)**.

One anomaly surfaced and was investigated rather than left unexplained: the
grid point directly east of the site sits at essentially the same
elevation as the center point (4877 m vs. 4859 m) but is 6.3°C colder — too
large a gap for elevation to explain. A land-cover and snow-cover check
ruled out the most obvious hypothesis (glacier/snow surface): the east
point has the same land-cover class as center and a *lower* snow fraction.
The most likely remaining explanation is that ERA5-Land's internal model
orography (smoothed to its ~9-11 km grid) diverges from fine-resolution DEM
elevation at that specific cell — a known reanalysis limitation in extreme
terrain, not fully confirmed here. This is reported as an open question,
not resolved into a false certainty.

**Independent sensor (MODIS LST)**: air temperature (ERA5-Land) was
cross-checked against land surface temperature from an independent
satellite sensor. Coverage is limited by cloud masking at this
high-elevation, monsoon-affected site — 9/30 valid days for 2026 using
Terra day-pass only (30% coverage), improving to 18/30 (60%) when Terra and
Aqua day and night passes are combined. Both versions independently rank
2026 in the warm tail of their own history (74th and 89th percentile
respectively), though the two are not numerically comparable to each other
or to ERA5-Land — LST has a much larger diurnal range than 2m air
temperature, and the day+night combination is a physically blended
quantity. The honest reading: MODIS qualitatively supports the direction of
the finding across two independent pooling methods, but neither should be
cited as a precise quantitative confirmation given sparse and heterogeneous
coverage.

**Broader climate context**: of the 6 historical years warmer than 2026 at
this site, 4 coincide with recognized global temperature extremes — the
1982-83 El Nino (one of the strongest of the 20th century) and the 2020,
2023, 2024 sequence of consecutive global temperature records (2020 tied
for warmest year on record without a major El Nino boost; 2023 then set a
new record; 2024 surpassed it, becoming the first year to clearly exceed
1.5°C above pre-industrial levels). This is not treated as independent
evidence of a causal link — local and global temperature are not
independent, so this is exactly what physical expectation predicts, not a
coincidence being mined for significance — but it is a mechanistically
grounded corroboration of the trend finding, distinct from and stronger
than a search for coincident regional disasters (which was also attempted
and found a weaker, more ambiguous pattern, appropriately not relied upon
here).

### Data gaps and honest limitations

- **No permafrost monitoring exists near this site.** The Global
  Terrestrial Network for Permafrost's own published site metadata (CALM
  active-layer network and TSP boreholes, checked directly, not assumed)
  shows zero stations within the regional bounding box. Since the likely
  actual destabilizing mechanism is permafrost or ground-ice temperature
  rather than air temperature, this analysis is limited to atmospheric/
  surface proxies — a real, stated limitation, not a detail glossed over.
- **Single-cell absolute values are not reliable in this terrain**; only
  relative (percentile, trend) statistics are used for cross-location
  comparison, for the reasons demonstrated in the spatial robustness
  section.
- **This is attribution-adjacent screening, not formal detection-and-
  attribution.** No probability-ratio claim is made anywhere in this
  analysis.

### What this analysis does and does not claim

**Does claim**: the thermal conditions preceding the August 26, 2026
collapse were anomalous relative to 76 years of record at this location,
most sharply in the days immediately before the event; this occurred
against a real, spatially consistent, though modest, long-term warming
trend at the site; and the pattern is corroborated, though not proven, by
an independent satellite sensor and by co-occurrence with globally
exceptional warm years.

**Does not claim**: that climate change caused this collapse, that the
warming trend at this site matches the true local rate (reanalysis
resolution limits this), or that the destabilizing mechanism has been
directly measured (no permafrost data exists here).
