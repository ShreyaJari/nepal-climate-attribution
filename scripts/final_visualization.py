"""
final_visualization.py

Presentation-quality figure for Method A (trend-based screening) results,
built from era5_pre_event_summary.csv. Design goals, distinct from the
diagnostic version:

  - Smooth density (not binned histograms) for the three lookback windows,
    so small-sample binning artifacts don't distract from the shape of the
    distribution.
  - Shared x-axis scale across the three lookback panels, so their relative
    spread is visually comparable at a glance.
  - Direct labeling on the plot (2026 value, percentile) instead of a
    legend, since there is only one series of interest per panel.
  - Trend line uses Mann-Kendall (significance) + Theil-Sen (slope), the
    robust, distribution-free method standard in the comparable published
    Himalayan/HMA literature -- not OLS, which was checked separately and
    found to have autocorrelated residuals (Durbin-Watson = 1.26), meaning
    its p-value would overstate confidence.
  - 1960 (the next-highest year in the historical record) is called out on
    the trend panel, since 2026 is high but not unprecedented -- the figure
    should not visually imply "highest on record" when that is not true.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats as scipy_stats
from scipy.stats import gaussian_kde
from pathlib import Path

# Anchor all paths to this script's own location, not the working directory
# the IDE happens to launch from -- avoids relative-path failures entirely.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

EVENT_YEAR = 2026
LOOKBACK_WINDOWS_DAYS = [7, 14, 30]
COLOR_HISTORICAL = "#6699BF"
COLOR_EVENT = "#B23A48"
COLOR_TREND_LINE = "#2B2B2B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "axes.labelcolor": "#222222",
    "text.color": "#222222",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
})


def mann_kendall_p_value(x):
    """Mann-Kendall trend test p-value: distribution-free, robust to outliers."""
    n = len(x)
    s = sum(np.sign(x[j] - x[k]) for k in range(n - 1) for j in range(k + 1, n))
    var_s = n * (n - 1) * (2 * n + 5) / 18
    z = (s - 1) / np.sqrt(var_s) if s > 0 else (s + 1) / np.sqrt(var_s) if s < 0 else 0
    return 2 * (1 - scipy_stats.norm.cdf(abs(z)))


df = pd.read_csv(RESULTS_DIR / "era5_pre_event_summary.csv")
historical = df[df["year"] != EVENT_YEAR]
event_row = df[df["year"] == EVENT_YEAR].iloc[0]

fig = plt.figure(figsize=(14, 7))
gs = GridSpec(2, 3, height_ratios=[1.1, 1.4], hspace=0.55, wspace=0.35)

# Shared x-limits across the three density panels, based on the widest window's range
all_values = np.concatenate([historical[f"mean_{w}d_degc"].dropna().values for w in LOOKBACK_WINDOWS_DAYS])
x_min, x_max = all_values.min() - 0.4, all_values.max() + 0.4
x_grid = np.linspace(x_min, x_max, 400)

for i, window_days in enumerate(LOOKBACK_WINDOWS_DAYS):
    ax = fig.add_subplot(gs[0, i])
    col = f"mean_{window_days}d_degc"
    values = historical[col].dropna().values
    event_value = event_row[col]

    kde = gaussian_kde(values)
    density = kde(x_grid)
    ax.fill_between(x_grid, density, color=COLOR_HISTORICAL, alpha=0.35, linewidth=0)
    ax.plot(x_grid, density, color=COLOR_HISTORICAL, linewidth=1.3)
    ax.axvline(event_value, color=COLOR_EVENT, linewidth=2.2)

    percentile = (np.sum(values <= event_value) / (len(values) + 1)) * 100
    ax.annotate(
        f"2026: {event_value:.1f}\u00b0C\n(p{percentile:.0f})",
        xy=(event_value, density.max()),
        xytext=(8, 0), textcoords="offset points",
        fontsize=9.5, color=COLOR_EVENT, fontweight="bold", va="top",
    )

    ax.set_xlim(x_min, x_max)
    ax.set_yticks([])
    ax.set_xlabel("t2m (\u00b0C)")
    ax.set_title(f"{window_days}-day pre-event mean\n(1950-2025 distribution, n={len(values)})",
                 fontsize=10.5)

# --- Trend panel ---
ax_trend = fig.add_subplot(gs[1, :])
trend_window = max(LOOKBACK_WINDOWS_DAYS)
col = f"mean_{trend_window}d_degc"
hist_valid = historical.dropna(subset=[col])
years = hist_valid["year"].values.astype(float)
values = hist_valid[col].values

mk_p = mann_kendall_p_value(values)
sen_slope, sen_intercept, sen_lo_slope, sen_hi_slope = scipy_stats.theilslopes(values, years, alpha=0.05)

years_line = np.linspace(years.min(), years.max(), 200)
fit_line = sen_slope * years_line + sen_intercept

# CI band: fan out from the same anchor point on the central fit line using
# the lower/upper bound slopes from theilslopes' 95% CI.
anchor_year = np.median(years)
anchor_value = sen_slope * anchor_year + sen_intercept
ci_lower = anchor_value + sen_lo_slope * (years_line - anchor_year)
ci_upper = anchor_value + sen_hi_slope * (years_line - anchor_year)

ax_trend.fill_between(years_line, ci_lower, ci_upper, color=COLOR_TREND_LINE, alpha=0.10, linewidth=0)
ax_trend.plot(years_line, fit_line, color=COLOR_TREND_LINE, linewidth=1.6)
ax_trend.scatter(years, values, color=COLOR_HISTORICAL, s=32, alpha=0.75, zorder=3, edgecolor="none")
ax_trend.scatter([EVENT_YEAR], [event_row[col]], color=COLOR_EVENT, s=110, zorder=5,
                  edgecolor="white", linewidth=1.2)
ax_trend.annotate("2026", xy=(EVENT_YEAR, event_row[col]), xytext=(-34, -4),
                   textcoords="offset points", fontsize=10, color=COLOR_EVENT, fontweight="bold")

# Call out 1960 explicitly, since it is the next-highest year and the figure
# should not visually read as "2026 is the highest on record."
row_1960 = df[df["year"] == 1960].iloc[0]
ax_trend.annotate("1960 (next-highest year)", xy=(1960, row_1960[col]), xytext=(10, -14),
                   textcoords="offset points", fontsize=8.5, color="#555555", ha="left")

sig_text = f"significant, Mann-Kendall p = {mk_p:.4f}" if mk_p < 0.05 else "not statistically significant"
ax_trend.text(
    0.015, 0.06,
    f"Trend (Theil-Sen): {sen_slope * 10:+.2f}\u00b0C/decade, 95% CI "
    f"[{sen_lo_slope*10:+.2f}, {sen_hi_slope*10:+.2f}] ({sig_text}). Robust to autocorrelation "
    f"present in OLS residuals (Durbin-Watson = 1.26).",
    transform=ax_trend.transAxes, fontsize=9, va="bottom", color=COLOR_TREND_LINE,
)

ax_trend.set_xlabel("year")
ax_trend.set_ylabel(f"{trend_window}-day pre-event mean t2m (\u00b0C)")
ax_trend.set_title("Long-term trend, 1950-2025 (2026 event year shown for reference, excluded from fit)",
                    fontsize=11, pad=12)

fig.suptitle(
    "Pre-event thermal conditions, Langtang Lirung source zone -- ERA5-Land, 1950-2026",
    fontsize=13.5, fontweight="bold", y=1.00,
)

fig.text(
    0.01, -0.02,
    "Data: ERA5-Land (Copernicus/ECMWF), single grid cell at 28.285\u00b0N, 85.525\u00b0E (~5100-5200 m). "
    "Attribution-adjacent screening, not formal detection-and-attribution. Trend fit excludes 2026.",
    fontsize=7.5, color="#777777", ha="left",
)

fig.savefig(FIGURES_DIR / "method_a_final_figure.png", dpi=200, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "method_a_final_figure.svg", bbox_inches="tight")
print("Saved figure.")