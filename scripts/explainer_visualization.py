"""
explainer_visualization.py

A plain-language figure for Method A results, for a reader with no
statistics background. Does NOT replace final_visualization.py (the
technical figure) -- this is the headline visual, that is the rigorous
appendix version.

Design: ONE chart, not two, so the "was this unusual" claim and the "is it
part of a trend" claim reinforce each other instead of requiring the reader
to mentally combine two separate panels.

  - A shaded band marks the "warmer than 91% of years" zone directly on the
    chart -- the ranking claim is something the reader SEES, not something
    only the caption tells them.
  - Points are colored by temperature (darker = warmer), so the ranking
    story is visible in the scatter itself, not just via the 2026 highlight.
  - The trend line and the ranking band sit in the same picture, so "hot"
    and "part of a warming trend" are visually one story, not two.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

EVENT_YEAR = 2026
TREND_WINDOW_DAYS = 30
COLOR_EVENT = "#B23A48"
COLOR_TREND_LINE = "#2B2B2B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
})

df = pd.read_csv(RESULTS_DIR / "era5_pre_event_summary.csv")
col = f"mean_{TREND_WINDOW_DAYS}d_degc"
historical = df[df["year"] != EVENT_YEAR].dropna(subset=[col])
event_value = df.loc[df["year"] == EVENT_YEAR, col].values[0]

years = historical["year"].values.astype(float)
values = historical[col].values
percentile = (np.sum(values <= event_value) / (len(values) + 1)) * 100

sen_slope, sen_intercept, _, _ = scipy_stats.theilslopes(values, years, alpha=0.05)

fig, ax = plt.subplots(1, 1, figsize=(12, 7.5))
fig.subplots_adjust(top=0.78, bottom=0.11)

# Shade the "warmer than 91% of years" zone directly on the chart.
warm_threshold = np.percentile(values, percentile)
y_pad = (values.max() - values.min()) * 0.18
ax.axhspan(warm_threshold, values.max() + y_pad, color=COLOR_EVENT, alpha=0.08, zorder=0)
n_years_above = int(np.sum(values > warm_threshold))
ax.text(
    years.min() + 14, warm_threshold + (values.max() + y_pad - warm_threshold) * 0.25,
    f"Only {n_years_above} of {len(values)} years since 1950\nwere ever this warm",
    fontsize=10, color=COLOR_EVENT, va="center", ha="left", style="italic",
)

years_line = np.array([years.min(), years.max()])
fit_line = sen_slope * years_line + sen_intercept
ax.plot(years_line, fit_line, color=COLOR_TREND_LINE, linewidth=2.2, zorder=2)

ax.scatter(years, values, c=values, cmap="Blues", s=70, alpha=0.9, zorder=3,
           edgecolor="white", linewidth=0.6, vmin=values.min(), vmax=values.max())

ax.scatter([EVENT_YEAR], [event_value], color=COLOR_EVENT, s=280, zorder=6,
           edgecolor="white", linewidth=2)
ax.annotate(
    f"2026: {event_value:.1f}\u00b0C\n(warmer than {percentile:.0f}% of years)",
    xy=(EVENT_YEAR, event_value), xytext=(-195, -8), textcoords="offset points",
    fontsize=12.5, fontweight="bold", color=COLOR_EVENT, va="center",
    arrowprops=dict(arrowstyle="-", color=COLOR_EVENT, lw=1.3),
)

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Average temperature in the\n30 days before Aug 26 (\u00b0C)", fontsize=12)
ax.set_ylim(values.min() - 0.3, values.max() + y_pad)

fig.text(0.5, 0.97, "Was it unusually warm before the Nepal ice-rock collapse?",
          fontsize=19, fontweight="bold", ha="center", color="#222222")
fig.text(0.5, 0.935, "Langtang Lirung source zone, ERA5-Land reanalysis data, 1950-2026",
          fontsize=11.5, color="#555555", ha="center")
fig.text(
    0.5, 0.885,
    f"2026 was warmer than {percentile:.0f}% of years since 1950 -- and part of a "
    f"long-term warming trend at this exact location.",
    fontsize=13, ha="center", color="#222222", fontweight="bold",
)
fig.text(
    0.5, 0.845,
    "Each dot is one year's average temperature in the 30 days before the collapse date. "
    "Darker dots were warmer.",
    fontsize=10, color="#555555", ha="center",
)

fig.text(
    0.01, -0.02,
    "This shows how unusual conditions were, not proof that climate change caused the collapse -- "
    "see the technical figure and writeup for the full statistical analysis, uncertainty, and limitations.",
    fontsize=8.5, color="#777777", ha="left", style="italic",
)

fig.savefig(FIGURES_DIR / "method_a_explainer_figure.png", dpi=200, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "method_a_explainer_figure.svg", bbox_inches="tight")
print("Saved explainer figure.")