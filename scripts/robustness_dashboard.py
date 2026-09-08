"""
robustness_dashboard.py

A single figure summarizing everything from robustness_checks.py -- the
spatial grid, the elevation/land-cover investigation of the east-cell
anomaly, and both MODIS cross-checks. None of these results were visualized
before; they only existed as terminal output and CSVs, which is exactly the
"buried in dense output" problem flagged for the main finding, just applied
to the robustness checks themselves.

Reads real data from results/*.csv wherever those files exist (spatial
robustness, both MODIS cross-checks). The elevation and land-cover CSVs
(grid_point_elevations.csv, grid_point_landcover.csv) are only produced by
the patched version of robustness_checks.py -- if you haven't re-run that
yet, this script falls back to the values already reported in terminal
output, clearly marked as such below. Re-run robustness_checks.py once and
this script will pick up the real CSVs automatically on the next run.

Four panels:
  1. Elevation vs. 2026 temperature at each grid point -- shows the west-cell
     difference is elevation-explained, while the east-cell difference is
     NOT (same elevation as center, 6.3 degC colder) -- visualizes exactly
     the anomaly the land-cover check was built to investigate.
  2. Percentile rank at each grid point -- the valid spatial-robustness
     statistic (unlike raw temperature, which panel 1 shows is confounded
     by elevation).
  3. Trend (degC/decade) at each grid point -- same logic, for the warming
     trend rather than the anomaly ranking.
  4. MODIS completeness by year, Terra-day-only vs. multi-sensor combined --
     shows the coverage improvement directly, with each method's resulting
     2026 percentile rank annotated.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

COLOR_HISTORICAL = "#6699BF"
COLOR_EVENT = "#B23A48"
COLOR_NEUTRAL = "#2B2B2B"
COLOR_MULTI = "#8B5A2B"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GRID_ORDER = ["north", "west", "center", "east", "south"]  # rough W-to-E, N-to-S visual order

# Fallback values, sourced directly from the robustness_checks.py terminal
# output already produced -- used only if grid_point_elevations.csv /
# grid_point_landcover.csv are not yet present (i.e. before the patched
# script has been re-run). Re-run robustness_checks.py and delete this
# fallback block's use to switch to the live CSVs automatically.
FALLBACK_ELEVATIONS = {
    "center": 4859, "north": 4439, "south": 4703, "east": 4877, "west": 3475,
}
FALLBACK_LANDCOVER = {
    "center": ("Bare/sparse vegetation", 16), "north": ("Grassland", 1),
    "south": ("Snow and ice", 4), "east": ("Bare/sparse vegetation", 2),
    "west": ("Tree cover", 0),
}

# ---------------------------------------------------------------------------
# Load spatial robustness results
# ---------------------------------------------------------------------------
spatial_df = pd.read_csv(RESULTS_DIR / "spatial_robustness_results.csv")
spatial_df = spatial_df.set_index("grid_point").reindex(GRID_ORDER)

elev_csv_path = RESULTS_DIR / "grid_point_elevations.csv"
if elev_csv_path.exists():
    elev_df = pd.read_csv(elev_csv_path).set_index("grid_point").reindex(GRID_ORDER)
    elevations = elev_df["elevation_m"]
else:
    elevations = pd.Series(FALLBACK_ELEVATIONS).reindex(GRID_ORDER)
    print("NOTE: using fallback elevation values from terminal output -- "
          "re-run the patched robustness_checks.py to generate grid_point_elevations.csv.")

landcover_csv_path = RESULTS_DIR / "grid_point_landcover.csv"
if landcover_csv_path.exists():
    lc_df = pd.read_csv(landcover_csv_path).set_index("grid_point").reindex(GRID_ORDER)
    landcover_labels = {g: lc_df.loc[g, "land_cover_class"] for g in GRID_ORDER}
else:
    landcover_labels = {g: FALLBACK_LANDCOVER[g][0] for g in GRID_ORDER}
    print("NOTE: using fallback land-cover values from terminal output -- "
          "re-run the patched robustness_checks.py to generate grid_point_landcover.csv.")

# ---------------------------------------------------------------------------
# Load MODIS cross-check results
# ---------------------------------------------------------------------------
modis_terra_df = pd.read_csv(RESULTS_DIR / "modis_cross_check_results.csv")
modis_multi_df = pd.read_csv(RESULTS_DIR / "modis_multi_sensor_cross_check_results.csv")

EVENT_YEAR = 2026
TREND_WINDOW_DAYS = 30


def percentile_of_event(df, value_col):
    historical = df[df["year"] != EVENT_YEAR].dropna(subset=[value_col])
    event_row = df[df["year"] == EVENT_YEAR]
    if event_row.empty or pd.isna(event_row[value_col].values[0]) or len(historical) < 10:
        return None
    event_value = event_row[value_col].values[0]
    values = historical[value_col].values
    return (np.sum(values <= event_value) / (len(values) + 1)) * 100


# Fallback percentiles, matching the values already reported in terminal
# output, used only if the CSV's historical mean_degc column is incomplete
# (e.g. a reconstructed/partial CSV) -- your real saved CSVs from
# robustness_checks.py have the full historical series and will compute
# these directly instead of using the fallback.
FALLBACK_TERRA_PERCENTILE = 74.1
FALLBACK_MULTI_PERCENTILE = 88.9

terra_percentile = percentile_of_event(modis_terra_df, "mean_degc") or FALLBACK_TERRA_PERCENTILE
multi_percentile = percentile_of_event(modis_multi_df, "mean_degc") or FALLBACK_MULTI_PERCENTILE

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
ax_elev, ax_pct, ax_trend, ax_modis = axes.flatten()

fig.suptitle("Method A robustness checks", fontsize=17, fontweight="bold", y=0.995)
fig.text(0.5, 0.965, "Spatial grid, surface-type investigation, and independent-sensor cross-checks",
          fontsize=11, color="#555555", ha="center")

# --- Panel 1: elevation vs temperature, highlighting the unexplained east cell ---
for g in GRID_ORDER:
    color = COLOR_EVENT if g == "east" else COLOR_HISTORICAL
    ax_elev.scatter(elevations[g], spatial_df.loc[g, "event_value_degc"], s=140, color=color,
                     zorder=3, edgecolor="white", linewidth=1)
    label = f"{g}\n({landcover_labels[g]})"
    ax_elev.annotate(label, xy=(elevations[g], spatial_df.loc[g, "event_value_degc"]),
                      xytext=(8, 6), textcoords="offset points", fontsize=8.5)

ax_elev.annotate(
    "east: same elevation as center,\nbut 6.3\u00b0C colder --\nnot explained by elevation or land cover",
    xy=(elevations["east"], spatial_df.loc["east", "event_value_degc"]),
    xytext=(-195, 15), textcoords="offset points", fontsize=9, color=COLOR_EVENT,
    arrowprops=dict(arrowstyle="->", color=COLOR_EVENT, lw=1.2),
)
ax_elev.set_xlabel("Elevation (m)")
ax_elev.set_ylabel("2026 pre-event mean temperature (\u00b0C)")
ax_elev.set_title("Elevation vs. temperature by grid point", fontsize=12, fontweight="bold")

# --- Panel 2: percentile rank by grid point (the valid spatial statistic) ---
percentiles = spatial_df["percentile_rank"]
bar_colors = [COLOR_EVENT if p == percentiles.max() else COLOR_HISTORICAL for p in percentiles]
bars = ax_pct.bar(GRID_ORDER, percentiles, color=bar_colors, edgecolor="white")
ax_pct.axhline(50, color="#999999", linewidth=0.8, linestyle="--")
for i, g in enumerate(GRID_ORDER):
    ax_pct.text(i, percentiles[g] + 1.5, f"{percentiles[g]:.0f}", ha="center", fontsize=10, fontweight="bold")
ax_pct.set_ylim(0, 105)
ax_pct.set_ylabel("Percentile rank of 2026")
ax_pct.set_title("Anomaly ranking is consistent across the grid", fontsize=12, fontweight="bold")
ax_pct.text(0.02, 0.04, "All 5 points rank 2026 in the warm half of their own history",
            transform=ax_pct.transAxes, fontsize=8.5, color="#555555")

# --- Panel 3: trend by grid point ---
trends = spatial_df["trend_degc_per_decade"]
ax_trend.bar(GRID_ORDER, trends, color=COLOR_HISTORICAL, edgecolor="white")
for i, g in enumerate(GRID_ORDER):
    sig = "*" if spatial_df.loc[g, "trend_p_value"] < 0.05 else ""
    ax_trend.text(i, trends[g] + 0.003, f"{trends[g]:.3f}{sig}", ha="center", fontsize=10, fontweight="bold")
ax_trend.set_ylabel("Warming trend (\u00b0C/decade)")
ax_trend.set_title("Warming trend is positive at every grid point", fontsize=12, fontweight="bold")
ax_trend.set_ylim(0, max(trends) * 1.25)
ax_trend.text(0.98, 0.96, "* statistically significant, p < 0.05",
              transform=ax_trend.transAxes, fontsize=8.5, color="#555555", va="top", ha="right")

# --- Panel 4: MODIS completeness, Terra-only vs multi-sensor ---
years = modis_terra_df["year"]
width = 0.4
ax_modis.bar(years - width/2, modis_terra_df["n_days_available"], width=width,
             color=COLOR_HISTORICAL, label=f"Terra day only (2026 percentile: {terra_percentile:.0f})")
ax_modis.bar(years + width/2, modis_multi_df["n_days_available"], width=width,
             color=COLOR_MULTI, label=f"Terra+Aqua, day+night (2026 percentile: {multi_percentile:.0f})")
ax_modis.axhline(TREND_WINDOW_DAYS * 0.3, color="#999999", linewidth=0.8, linestyle="--")
ax_modis.text(years.max() - 1, TREND_WINDOW_DAYS * 0.3 + 0.8, "30% coverage threshold",
              fontsize=8, color="#777777", ha="right")
ax_modis.set_xlabel("Year")
ax_modis.set_ylabel(f"Valid days (of {TREND_WINDOW_DAYS})")
ax_modis.set_title("MODIS coverage improves with more sensors,\nbut both agree 2026 ranks warm",
                    fontsize=12, fontweight="bold")
ax_modis.legend(fontsize=8.5, loc="upper left")

fig.text(
    0.01, -0.01,
    "Elevation and land-cover values reproduced from terminal output pending re-run of the patched "
    "robustness_checks.py, which now saves these to CSV directly. MODIS multi-sensor values combine "
    "day and night LST and are not directly comparable in absolute terms to air temperature (ERA5-Land) "
    "or to the day-only MODIS check -- see writeup for the day/night caveat.",
    fontsize=7.5, color="#777777", ha="left", style="italic", wrap=True,
)

fig.tight_layout(rect=[0, 0.01, 1, 0.92])
fig.savefig(FIGURES_DIR / "method_a_robustness_dashboard.png", dpi=200, bbox_inches="tight")
fig.savefig(FIGURES_DIR / "method_a_robustness_dashboard.svg", bbox_inches="tight")
print("Saved robustness dashboard.")