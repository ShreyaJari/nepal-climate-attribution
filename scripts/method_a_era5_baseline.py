"""
method_a_era5_baseline.py

Purpose
-------
Method A (trend-based screening) for the "Climate Attribution of the Nepal
Ice-Rock Collapse Preconditions" project.

This script does NOT attempt formal climate attribution. It answers a
narrower, honestly-scoped question: were the thermal conditions in the days
preceding the August 26, 2026 collapse anomalous relative to the historical
record for this exact location and season, and is the site's multi-decade
trend consistent with regional warming? This is an attribution-adjacent
screening statement, not a probability-ratio attribution claim.

What it does
------------
1. Fetches daily ERA5-Land 2m air temperature for the source-zone grid cell,
   restricted to a window of days preceding August 26 each year from 1950
   through 2026 (fetched in decade-sized chunks to avoid a single oversized
   request).
2. For three lookback windows (7, 14, 30 days), computes the mean pre-event
   temperature for every historical year (1950-2025) and for 2026, then
   reports 2026's percentile rank against the historical distribution.
3. Fits a simple linear trend (temperature vs. year) using the 30-day
   pre-event mean as the annual metric, to characterize the long-term
   regional warming signal independent of the 2026 event itself.
4. Saves all intermediate and summary results to local CSV files, and
   produces a first diagnostic plot (not the final presentation version --
   see accompanying discussion).

Requirements
------------
    pip install earthengine-api pandas numpy matplotlib scipy

Run this locally, then check the terminal output and the saved PNG before
moving on to the presentation-quality visualization.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats as scipy_stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Anchor all paths to this script's own location, not the working directory
# the IDE happens to launch from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Site and analysis configuration
# ---------------------------------------------------------------------------
SITE_LAT = 28.28531746075177
SITE_LON = 85.52515461726692
SITE_LABEL = "Langtang Lirung source zone"

EVENT_YEAR = 2026
EVENT_MONTH = 8
EVENT_DAY = 26

HISTORICAL_START_YEAR = 1950  # ERA5-Land daily aggregate begins 1950-01-02
HISTORICAL_END_YEAR = 2025    # 2026 is held out as the test case, not part of the reference distribution

LOOKBACK_WINDOWS_DAYS = [7, 14, 30]
FETCH_BUFFER_DAYS = max(LOOKBACK_WINDOWS_DAYS) + 5  # small safety margin
CHUNK_SIZE_YEARS = 10  # fetch this many years per GEE request, to avoid oversized single requests

RAW_SERIES_CSV = RESULTS_DIR / "era5_pre_event_daily_series.csv"
SUMMARY_CSV = RESULTS_DIR / "era5_pre_event_summary.csv"
DIAGNOSTIC_PLOT_PNG = FIGURES_DIR / "era5_pre_event_diagnostic.png"


def print_section(title):
    """Print a formatted section header so terminal output is easy to scan."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# Earth Engine initialization
# ---------------------------------------------------------------------------
def initialize_earth_engine():
    """Initialize Earth Engine using the existing GEE project handle."""
    import ee
    print_section("STEP 1: Earth Engine initialization")
    try:
        ee.Initialize(project="carbon-verification-toolkit")
        print("OK: Earth Engine initialized with project 'carbon-verification-toolkit'.")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


# ---------------------------------------------------------------------------
# Data fetch: daily ERA5-Land t2m for the pre-event window, every year
# ---------------------------------------------------------------------------
def fetch_pre_event_daily_series(historical_start_year, historical_end_year,
                                  event_year, event_month, event_day,
                                  buffer_days, chunk_size_years):
    """
    Fetch daily ERA5-Land 2m temperature at the site for the window
    [event_date - buffer_days, event_date) in every year from
    historical_start_year through event_year (inclusive), fetched in
    decade-sized chunks.

    Returns a pandas DataFrame with columns:
      date               -- the calendar date (datetime64)
      year               -- the year that date's pre-event window belongs to
      t2m_degc           -- 2m air temperature in degrees C
      days_before_event  -- how many days before that year's event date
                             (1 = the day immediately before Aug 26, etc.)
    """
    import ee

    print_section("STEP 2: Fetching ERA5-Land pre-event daily series (1950-2026)")

    point = ee.Geometry.Point([SITE_LON, SITE_LAT])
    all_years = list(range(historical_start_year, event_year + 1))
    chunks = [all_years[i:i + chunk_size_years] for i in range(0, len(all_years), chunk_size_years)]

    records = []
    for chunk_index, chunk_years in enumerate(chunks, start=1):
        year_windows = []
        for year in chunk_years:
            event_date = date(year, event_month, event_day)
            start_date = event_date - timedelta(days=buffer_days)
            year_windows.append((year, start_date, event_date))

        date_filters = [
            ee.Filter.date(start_date.isoformat(), event_date.isoformat())
            for (_, start_date, event_date) in year_windows
        ]
        combined_filter = ee.Filter.Or(date_filters)

        collection = (
            ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
            .select("temperature_2m")
            .filter(combined_filter)
        )

        def extract_value(img):
            t2m_kelvin = img.reduceRegion(
                reducer=ee.Reducer.first(), geometry=point, scale=9000
            ).get("temperature_2m")
            date_str = img.date().format("YYYY-MM-dd")
            return ee.Feature(None, {"date": date_str, "t2m_kelvin": t2m_kelvin})

        feature_collection = ee.FeatureCollection(collection.map(extract_value))

        try:
            result = feature_collection.getInfo()
        except Exception as exc:
            print(f"  Chunk {chunk_index}/{len(chunks)} (years {chunk_years[0]}-{chunk_years[-1]}): "
                  f"FAILED -- {exc}")
            print("  If this is a timeout, try reducing CHUNK_SIZE_YEARS and retrying this chunk.")
            continue

        chunk_record_count = 0
        for feature in result["features"]:
            props = feature["properties"]
            if props["t2m_kelvin"] is None:
                continue
            records.append({
                "date": props["date"],
                "t2m_degc": props["t2m_kelvin"] - 273.15,
            })
            chunk_record_count += 1

        print(f"  Chunk {chunk_index}/{len(chunks)} (years {chunk_years[0]}-{chunk_years[-1]}): "
              f"{chunk_record_count} daily values retrieved.")

    if not records:
        print("FAILED: no records retrieved across any chunk.")
        return pd.DataFrame(columns=["date", "year", "t2m_degc", "days_before_event"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    # Assign each date to the event-year window it belongs to: a date belongs
    # to year Y's window if it falls in [event_date(Y) - buffer, event_date(Y)).
    def assign_year_and_offset(d):
        # The date could only plausibly belong to the window for the year
        # matching its own calendar year or (rarely, if buffer crosses
        # Dec 31) the following year -- not relevant here since the window
        # is entirely within July-August, so this is a direct lookup.
        year = d.year
        event_date_this_year = pd.Timestamp(year=year, month=event_month, day=event_day)
        offset = (event_date_this_year - d).days
        return year, offset

    assigned = df["date"].apply(assign_year_and_offset)
    df["year"] = assigned.apply(lambda t: t[0])
    df["days_before_event"] = assigned.apply(lambda t: t[1])

    df = df.sort_values(["year", "date"]).reset_index(drop=True)
    df.to_csv(RAW_SERIES_CSV, index=False)
    print(f"\nSaved raw daily series ({len(df)} rows) to {RAW_SERIES_CSV}")
    return df


# ---------------------------------------------------------------------------
# Compute lookback-window means per year
# ---------------------------------------------------------------------------
def compute_lookback_means(daily_df, lookback_windows_days):
    """
    For every year present in daily_df, compute the mean t2m_degc over each
    lookback window (the N days immediately preceding that year's event
    date, i.e. days_before_event in [1, N]).

    Returns a DataFrame indexed by year, with one column per lookback window
    (e.g. 'mean_7d_degc', 'mean_14d_degc', 'mean_30d_degc') and a column
    'n_days_available_<N>' recording how many of the N days actually had
    valid data, so incomplete years can be flagged rather than silently
    treated as equal to complete years.
    """
    print_section("STEP 3: Computing lookback-window means per year")

    results = []
    for year, year_group in daily_df.groupby("year"):
        row = {"year": year}
        for window_days in lookback_windows_days:
            window_data = year_group[
                (year_group["days_before_event"] >= 1) &
                (year_group["days_before_event"] <= window_days)
            ]
            row[f"mean_{window_days}d_degc"] = window_data["t2m_degc"].mean()
            row[f"n_days_available_{window_days}d"] = len(window_data)
        results.append(row)

    summary_df = pd.DataFrame(results).sort_values("year").reset_index(drop=True)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-year lookback-window means ({len(summary_df)} years) to {SUMMARY_CSV}")

    for window_days in lookback_windows_days:
        completeness = summary_df[f"n_days_available_{window_days}d"] / window_days
        incomplete_years = summary_df.loc[completeness < 0.8, "year"].tolist()
        if incomplete_years:
            print(f"NOTE: years with <80% data completeness for the {window_days}-day window "
                  f"(consider excluding from the distribution): {incomplete_years}")

    return summary_df


# ---------------------------------------------------------------------------
# Percentile rank of 2026 against the historical distribution
# ---------------------------------------------------------------------------
def compute_percentile_rank(historical_values, test_value):
    """
    Compute the percentile rank of test_value against a historical
    distribution using the standard plotting-position formula:
    rank = (count of historical values <= test_value) / (n + 1) * 100.
    Using n+1 in the denominator (rather than n) avoids ever reporting an
    exact 0th or 100th percentile from a finite sample, which would overstate
    confidence in the tail.
    """
    historical_values = np.asarray(historical_values)
    historical_values = historical_values[~np.isnan(historical_values)]
    n = len(historical_values)
    if n == 0:
        return np.nan, 0
    count_less_equal = np.sum(historical_values <= test_value)
    percentile = (count_less_equal / (n + 1)) * 100
    return percentile, n


def evaluate_2026_against_history(summary_df, lookback_windows_days, event_year):
    """
    For each lookback window, split the summary into historical years vs.
    the event year, then compute 2026's percentile rank against the
    historical distribution. Prints a clear results table.
    """
    print_section("STEP 4: Percentile rank of 2026 pre-event conditions")

    historical = summary_df[summary_df["year"] != event_year]
    event_row = summary_df[summary_df["year"] == event_year]

    if event_row.empty:
        print(f"FAILED: no data found for event year {event_year} in the summary.")
        return {}

    results = {}
    for window_days in lookback_windows_days:
        col = f"mean_{window_days}d_degc"
        historical_values = historical[col].values
        event_value = event_row[col].values[0]

        if np.isnan(event_value):
            print(f"  {window_days}-day window: 2026 value is missing (insufficient data). Skipping.")
            continue

        percentile, n_historical = compute_percentile_rank(historical_values, event_value)
        historical_mean = np.nanmean(historical_values)
        historical_std = np.nanstd(historical_values)

        print(f"  {window_days}-day pre-event mean: 2026 = {event_value:.2f} deg C "
              f"| historical mean = {historical_mean:.2f} deg C, std = {historical_std:.2f} deg C "
              f"| percentile rank = {percentile:.1f} (n={n_historical} historical years)")

        results[window_days] = {
            "event_value_degc": event_value,
            "historical_mean_degc": historical_mean,
            "historical_std_degc": historical_std,
            "percentile_rank": percentile,
            "n_historical_years": n_historical,
        }

    return results


# ---------------------------------------------------------------------------
# Long-term trend
# ---------------------------------------------------------------------------
def compute_long_term_trend(summary_df, trend_window_days, event_year):
    """
    Fit the long-term trend of the trend_window_days pre-event mean
    temperature against year, using historical years only (excluding the
    event year itself).

    Uses Mann-Kendall (significance) + Theil-Sen (slope estimate) as the
    PRIMARY method: distribution-free, robust to outliers, and the standard
    approach in the comparable published Himalayan/High-Mountain-Asia
    literature. OLS is also reported as a secondary comparison, alongside a
    Durbin-Watson check for residual autocorrelation -- OLS assumes
    independent errors, which annual climate data often violates, and an
    autocorrelation check makes that assumption verifiable rather than
    silently assumed.
    """
    print_section(f"STEP 5: Long-term trend ({trend_window_days}-day pre-event mean vs. year)")

    historical = summary_df[summary_df["year"] != event_year].dropna(
        subset=[f"mean_{trend_window_days}d_degc"]
    )
    years = historical["year"].values.astype(float)
    values = historical[f"mean_{trend_window_days}d_degc"].values

    if len(years) < 10:
        print(f"FAILED: only {len(years)} valid historical years available -- too few for a trend fit.")
        return {}

    results = {"n_years": len(years)}

    # --- Primary: Mann-Kendall significance + Theil-Sen slope ---
    def mann_kendall(x):
        n = len(x)
        s = 0
        for k in range(n - 1):
            for j in range(k + 1, n):
                s += np.sign(x[j] - x[k])
        var_s = n * (n - 1) * (2 * n + 5) / 18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z))) if SCIPY_AVAILABLE else np.nan
        return s, z, p

    if SCIPY_AVAILABLE:
        mk_s, mk_z, mk_p = mann_kendall(values)
        sen_slope, sen_intercept, sen_lo, sen_hi = scipy_stats.theilslopes(values, years, alpha=0.05)
        print(f"  Mann-Kendall / Theil-Sen (primary): slope = {sen_slope * 10:.3f} deg C per decade "
              f"(95% CI: {sen_lo * 10:.3f} to {sen_hi * 10:.3f}) | p = {mk_p:.4f} | n = {len(years)} years")
        if mk_p < 0.05:
            print("  Trend is statistically significant at the 0.05 level (Mann-Kendall).")
        else:
            print("  Trend is NOT statistically significant at the 0.05 level (Mann-Kendall) -- "
                  "report this honestly, do not describe the trend as significant if it is not.")
        results.update({
            "sen_slope_degc_per_decade": sen_slope * 10,
            "sen_ci_lo_degc_per_decade": sen_lo * 10,
            "sen_ci_hi_degc_per_decade": sen_hi * 10,
            "mann_kendall_p_value": mk_p,
        })
    else:
        print("  scipy not available -- cannot compute Mann-Kendall/Theil-Sen. Install scipy.")

    # --- Secondary: OLS, plus a Durbin-Watson check for autocorrelation ---
    if SCIPY_AVAILABLE:
        ols_slope, ols_intercept, r_value, ols_p, std_err = scipy_stats.linregress(years, values)
        r_squared = r_value ** 2
        residuals = values - (ols_slope * years + ols_intercept)
        durbin_watson = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        lag1_autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]

        print(f"  OLS (secondary, for comparison): slope = {ols_slope * 10:.3f} deg C per decade "
              f"| R2 = {r_squared:.3f} | p = {ols_p:.4f}")
        print(f"  Durbin-Watson statistic on OLS residuals: {durbin_watson:.3f} "
              f"(2.0 = no autocorrelation; lag-1 residual autocorrelation = {lag1_autocorr:.3f})")
        if durbin_watson < 1.5:
            print("  NOTE: Durbin-Watson well below 2 indicates positive autocorrelation in OLS "
                  "residuals -- OLS's p-value likely overstates confidence. Report the Mann-Kendall "
                  "p-value as primary, not the OLS p-value.")

        results.update({
            "ols_slope_degc_per_decade": ols_slope * 10,
            "ols_r_squared": r_squared,
            "ols_p_value": ols_p,
            "durbin_watson": durbin_watson,
            "lag1_residual_autocorrelation": lag1_autocorr,
        })

    return results


# ---------------------------------------------------------------------------
# Diagnostic plot (first pass -- not the final presentation version)
# ---------------------------------------------------------------------------
def make_diagnostic_plot(summary_df, percentile_results, trend_result,
                          lookback_windows_days, event_year):
    """
    Produce a first-pass diagnostic figure: one panel per lookback window
    showing the historical distribution (histogram) with the 2026 value
    marked, plus a trend panel. This is for sanity-checking the numbers,
    not the polished version for the portfolio -- that will be designed
    once these results are confirmed.
    """
    print_section("STEP 6: Diagnostic plot")

    n_panels = len(lookback_windows_days) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    historical = summary_df[summary_df["year"] != event_year]

    for i, window_days in enumerate(lookback_windows_days):
        ax = axes[i]
        col = f"mean_{window_days}d_degc"
        ax.hist(historical[col].dropna(), bins=20, color="steelblue", alpha=0.7)
        if window_days in percentile_results:
            event_value = percentile_results[window_days]["event_value_degc"]
            percentile = percentile_results[window_days]["percentile_rank"]
            ax.axvline(event_value, color="firebrick", linewidth=2)
            ax.set_title(f"{window_days}-day pre-event mean\n2026: {event_value:.1f} degC "
                         f"(p{percentile:.0f})")
        ax.set_xlabel("t2m_degc")
        ax.set_ylabel("count of historical years")

    ax_trend = axes[-1]
    trend_window = max(lookback_windows_days)
    col = f"mean_{trend_window}d_degc"
    ax_trend.scatter(historical["year"], historical[col], color="steelblue", alpha=0.7)
    event_row = summary_df[summary_df["year"] == event_year]
    if not event_row.empty and not np.isnan(event_row[col].values[0]):
        ax_trend.scatter(event_row["year"], event_row[col], color="firebrick", s=80, zorder=5)
    if "sen_slope_degc_per_decade" in trend_result:
        valid_years = historical.dropna(subset=[col])["year"].values.astype(float)
        valid_values = historical.dropna(subset=[col])[col].values
        sen_slope, sen_intercept, _, _ = scipy_stats.theilslopes(valid_values, valid_years, alpha=0.05)
        trend_line_years = np.array([valid_years.min(), valid_years.max()])
        ax_trend.plot(trend_line_years, sen_slope * trend_line_years + sen_intercept,
                      color="black", linewidth=1.5)
    ax_trend.set_xlabel("year")
    ax_trend.set_ylabel(f"{trend_window}-day pre-event mean t2m_degc")
    ax_trend.set_title("Long-term trend")

    fig.tight_layout()
    fig.savefig(DIAGNOSTIC_PLOT_PNG, dpi=150)
    print(f"Saved diagnostic plot to {DIAGNOSTIC_PLOT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if not initialize_earth_engine():
        sys.exit(1)

    daily_df = fetch_pre_event_daily_series(
        historical_start_year=HISTORICAL_START_YEAR,
        historical_end_year=HISTORICAL_END_YEAR,
        event_year=EVENT_YEAR,
        event_month=EVENT_MONTH,
        event_day=EVENT_DAY,
        buffer_days=FETCH_BUFFER_DAYS,
        chunk_size_years=CHUNK_SIZE_YEARS,
    )

    if daily_df.empty:
        print("\nNo data retrieved -- stopping before further analysis.")
        sys.exit(1)

    summary_df = compute_lookback_means(daily_df, LOOKBACK_WINDOWS_DAYS)
    percentile_results = evaluate_2026_against_history(summary_df, LOOKBACK_WINDOWS_DAYS, EVENT_YEAR)
    trend_result = compute_long_term_trend(summary_df, max(LOOKBACK_WINDOWS_DAYS), EVENT_YEAR)
    make_diagnostic_plot(summary_df, percentile_results, trend_result, LOOKBACK_WINDOWS_DAYS, EVENT_YEAR)

    print_section("DONE")
    print("Baseline analysis complete. Review the summary CSV and diagnostic PNG before "
          "building the final visualization.")