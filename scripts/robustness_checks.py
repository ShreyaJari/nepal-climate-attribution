"""
robustness_checks.py

Two independent robustness checks on the Method A percentile-rank and trend
findings, addressing two specific gaps:

  1. Spatial robustness: the original analysis used a single ERA5-Land grid
     cell (~9-11 km). This repeats the same percentile-rank and trend
     computation at 5 grid points -- the center cell plus its N/S/E/W
     neighbors, spaced at ERA5-Land's native ~0.1 degree resolution -- to
     check whether the finding holds across the surrounding grid or is an
     artifact of one noisy pixel.

  2. Independent-sensor cross-check: ERA5-Land is a reanalysis product (a
     model blended with observations), not a direct measurement. MODIS LST
     is an independent satellite sensor. This computes the same percentile-
     rank statistic using MODIS LST instead, over the years it is actually
     available (2000-2025 -- explicitly NOT the same 76-year baseline as
     ERA5-Land, and stated as such), using only non-cloud-masked valid days.

Both checks report their result honestly even if it does NOT confirm the
original finding -- a robustness check that only ever confirms is not
actually checking anything.

Requirements
------------
    pip install earthengine-api pandas numpy scipy

Run locally and review the printed results.
"""

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Anchor all paths to this script's own location, not the working directory
# the IDE happens to launch from.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Site and analysis configuration (matches method_a_era5_baseline.py)
# ---------------------------------------------------------------------------
SITE_LAT = 28.28531746075177
SITE_LON = 85.52515461726692

EVENT_YEAR = 2026
EVENT_MONTH = 8
EVENT_DAY = 26

ERA5_HISTORICAL_START_YEAR = 1950
ERA5_HISTORICAL_END_YEAR = 2025

# MODIS Terra (MOD11A1) record starts February 2000. Using 2000-2025 as the
# historical baseline for the MODIS cross-check -- a genuinely different,
# shorter period than the ERA5-Land baseline, stated explicitly rather than
# implying equivalence.
MODIS_HISTORICAL_START_YEAR = 2000
MODIS_HISTORICAL_END_YEAR = 2025

TREND_WINDOW_DAYS = 30  # the lookback window used for percentile rank and trend in both checks
FETCH_BUFFER_DAYS = TREND_WINDOW_DAYS + 5
CHUNK_SIZE_YEARS = 10

# ERA5-Land's native grid spacing is 0.1 degrees (~9-11 km at this latitude).
# Using that spacing for neighbor points keeps each point on a genuinely
# distinct grid cell rather than resampling within the same cell.
GRID_SPACING_DEG = 0.1
GRID_POINTS = {
    "center": (SITE_LAT, SITE_LON),
    "north": (SITE_LAT + GRID_SPACING_DEG, SITE_LON),
    "south": (SITE_LAT - GRID_SPACING_DEG, SITE_LON),
    "east": (SITE_LAT, SITE_LON + GRID_SPACING_DEG),
    "west": (SITE_LAT, SITE_LON - GRID_SPACING_DEG),
}

SPATIAL_ROBUSTNESS_CSV = RESULTS_DIR / "spatial_robustness_results.csv"
MODIS_CROSS_CHECK_CSV = RESULTS_DIR / "modis_cross_check_results.csv"


def print_section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def initialize_earth_engine():
    import ee
    print_section("STEP 1: Earth Engine initialization")
    try:
        ee.Initialize(project="carbon-verification-toolkit")
        print("OK: Earth Engine initialized.")
        return True
    except Exception as exc:
        print(f"FAILED: {exc}")
        return False


def compute_percentile_rank(historical_values, test_value):
    """Same plotting-position formula used in Method A: rank = count(<=)/(n+1)*100."""
    historical_values = np.asarray(historical_values)
    historical_values = historical_values[~np.isnan(historical_values)]
    n = len(historical_values)
    if n == 0:
        return np.nan, 0
    count_less_equal = np.sum(historical_values <= test_value)
    return (count_less_equal / (n + 1)) * 100, n


# ---------------------------------------------------------------------------
# Generic fetch: daily band values for a point, pre-event window, all years
# ---------------------------------------------------------------------------
def fetch_pre_event_daily_series(dataset_id, band_name, lat, lon, scale_m,
                                  historical_start_year, historical_end_year,
                                  event_year, buffer_days, chunk_size_years,
                                  kelvin_to_celsius, lst_scale_factor=1.0):
    """
    Fetch daily values of band_name from dataset_id at (lat, lon), for the
    window [event_date - buffer_days, event_date) in every year from
    historical_start_year through event_year, in decade-sized chunks.

    kelvin_to_celsius: if True, subtract 273.15 from the raw value.
    lst_scale_factor: multiply raw value by this before unit conversion
    (MODIS LST bands are stored as scaled integers; ERA5-Land is not).

    Returns a DataFrame with columns: date, year, days_before_event, value_degc.
    """
    import ee

    point = ee.Geometry.Point([lon, lat])
    all_years = list(range(historical_start_year, event_year + 1))
    chunks = [all_years[i:i + chunk_size_years] for i in range(0, len(all_years), chunk_size_years)]

    records = []
    for chunk_years in chunks:
        year_windows = []
        for year in chunk_years:
            event_date = date(year, EVENT_MONTH, EVENT_DAY)
            start_date = event_date - timedelta(days=buffer_days)
            year_windows.append((year, start_date, event_date))

        date_filters = [
            ee.Filter.date(start_date.isoformat(), event_date.isoformat())
            for (_, start_date, event_date) in year_windows
        ]
        combined_filter = ee.Filter.Or(date_filters)

        collection = ee.ImageCollection(dataset_id).select(band_name).filter(combined_filter)

        def extract_value(img):
            raw_value = img.reduceRegion(
                reducer=ee.Reducer.first(), geometry=point, scale=scale_m
            ).get(band_name)
            date_str = img.date().format("YYYY-MM-dd")
            return ee.Feature(None, {"date": date_str, "raw_value": raw_value})

        feature_collection = ee.FeatureCollection(collection.map(extract_value))
        try:
            result = feature_collection.getInfo()
        except Exception as exc:
            print(f"  Chunk {chunk_years[0]}-{chunk_years[-1]}: FAILED -- {exc}")
            continue

        chunk_count = 0
        for feature in result["features"]:
            props = feature["properties"]
            # GEE drops a property key entirely (rather than setting it to
            # null) when reduceRegion returns empty -- e.g. a fully
            # cloud-masked pixel. Use .get() with a default, not direct
            # indexing, or this raises KeyError instead of being treated as
            # missing data. ERA5-Land rarely masks land pixels so this never
            # surfaced there; MODIS LST is frequently cloud-masked here.
            raw_value = props.get("raw_value")
            if raw_value is None:
                continue
            value = raw_value * lst_scale_factor
            if kelvin_to_celsius:
                value = value - 273.15
            records.append({"date": props["date"], "value_degc": value})
            chunk_count += 1
        print(f"  Chunk {chunk_years[0]}-{chunk_years[-1]}: {chunk_count} valid daily values.")

    if not records:
        return pd.DataFrame(columns=["date", "year", "days_before_event", "value_degc"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    def assign_year_and_offset(d):
        year = d.year
        event_date_this_year = pd.Timestamp(year=year, month=EVENT_MONTH, day=EVENT_DAY)
        return year, (event_date_this_year - d).days

    assigned = df["date"].apply(assign_year_and_offset)
    df["year"] = assigned.apply(lambda t: t[0])
    df["days_before_event"] = assigned.apply(lambda t: t[1])
    return df.sort_values(["year", "date"]).reset_index(drop=True)


def compute_window_mean_per_year(daily_df, window_days):
    """Mean value_degc over days_before_event in [1, window_days], per year, plus data count."""
    rows = []
    for year, group in daily_df.groupby("year"):
        window_data = group[(group["days_before_event"] >= 1) & (group["days_before_event"] <= window_days)]
        rows.append({
            "year": year,
            "mean_degc": window_data["value_degc"].mean(),
            "n_days_available": len(window_data),
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def print_grid_point_elevations():
    """
    Sample Copernicus DEM elevation at each of the 5 grid points. This is
    here specifically to check a hypothesis: if event_value_degc varies
    wildly across grid points (much more than reanalysis noise would
    explain), the likely cause is that ERA5-Land grid cells ~11 km apart in
    this terrain sit at genuinely different elevations -- confirmed
    empirically here rather than just asserted. Saved to CSV so downstream
    analysis (e.g. the robustness dashboard) doesn't need to re-run this.
    """
    import ee
    print_section("Grid point elevations (Copernicus DEM, for interpreting spatial spread)")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").mosaic().select("DEM")
    rows = []
    for label, (lat, lon) in GRID_POINTS.items():
        point = ee.Geometry.Point([lon, lat])
        try:
            elev_m = dem.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=30).get("DEM").getInfo()
            print(f"  {label}: {elev_m:.0f} m")
            rows.append({"grid_point": label, "lat": lat, "lon": lon, "elevation_m": elev_m})
        except Exception as exc:
            print(f"  {label}: FAILED -- {exc}")
            rows.append({"grid_point": label, "lat": lat, "lon": lon, "elevation_m": None})

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "grid_point_elevations.csv", index=False)
    print(f"Saved to {RESULTS_DIR / 'grid_point_elevations.csv'}")


def check_land_cover_at_grid_points():
    """
    Sample ESA WorldCover land-cover class and MODIS snow-cover fraction at
    each of the 5 grid points. This is here specifically to check a
    hypothesis raised by the elevation results: 'east' and 'center' sit at
    almost identical elevation (4877 m vs 4859 m, 18 m apart) but 'east' is
    6.3 degC colder -- elevation cannot explain that, so this checks whether
    a surface-type difference (e.g. glacier ice vs bare rock) can.
    """
    import ee
    print_section("Land cover and snow-cover check (for the unexplained east-cell anomaly)")

    worldcover_class_names = {
        10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
        50: "Built-up", 60: "Bare/sparse vegetation", 70: "Snow and ice",
        80: "Permanent water bodies", 90: "Herbaceous wetland",
        95: "Mangroves", 100: "Moss and lichen",
    }
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    event_date = date(EVENT_YEAR, EVENT_MONTH, EVENT_DAY)
    start_date = event_date - timedelta(days=TREND_WINDOW_DAYS)
    modis_snow = (
        ee.ImageCollection("MODIS/061/MOD10A1")
        .select("NDSI_Snow_Cover")
        .filterDate(start_date.isoformat(), event_date.isoformat())
    )

    rows = []
    for label, (lat, lon) in GRID_POINTS.items():
        point = ee.Geometry.Point([lon, lat])

        try:
            cls = worldcover.reduceRegion(
                reducer=ee.Reducer.mode(), geometry=point.buffer(50), scale=10
            ).get("Map").getInfo()
            cls_name = worldcover_class_names.get(cls, f"Unknown class {cls}")
        except Exception as exc:
            cls_name = f"FAILED -- {exc}"

        try:
            def extract_snow(img):
                v = img.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=500).get("NDSI_Snow_Cover")
                return ee.Feature(None, {"v": v})
            snow_vals = ee.FeatureCollection(modis_snow.map(extract_snow)).aggregate_array("v").getInfo()
            # NDSI_Snow_Cover: 0-100 is a valid fraction; values above 100 are
            # QA flags (cloud=250, missing=200, ocean=239, etc.), not data.
            valid_snow = [v for v in snow_vals if v is not None and v <= 100]
            mean_snow_fraction = np.mean(valid_snow) if valid_snow else None
            n_clear_sky = len(valid_snow)
            n_total = len(snow_vals)
            snow_str = (f"{mean_snow_fraction:.0f}% (n={n_clear_sky}/{n_total} clear-sky days)"
                        if mean_snow_fraction is not None else "no clear-sky data in window")
        except Exception as exc:
            mean_snow_fraction, n_clear_sky, n_total = None, None, None
            snow_str = f"FAILED -- {exc}"

        print(f"  {label}: land cover = {cls_name} | 2026 pre-event mean snow-cover fraction = {snow_str}")
        rows.append({
            "grid_point": label, "lat": lat, "lon": lon, "land_cover_class": cls_name,
            "mean_snow_cover_fraction_pct": mean_snow_fraction,
            "n_clear_sky_days": n_clear_sky, "n_days_in_window": n_total,
        })

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "grid_point_landcover.csv", index=False)
    print(f"Saved to {RESULTS_DIR / 'grid_point_landcover.csv'}")

    print("\nIf 'east' comes back as snow/ice cover (or a much higher snow fraction) while 'center' "
          "does not, that is a plausible physical explanation for the elevation-inconsistent cold "
          "anomaly at 'east': a snow/ice surface stays near 0 degC and reflects more incoming energy, "
          "versus bare rock at the same elevation warming well above freezing in summer.")


# ---------------------------------------------------------------------------
# Check 1: spatial robustness across the 5-point grid
# ---------------------------------------------------------------------------
def run_spatial_robustness_check():
    print_section("CHECK 1: Spatial robustness (5-point ERA5-Land grid)")

    results = []
    for label, (lat, lon) in GRID_POINTS.items():
        print(f"\n--- Grid point: {label} ({lat:.4f}, {lon:.4f}) ---")
        daily_df = fetch_pre_event_daily_series(
            dataset_id="ECMWF/ERA5_LAND/DAILY_AGGR", band_name="temperature_2m",
            lat=lat, lon=lon, scale_m=9000,
            historical_start_year=ERA5_HISTORICAL_START_YEAR, historical_end_year=ERA5_HISTORICAL_END_YEAR,
            event_year=EVENT_YEAR, buffer_days=FETCH_BUFFER_DAYS, chunk_size_years=CHUNK_SIZE_YEARS,
            kelvin_to_celsius=True,
        )
        if daily_df.empty:
            print(f"  {label}: FAILED to retrieve data.")
            continue

        window_df = compute_window_mean_per_year(daily_df, TREND_WINDOW_DAYS)
        historical = window_df[window_df["year"] != EVENT_YEAR].dropna(subset=["mean_degc"])
        event_row = window_df[window_df["year"] == EVENT_YEAR]

        if event_row.empty or np.isnan(event_row["mean_degc"].values[0]):
            print(f"  {label}: no valid 2026 value.")
            continue

        event_value = event_row["mean_degc"].values[0]
        percentile, n_hist = compute_percentile_rank(historical["mean_degc"].values, event_value)

        years = historical["year"].values.astype(float)
        values = historical["mean_degc"].values
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(years, values)

        print(f"  {label}: 2026 = {event_value:.2f} degC | percentile = {percentile:.1f} "
              f"| trend = {slope*10:+.3f} degC/decade (p={p_value:.4f}, R2={r_value**2:.3f})")

        results.append({
            "grid_point": label, "lat": lat, "lon": lon,
            "event_value_degc": event_value, "percentile_rank": percentile,
            "n_historical_years": n_hist, "trend_degc_per_decade": slope * 10,
            "trend_p_value": p_value, "trend_r_squared": r_value ** 2,
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(SPATIAL_ROBUSTNESS_CSV, index=False)
    print(f"\nSaved spatial robustness results to {SPATIAL_ROBUSTNESS_CSV}")

    if len(results_df) > 1:
        pct_range = results_df["percentile_rank"].max() - results_df["percentile_rank"].min()
        trend_range = results_df["trend_degc_per_decade"].max() - results_df["trend_degc_per_decade"].min()
        temp_range = results_df["event_value_degc"].max() - results_df["event_value_degc"].min()
        print(f"\nSpread across grid points:")
        print(f"  Raw event_value_degc range: {temp_range:.2f} degC -- DO NOT interpret this as "
              f"reanalysis noise or spatial uncertainty. See grid point elevations above: at ~11 km "
              f"spacing in this terrain, neighboring cells sit at genuinely different elevations, so "
              f"absolute temperatures are not directly comparable across grid points.")
        print(f"  Percentile rank range: {pct_range:.1f} points -- THIS is the valid spatial-robustness "
              f"statistic, since it compares each cell against its own historical distribution, largely "
              f"canceling a constant elevation offset.")
        print(f"  Trend range: {trend_range:.3f} degC/decade -- also valid for the same reason.")
        if pct_range < 25 and (results_df["percentile_rank"] > 50).all():
            print("  All grid points rank 2026 in the warm half of their own history, with a moderate "
                  "spread -- this supports reporting the anomaly as spatially robust, using percentile "
                  "rank (not absolute temperature) as the comparable statistic across cells.")

    return results_df


# ---------------------------------------------------------------------------
# Check 2: MODIS LST independent-sensor cross-check
# ---------------------------------------------------------------------------
def run_modis_cross_check():
    print_section("CHECK 2: MODIS LST independent-sensor cross-check (2000-2025 baseline)")

    daily_df = fetch_pre_event_daily_series(
        dataset_id="MODIS/061/MOD11A1", band_name="LST_Day_1km",
        lat=SITE_LAT, lon=SITE_LON, scale_m=1000,
        historical_start_year=MODIS_HISTORICAL_START_YEAR, historical_end_year=MODIS_HISTORICAL_END_YEAR,
        event_year=EVENT_YEAR, buffer_days=FETCH_BUFFER_DAYS, chunk_size_years=CHUNK_SIZE_YEARS,
        kelvin_to_celsius=True, lst_scale_factor=0.02,  # MODIS LST scale factor per product spec
    )

    if daily_df.empty:
        print("FAILED: no MODIS LST data retrieved.")
        return pd.DataFrame()

    daily_df.to_csv(RESULTS_DIR / "modis_lst_daily_series.csv", index=False)
    window_df = compute_window_mean_per_year(daily_df, TREND_WINDOW_DAYS)
    window_df.to_csv(MODIS_CROSS_CHECK_CSV, index=False)

    completeness = window_df["n_days_available"] / TREND_WINDOW_DAYS
    print(f"\nData completeness per year (of {TREND_WINDOW_DAYS} possible days):")
    print(window_df[["year", "n_days_available"]].to_string(index=False))
    low_completeness_years = window_df.loc[completeness < 0.3, "year"].tolist()
    if low_completeness_years:
        print(f"\nNOTE: years with <30% valid-pixel coverage (cloud masking) -- treat their values as "
              f"unreliable, not as confirming or refuting anything: {low_completeness_years}")

    historical = window_df[(window_df["year"] != EVENT_YEAR)].dropna(subset=["mean_degc"])
    event_row = window_df[window_df["year"] == EVENT_YEAR]

    if event_row.empty or np.isnan(event_row["mean_degc"].values[0]):
        print("\nRESULT: 2026 MODIS LST value is missing or fully cloud-masked in this window -- "
              "the independent-sensor cross-check is INCONCLUSIVE, not confirming.")
        return window_df

    event_value = event_row["mean_degc"].values[0]
    event_completeness = event_row["n_days_available"].values[0] / TREND_WINDOW_DAYS
    percentile, n_hist = compute_percentile_rank(historical["mean_degc"].values, event_value)

    print(f"\nRESULT: 2026 MODIS LST {TREND_WINDOW_DAYS}-day pre-event mean = {event_value:.2f} degC "
          f"(from {event_row['n_days_available'].values[0]}/{TREND_WINDOW_DAYS} valid days, "
          f"{event_completeness*100:.0f}% coverage) | percentile rank = {percentile:.1f} "
          f"(n={n_hist} historical years, 2000-2025)")
    if event_completeness < 0.3:
        print("CAUTION: 2026's own coverage is below 30% -- report this percentile as indicative only, "
              "not as independent confirmation.")

    return window_df


def fetch_multi_sensor_lst_daily_series(lat, lon, historical_start_year, historical_end_year,
                                          event_year, buffer_days, chunk_size_years):
    """
    Fetch daily LST from 4 sources -- MODIS Terra day, Terra night, Aqua day,
    Aqua night -- and merge into one per-day series by averaging whichever
    of the 4 are valid (non-cloud-masked) that day. This roughly doubles the
    number of independent looks per day versus Terra-day alone, since each
    satellite crosses twice daily and clouds don't always mask every pass.

    Aqua's MYD11A1 record starts July 2002 (vs Terra's February 2000), so
    Aqua contributes nothing for 2000-mid 2002 -- those years fall back to
    Terra-only, same as the original single-sensor check.

    Returns a DataFrame with columns: date, year, days_before_event,
    value_degc (the per-day mean across available sources), n_sources_available.
    """
    import ee

    point = ee.Geometry.Point([lon, lat])
    sources = [
        ("MODIS/061/MOD11A1", "LST_Day_1km", "terra_day"),
        ("MODIS/061/MOD11A1", "LST_Night_1km", "terra_night"),
        ("MODIS/061/MYD11A1", "LST_Day_1km", "aqua_day"),
        ("MODIS/061/MYD11A1", "LST_Night_1km", "aqua_night"),
    ]

    all_years = list(range(historical_start_year, event_year + 1))
    chunks = [all_years[i:i + chunk_size_years] for i in range(0, len(all_years), chunk_size_years)]

    per_source_records = {name: [] for _, _, name in sources}
    for dataset_id, band_name, source_name in sources:
        for chunk_years in chunks:
            year_windows = []
            for year in chunk_years:
                event_date = date(year, EVENT_MONTH, EVENT_DAY)
                start_date = event_date - timedelta(days=buffer_days)
                year_windows.append((year, start_date, event_date))

            date_filters = [
                ee.Filter.date(start_date.isoformat(), event_date.isoformat())
                for (_, start_date, event_date) in year_windows
            ]
            combined_filter = ee.Filter.Or(date_filters)
            collection = ee.ImageCollection(dataset_id).select(band_name).filter(combined_filter)

            def extract_value(img):
                raw_value = img.reduceRegion(reducer=ee.Reducer.first(), geometry=point, scale=1000).get(band_name)
                date_str = img.date().format("YYYY-MM-dd")
                return ee.Feature(None, {"date": date_str, "raw_value": raw_value})

            try:
                result = ee.FeatureCollection(collection.map(extract_value)).getInfo()
            except Exception as exc:
                print(f"  {source_name}, chunk {chunk_years[0]}-{chunk_years[-1]}: FAILED -- {exc}")
                continue

            for feature in result["features"]:
                props = feature["properties"]
                raw_value = props.get("raw_value")
                if raw_value is None:
                    continue
                per_source_records[source_name].append({
                    "date": props["date"], "value_degc": raw_value * 0.02 - 273.15,
                })

        print(f"  {source_name}: {len(per_source_records[source_name])} valid daily values across all years.")

    # Merge the 4 sources on date, averaging whichever are present per day.
    frames = []
    for source_name, records in per_source_records.items():
        if records:
            frames.append(pd.DataFrame(records).rename(columns={"value_degc": source_name}).set_index("date"))
    if not frames:
        return pd.DataFrame(columns=["date", "year", "days_before_event", "value_degc", "n_sources_available"])

    merged = pd.concat(frames, axis=1, join="outer")
    source_cols = [c for c in merged.columns]
    merged["value_degc"] = merged[source_cols].mean(axis=1, skipna=True)
    merged["n_sources_available"] = merged[source_cols].notna().sum(axis=1)
    merged = merged.reset_index().rename(columns={"index": "date"})
    merged["date"] = pd.to_datetime(merged["date"])

    def assign_year_and_offset(d):
        year = d.year
        event_date_this_year = pd.Timestamp(year=year, month=EVENT_MONTH, day=EVENT_DAY)
        return year, (event_date_this_year - d).days

    assigned = merged["date"].apply(assign_year_and_offset)
    merged["year"] = assigned.apply(lambda t: t[0])
    merged["days_before_event"] = assigned.apply(lambda t: t[1])

    return merged[["date", "year", "days_before_event", "value_degc", "n_sources_available"]].sort_values(
        ["year", "date"]
    ).reset_index(drop=True)


def run_modis_multi_sensor_cross_check():
    """
    Repeats the MODIS cross-check using the Terra+Aqua, day+night combined
    series instead of Terra-day alone, to see whether better coverage
    changes the strength of the cross-check (rather than just its number).
    """
    print_section("CHECK 2b: MODIS multi-sensor cross-check (Terra+Aqua, day+night)")

    daily_df = fetch_multi_sensor_lst_daily_series(
        lat=SITE_LAT, lon=SITE_LON,
        historical_start_year=MODIS_HISTORICAL_START_YEAR, historical_end_year=MODIS_HISTORICAL_END_YEAR,
        event_year=EVENT_YEAR, buffer_days=FETCH_BUFFER_DAYS, chunk_size_years=CHUNK_SIZE_YEARS,
    )

    if daily_df.empty:
        print("FAILED: no multi-sensor MODIS data retrieved.")
        return pd.DataFrame()

    daily_df.to_csv(RESULTS_DIR / "modis_multi_sensor_daily_series.csv", index=False)
    window_df = compute_window_mean_per_year(daily_df, TREND_WINDOW_DAYS)
    window_df.to_csv(RESULTS_DIR / "modis_multi_sensor_cross_check_results.csv", index=False)

    completeness = window_df["n_days_available"] / TREND_WINDOW_DAYS
    print(f"\nData completeness per year (of {TREND_WINDOW_DAYS} possible days), multi-sensor combined:")
    print(window_df[["year", "n_days_available"]].to_string(index=False))
    low_completeness_years = window_df.loc[completeness < 0.3, "year"].tolist()
    if low_completeness_years:
        print(f"\nNOTE: years still below 30% coverage even with 4 sources combined: {low_completeness_years}")

    mean_completeness = completeness.mean() * 100
    print(f"\nMean completeness across all years: {mean_completeness:.0f}% "
          f"(compare against the Terra-day-only check's per-year numbers above -- this tells us "
          f"whether adding sources meaningfully helped, or whether cloud cover here is dense enough "
          f"that it doesn't matter how many satellite passes you combine).")

    historical = window_df[window_df["year"] != EVENT_YEAR].dropna(subset=["mean_degc"])
    event_row = window_df[window_df["year"] == EVENT_YEAR]

    if event_row.empty or np.isnan(event_row["mean_degc"].values[0]):
        print("\nRESULT: 2026 value still missing even with 4 sources combined -- INCONCLUSIVE.")
        return window_df

    event_value = event_row["mean_degc"].values[0]
    event_completeness = event_row["n_days_available"].values[0] / TREND_WINDOW_DAYS
    percentile, n_hist = compute_percentile_rank(historical["mean_degc"].values, event_value)

    print(f"\nRESULT: 2026 multi-sensor {TREND_WINDOW_DAYS}-day pre-event mean = {event_value:.2f} degC "
          f"(from {event_row['n_days_available'].values[0]}/{TREND_WINDOW_DAYS} valid days, "
          f"{event_completeness*100:.0f}% coverage) | percentile rank = {percentile:.1f} "
          f"(n={n_hist} historical years, 2000-2025)")
    if event_completeness < 0.3:
        print("CAUTION: still below 30% coverage for 2026 even combining 4 sources -- this cross-check "
              "may simply not be viable at this site regardless of sensor combination.")

    return window_df


if __name__ == "__main__":
    if not initialize_earth_engine():
        sys.exit(1)

    print_grid_point_elevations()
    check_land_cover_at_grid_points()
    spatial_results = run_spatial_robustness_check()
    modis_results = run_modis_cross_check()
    modis_multi_results = run_modis_multi_sensor_cross_check()

    print_section("DONE")
    print("Robustness checks complete. Compare the spatial-robustness spread, and the Terra-day-only "
          "vs. multi-sensor MODIS percentile ranks and completeness, before writing up these findings.")