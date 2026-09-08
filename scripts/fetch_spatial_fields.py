"""
fetch_spatial_fields.py

Fetches daily ERA5-Land 2m temperature as 2D spatial fields (not single
point values) over a region surrounding the Langtang Lirung source zone,
for Method B (VAE-based anomaly detection).

This is structurally different from method_a_era5_baseline.py, which
extracted a single reduced value per day at one point. A convolutional VAE
needs actual spatial structure to reconstruct, so this extracts a full 2D
array per day using ee.Image.sampleRectangle() -- NOT reduceRegion() with a
list reducer, which does not guarantee pixel order matches the 2D grid
layout and would silently scramble every training image.

Design choices:
  - 2 degree x 2 degree box centered on the site (~220 km x 220 km), giving
    roughly a 21x21 pixel field per day at ERA5-Land's ~0.1 degree native
    resolution -- small enough to be a tractable VAE input, large enough to
    have real spatial structure.
  - 60-day training window before Aug 26 each year (not Method A's 7/14/30
    day windows -- those were sized for statistical tests on a single
    scalar; a VAE needs more training examples per year to learn a
    plausible manifold of "normal" spatial patterns).
  - Fetched and saved ONE YEAR AT A TIME, not in multi-year chunks like
    Method A's point extraction. Each daily field is a full 2D array
    (~441 values vs. 1), so per-request payload size matters more here;
    single-year granularity also gives a natural resumability checkpoint
    for a fetch this expensive -- if it fails partway through the full
    77-year run, already-saved years are not re-fetched.

Requirements
------------
    pip install earthengine-api numpy

Run locally. This will take considerably longer than any previous fetch
in this project -- each year is its own request, 77 years total. If you
only want to test the pipeline first, set TEST_YEAR_ONLY below to a single
year before running.
"""

import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "t2m_fields"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SITE_LAT = 28.28531746075177
SITE_LON = 85.52515461726692

EVENT_YEAR = 2026
EVENT_MONTH = 8
EVENT_DAY = 26

HISTORICAL_START_YEAR = 1950
TRAINING_WINDOW_DAYS = 60  # days before Aug 26 each year, included in the fetch

BOX_HALF_WIDTH_DEG = 1.0  # -> 2 degree x 2 degree box, ~220 km x 220 km
FILL_VALUE = -9999.0      # sentinel for any masked/missing pixel; converted to NaN after fetch

# Set to a single year (e.g. 2024) to fetch only that year first, as a quick
# pipeline test before committing to the full 77-year run. Set back to None
# for the full run.
TEST_YEAR_ONLY = None


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


def fetch_year(year, region):
    """
    Fetch the 60-day pre-event daily t2m field for a single year, as a 2D
    array per day, stacked into shape (n_days, H, W). Returns (dates_array,
    fields_array) or (None, None) on failure.
    """
    import ee

    event_date = date(year, EVENT_MONTH, EVENT_DAY)
    start_date = event_date - timedelta(days=TRAINING_WINDOW_DAYS)

    collection = (
        ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        .select("temperature_2m")
        .filterDate(start_date.isoformat(), event_date.isoformat())
    )

    def extract_field(img):
        sampled = img.sampleRectangle(region=region, properties=[], defaultValue=FILL_VALUE)
        date_str = img.date().format("YYYY-MM-dd")
        return ee.Feature(None, {
            "date": date_str,
            "field": sampled.get("temperature_2m"),
        })

    try:
        result = ee.FeatureCollection(collection.map(extract_field)).getInfo()
    except Exception as exc:
        print(f"  {year}: FAILED -- {exc}")
        return None, None

    dates = []
    fields = []
    expected_shape = None
    for feature in result["features"]:
        props = feature["properties"]
        field_raw = props.get("field")
        if field_raw is None:
            continue
        field_array = np.array(field_raw, dtype=np.float32)
        if expected_shape is None:
            expected_shape = field_array.shape
        elif field_array.shape != expected_shape:
            print(f"  {year}: WARNING -- inconsistent field shape on {props['date']} "
                  f"({field_array.shape} vs expected {expected_shape}), skipping this day.")
            continue
        dates.append(props["date"])
        fields.append(field_array)

    if not fields:
        print(f"  {year}: FAILED -- no valid fields retrieved.")
        return None, None

    fields_array = np.stack(fields, axis=0)
    # Convert Kelvin to Celsius; leave the fill-value sentinel as-is until
    # after this conversion, then convert sentinel to NaN.
    fields_array = fields_array - 273.15
    fields_array = np.where(fields_array < (FILL_VALUE - 273.15) + 1, np.nan, fields_array)

    dates_array = np.array(dates)
    print(f"  {year}: {len(dates)} days, field shape {fields_array.shape[1:]}")
    return dates_array, fields_array


def fetch_all_years(historical_start_year, event_year):
    import ee

    print_section("STEP 2: Fetching spatial t2m fields, one year at a time")

    region = ee.Geometry.Rectangle([
        SITE_LON - BOX_HALF_WIDTH_DEG, SITE_LAT - BOX_HALF_WIDTH_DEG,
        SITE_LON + BOX_HALF_WIDTH_DEG, SITE_LAT + BOX_HALF_WIDTH_DEG,
    ])

    years = [TEST_YEAR_ONLY] if TEST_YEAR_ONLY is not None else list(range(historical_start_year, event_year + 1))
    if TEST_YEAR_ONLY is not None:
        print(f"TEST MODE: fetching only {TEST_YEAR_ONLY}. Set TEST_YEAR_ONLY = None for the full run.")

    start_time = time.time()
    n_fetched, n_skipped, n_failed = 0, 0, 0

    for year in years:
        output_path = DATA_DIR / f"t2m_fields_{year}.npz"
        if output_path.exists():
            print(f"  {year}: already fetched, skipping (delete {output_path.name} to re-fetch).")
            n_skipped += 1
            continue

        dates_array, fields_array = fetch_year(year, region)
        if dates_array is None:
            n_failed += 1
            continue

        np.savez_compressed(output_path, dates=dates_array, t2m_degc=fields_array)
        n_fetched += 1

    elapsed_minutes = (time.time() - start_time) / 60
    print(f"\nDone: {n_fetched} years fetched, {n_skipped} already present, {n_failed} failed. "
          f"Elapsed: {elapsed_minutes:.1f} minutes.")
    if n_failed > 0:
        print("Re-run this script to retry failed years -- already-fetched years will be skipped "
              "automatically, so this only re-attempts what's missing.")


def validate_fetched_data():
    """
    Load every saved .npz and check that field shapes are consistent across
    years (a VAE needs uniform input dimensions) and report any NaN
    fraction, so data quality issues surface now rather than during
    training.
    """
    print_section("STEP 3: Validating fetched data")

    files = sorted(DATA_DIR.glob("t2m_fields_*.npz"))
    if not files:
        print("No fetched files found.")
        return

    shapes = {}
    nan_fractions = {}
    for f in files:
        data = np.load(f)
        year = f.stem.split("_")[-1]
        shapes[year] = data["t2m_degc"].shape
        nan_fractions[year] = float(np.isnan(data["t2m_degc"]).mean())

    unique_field_shapes = set(s[1:] for s in shapes.values())
    print(f"Years with data: {len(files)}")
    print(f"Unique field (H, W) shapes across all years: {unique_field_shapes}")
    if len(unique_field_shapes) > 1:
        print("WARNING: inconsistent field shapes across years -- a VAE needs uniform input size. "
              "Inspect which years differ before training:")
        for year, shape in shapes.items():
            print(f"  {year}: {shape}")

    high_nan_years = {y: f for y, f in nan_fractions.items() if f > 0.01}
    if high_nan_years:
        print(f"\nYears with >1% NaN pixels (masked/missing data): {high_nan_years}")
    else:
        print("\nNo years with significant NaN content.")


if __name__ == "__main__":
    if not initialize_earth_engine():
        sys.exit(1)

    fetch_all_years(HISTORICAL_START_YEAR, EVENT_YEAR)
    validate_fetched_data()

    print_section("DONE")
    print("Fetch complete. Check for any WARNING or FAILED lines above before training the VAE.")