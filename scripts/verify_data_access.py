"""
verify_data_access.py

Purpose
-------
Stage 0 verification for the "Climate Attribution of the Nepal Ice-Rock Collapse
Preconditions" project. Before building any analysis, this script confirms:

  1. ERA5-Land is accessible via Google Earth Engine (GEE) for the source-zone
     grid cell, and reports the ACTUAL latest available date in the collection
     (this is the critical check -- ERA5-Land's finalized product in GEE lags
     real time by ~2-3 months, so the pre-event window in August 2026 may not
     yet be present).
  2. MODIS LST (MOD11A1) and Landsat Collection 2 Level-2 surface temperature
     are accessible for the site, as an independent cross-check on reanalysis
     air temperature in steep, high-elevation terrain.
  3. MODIS snow-cover (MOD10A1) and Sentinel-2 surface reflectance (for NDSI)
     are accessible for the site.
  4. Copernicus DEM (GLO-30) is accessible and returns a plausible elevation
     for the ~5200 m source zone, to precisely localize the grid cell(s).
  5. Whether any GTN-P (Global Terrestrial Network for Permafrost) monitoring
     site exists near this location, checked against GTN-P's own public
     metadata files rather than assumed absent.

Run this locally (VS Code). Nothing in this script performs any analysis --
it only checks access, date coverage, and data presence.

Requirements
------------
    pip install earthengine-api pandas requests

You will need to have already run `earthengine authenticate` once, or have
application-default credentials set up, for the GEE calls to succeed.
"""

import io
import sys
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Site definition
# ---------------------------------------------------------------------------
# Source zone of the August 26, 2026 ice-rock collapse, north face of Langtang
# Lirung, Nepal-Tibet border. Coordinates cross-checked against multiple
# independent public sources; consistent with the coordinates already used in
# the completed Transboundary Glacial Hazard Monitor project.
SITE_LAT = 28.28531746075177
SITE_LON = 85.52515461726692
SITE_LABEL = "Langtang Lirung source zone (Lehende Khola headwall)"
APPROX_SOURCE_ELEV_M = 5200  # reported source-zone elevation, for sanity check
EVENT_DATE = "2026-08-26"

# Bounding box for the broader region (used later for the VAE regional field
# in Method B, and here for the permafrost station proximity check).
REGION_MIN_LAT, REGION_MAX_LAT = 27.7, 28.7
REGION_MIN_LON, REGION_MAX_LON = 85.0, 86.1


def print_section(title):
    """Print a formatted section header so terminal output is easy to scan."""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Google Earth Engine initialization
# ---------------------------------------------------------------------------
def initialize_earth_engine():
    """
    Initialize the Earth Engine Python API using the existing GEE project
    handle from prior portfolio projects. Returns True on success.
    """
    print_section("STEP 1: Earth Engine initialization")
    try:
        import ee
    except ImportError:
        print("FAILED: earthengine-api is not installed. Run: pip install earthengine-api")
        return False

    try:
        ee.Initialize(project="carbon-verification-toolkit")
        print("OK: Earth Engine initialized with project 'carbon-verification-toolkit'.")
        return True
    except Exception as exc:
        print(f"FAILED to initialize with existing project handle: {exc}")
        print("Attempting ee.Authenticate() + default initialization...")
        try:
            ee.Authenticate()
            ee.Initialize(project="carbon-verification-toolkit")
            print("OK: Earth Engine initialized after interactive authentication.")
            return True
        except Exception as exc2:
            print(f"FAILED: {exc2}")
            return False


# ---------------------------------------------------------------------------
# 2. ERA5-Land access and latency check (the critical check)
# ---------------------------------------------------------------------------
def check_era5_land():
    """
    Check ERA5-Land Daily Aggregated access via GEE. Reports:
      - whether the collection is reachable at all
      - the date of the MOST RECENT image in the collection (critical: this
        tells us whether the August 2026 pre-event window is actually present
        yet, given ERA5-Land's known 2-3 month finalization lag)
      - a sampled t2m value at the site for a known historical date, to
        confirm the data pipeline works end to end
    """
    import ee

    print_section("STEP 2: ERA5-Land access and latency check")
    try:
        collection = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
        point = ee.Geometry.Point([SITE_LON, SITE_LAT])

        # Find the most recent image date currently in the collection.
        most_recent = collection.sort("system:time_start", False).first()
        most_recent_date = ee.Date(most_recent.get("system:time_start")).format("YYYY-MM-dd").getInfo()
        print(f"OK: Collection reachable. Most recent available date: {most_recent_date}")

        event_dt = datetime.strptime(EVENT_DATE, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        latest_dt = datetime.strptime(most_recent_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        gap_days = (event_dt - latest_dt).days

        if latest_dt >= event_dt:
            print(f"RESULT: Finalized ERA5-Land data COVERS the event date ({EVENT_DATE}).")
        else:
            print(
                f"RESULT: Finalized ERA5-Land data STOPS {gap_days} days BEFORE the event date "
                f"({EVENT_DATE}). The pre-event window will need ERA5-Land-T via the CDS API "
                f"directly, not GEE, until the finalized product catches up."
            )

        # Sample a known historical date to confirm the data pipeline works.
        test_date = "2020-08-26"
        test_image = collection.filterDate(test_date, "2020-08-27").first()
        t2m_kelvin = test_image.select("temperature_2m").reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=9000
        ).get("temperature_2m").getInfo()
        t2m_degc = t2m_kelvin - 273.15
        print(f"Sanity check ({test_date}): t2m_degc = {t2m_degc:.2f} deg C at site")

    except Exception as exc:
        print(f"FAILED: {exc}")


# ---------------------------------------------------------------------------
# 3. MODIS and Landsat land surface temperature (independent cross-check)
# ---------------------------------------------------------------------------
def check_lst_cross_check():
    """
    Check MODIS MOD11A1 and Landsat Collection 2 Level-2 surface temperature
    access for the site. In steep, high-elevation terrain, reanalysis products
    like ERA5-Land are known to have systematic biases, so these serve as an
    independent cross-check -- but note upfront that cloud cover and terrain
    shadow are likely to cause substantial data gaps at this site.
    """
    import ee

    print_section("STEP 3: MODIS / Landsat LST cross-check access")
    point = ee.Geometry.Point([SITE_LON, SITE_LAT])

    try:
        modis_lst = ee.ImageCollection("MODIS/061/MOD11A1").filterDate("2026-06-01", "2026-08-26")
        modis_count = modis_lst.size().getInfo()
        print(f"OK: MODIS MOD11A1 reachable. {modis_count} daily images found, 2026-06-01 to {EVENT_DATE}.")

        valid_pixel_count = 0
        modis_list = modis_lst.toList(modis_count)
        for i in range(min(modis_count, 10)):  # spot-check first 10 images only
            img = ee.Image(modis_list.get(i))
            lst_raw = img.select("LST_Day_1km").reduceRegion(
                reducer=ee.Reducer.first(), geometry=point, scale=1000
            ).get("LST_Day_1km").getInfo()
            if lst_raw is not None:
                valid_pixel_count += 1
        print(f"Spot check: {valid_pixel_count}/10 sampled MODIS LST images have a valid (non-masked) "
              f"pixel at the exact site coordinate -- expect this to be low given cloud cover at 5200 m.")

    except Exception as exc:
        print(f"FAILED (MODIS MOD11A1): {exc}")

    try:
        landsat = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(point).filterDate(
            "2026-01-01", "2026-08-26"
        )
        landsat_count = landsat.size().getInfo()
        print(f"OK: Landsat 8 C2L2 reachable. {landsat_count} scenes intersecting site, "
              f"2026-01-01 to {EVENT_DATE}.")
    except Exception as exc:
        print(f"FAILED (Landsat C2L2): {exc}")


# ---------------------------------------------------------------------------
# 4. Snow-cover data access (MODIS and Sentinel-2)
# ---------------------------------------------------------------------------
def check_snow_cover():
    """
    Check MODIS MOD10A1 snow-cover and Sentinel-2 surface reflectance access
    (Sentinel-2 has no native snow product; NDSI would need to be computed
    from bands B3 and B11 if used).
    """
    import ee

    print_section("STEP 4: Snow-cover data access")
    point = ee.Geometry.Point([SITE_LON, SITE_LAT])

    try:
        modis_snow = ee.ImageCollection("MODIS/061/MOD10A1").filterDate("2026-06-01", "2026-08-26")
        snow_count = modis_snow.size().getInfo()
        print(f"OK: MODIS MOD10A1 reachable. {snow_count} daily images found, 2026-06-01 to {EVENT_DATE}.")
    except Exception as exc:
        print(f"FAILED (MODIS MOD10A1): {exc}")

    try:
        s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(point).filterDate(
            "2026-06-01", "2026-08-26"
        )
        s2_count = s2.size().getInfo()
        print(f"OK: Sentinel-2 SR reachable. {s2_count} scenes intersecting site, "
              f"2026-06-01 to {EVENT_DATE}. NDSI would need computing from B3/B11.")
    except Exception as exc:
        print(f"FAILED (Sentinel-2 SR): {exc}")


# ---------------------------------------------------------------------------
# 5. Copernicus DEM access and elevation sanity check
# ---------------------------------------------------------------------------
def check_dem():
    """
    Check Copernicus DEM GLO-30 access and confirm the sampled elevation at
    the site coordinates is consistent with the reported ~5200 m source zone.
    Samples both the exact point and the maximum elevation within a small
    neighborhood, since a single-pixel sample on a steep slope can land on a
    lower shoulder rather than the true detachment scar. Uses the current
    (non-deprecated) GLO30_2024_1 asset.
    """
    import ee

    print_section("STEP 5: Copernicus DEM access and elevation check")
    point = ee.Geometry.Point([SITE_LON, SITE_LAT])
    neighborhood = point.buffer(200)  # 200 m radius, steep terrain sanity check

    try:
        dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").filterBounds(point).mosaic()
        elev_at_point_m = dem.select("DEM").reduceRegion(
            reducer=ee.Reducer.first(), geometry=point, scale=30
        ).get("DEM").getInfo()
        elev_max_nearby_m = dem.select("DEM").reduceRegion(
            reducer=ee.Reducer.max(), geometry=neighborhood, scale=30
        ).get("DEM").getInfo()
        print(f"OK: Copernicus DEM (GLO30_2024_1) reachable.")
        print(f"  Elevation at exact coordinate: {elev_at_point_m:.0f} m")
        print(f"  Maximum elevation within 200 m radius: {elev_max_nearby_m:.0f} m")
        print(f"  Reported source-zone elevation: ~{APPROX_SOURCE_ELEV_M} m")

        diff_point = abs(elev_at_point_m - APPROX_SOURCE_ELEV_M)
        diff_max = abs(elev_max_nearby_m - APPROX_SOURCE_ELEV_M)
        if diff_max > 300:
            print(f"NOTE: Even the nearby maximum is {diff_max:.0f} m from the reported elevation -- "
                  f"the coordinate itself likely needs re-checking against the original source "
                  f"(not just a DEM sampling artifact).")
        elif diff_point > 300 and diff_max <= 300:
            print(f"NOTE: The exact point is {diff_point:.0f} m off, but a nearby cell within 200 m "
                  f"reaches within {diff_max:.0f} m of the reported elevation -- consistent with the "
                  f"coordinate landing slightly off the true detachment scar on a steep slope, not a "
                  f"data problem.")
    except Exception as exc:
        print(f"FAILED: {exc}")


# ---------------------------------------------------------------------------
# 6. GTN-P permafrost station proximity check (no GEE required)
# ---------------------------------------------------------------------------
def _load_pangaea_site_table(doi, label):
    """
    Fetch a PANGAEA dataset as a tab-delimited text file and parse it into a
    DataFrame. PANGAEA text-file exports have a variable-length header block
    terminated by a line of exactly "*/" -- this finds that line and reads
    the tab-separated table beneath it. Returns None on any failure, with the
    reason printed, rather than silently reporting an empty/misleading result.
    """
    url = f"https://doi.pangaea.de/{doi}?format=textfile"
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    # PANGAEA sometimes ignores format=textfile and returns a zip archive
    # containing the .txt file instead. Detect the zip magic bytes and
    # extract the inner text file if so.
    if response.content[:2] == b"PK":
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        txt_names = [n for n in archive.namelist() if n.lower().endswith(".txt")]
        if not txt_names:
            print(f"  {label}: FAILED -- response was a zip but contained no .txt file. "
                  f"Contents: {archive.namelist()}")
            return None
        with archive.open(txt_names[0]) as f:
            text = f.read().decode("utf-8", errors="replace")
    else:
        text = response.text

    # Find the PANGAEA header terminator as an exact standalone line ("*/"),
    # not a raw substring search -- a substring search can match "*/" inside
    # unrelated metadata text elsewhere in the file and skip past the real
    # header into the data rows.
    lines = text.split("\n")
    marker_line_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "*/":
            marker_line_idx = i
            break

    if marker_line_idx is None:
        print(f"  {label}: FAILED -- no line containing only '*/' found. This dataset's export "
              f"format may differ from the expected PANGAEA layout.")
        return None

    # The header may not be the very next line, so search forward from the
    # marker for the first line that looks like a header row.
    header_idx = None
    for i in range(marker_line_idx + 1, min(marker_line_idx + 20, len(lines))):
        fields = [f.strip().lower() for f in lines[i].split("\t")]
        if len(fields) > 3 and any("latitude" in f for f in fields):
            header_idx = i
            break

    if header_idx is None:
        print(f"  {label}: FAILED -- could not locate a header row containing 'Latitude' within "
              f"20 lines of the terminator. Lines after terminator: {lines[marker_line_idx+1:marker_line_idx+4]}")
        return None

    table_text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(table_text), sep="\t")
    print(f"  {label}: OK -- parsed {len(df)} rows, columns: {list(df.columns)}")
    return df


def check_gtn_p_coverage():
    """
    Download GTN-P's own public site-metadata tables directly from their
    PANGAEA DOIs (CALM active-layer sites: 10.1594/PANGAEA.842815; TSP
    boreholes: 10.1594/PANGAEA.842820 -- these are the actual per-site tables;
    the parent DOI 10.1594/PANGAEA.842821 only returns a summary page, not
    the data) and check whether any site falls within the regional bounding
    box. This confirms coverage rather than assuming it.
    """
    print_section("STEP 6: GTN-P permafrost station proximity check")

    datasets = {
        "CALM active-layer sites": "10.1594/PANGAEA.842815",
        "TSP boreholes": "10.1594/PANGAEA.842820",
    }

    any_table_loaded = False
    for label, doi in datasets.items():
        try:
            df = _load_pangaea_site_table(doi, label)
        except Exception as exc:
            print(f"  {label}: FAILED to download/parse -- {exc}")
            continue
        if df is None:
            continue

        lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
        lon_col = next((c for c in df.columns if "lon" in c.lower()), None)
        if lat_col is None or lon_col is None:
            print(f"  {label}: loaded but no latitude/longitude column found -- "
                  f"inspect columns above manually.")
            continue

        any_table_loaded = True
        df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
        df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
        nearby = df[
            (df[lat_col] >= REGION_MIN_LAT) & (df[lat_col] <= REGION_MAX_LAT) &
            (df[lon_col] >= REGION_MIN_LON) & (df[lon_col] <= REGION_MAX_LON)
        ]
        print(f"  {label}: {len(nearby)} of {len(df)} sites fall within the regional bounding box "
              f"(lat {REGION_MIN_LAT}-{REGION_MAX_LAT}, lon {REGION_MIN_LON}-{REGION_MAX_LON}).")
        if len(nearby) > 0:
            print(nearby.to_string())

    if any_table_loaded:
        print("\nIf both counts above are 0, there is no GTN-P CALM site or TSP borehole near this "
              "location in the network's own published metadata.")
    else:
        print("\nINCONCLUSIVE -- neither table parsed successfully. Do not treat this as confirming "
              "an absence of coverage. Manual fallback: browse https://data.gtn-p.org/ and visually "
              f"check the map for lat {REGION_MIN_LAT}-{REGION_MAX_LAT}, lon {REGION_MIN_LON}-{REGION_MAX_LON}.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Verifying data access for: {SITE_LABEL}")
    print(f"Coordinates: {SITE_LAT}, {SITE_LON}")

    ee_ready = initialize_earth_engine()
    if ee_ready:
        check_era5_land()
        check_lst_cross_check()
        check_snow_cover()
        check_dem()
    else:
        print("\nSkipping GEE-dependent checks (Steps 2-5) since Earth Engine did not initialize.")

    check_gtn_p_coverage()

    print_section("DONE")
    print("Data verification complete. Review the output above before starting Method A.")