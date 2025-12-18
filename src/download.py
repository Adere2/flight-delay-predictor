import argparse
import calendar
import io
import os
import time
import zipfile

import requests
import urllib3
from bs4 import BeautifulSoup
from tqdm import tqdm

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = (
    "https://transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr"
)

# The columns from your network analysis
COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER",
    "ORIGIN_AIRPORT_ID",
    "ORIGIN",
    "DEST_AIRPORT_ID",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_DELAY",
    "DEP_DEL15",
    "CRS_ARR_TIME",
    "ARR_DELAY",
    "ARR_DEL15",
    "CANCELLED",
    "DIVERTED",
    "DISTANCE",
]


def get_tokens(session):
    """Get fresh ASP.NET tokens using BeautifulSoup"""
    r = session.get(URL, verify=False)
    try:
        soup = BeautifulSoup(r.text, "html.parser")
        viewstate = soup.find("input", {"id": "__VIEWSTATE"})["value"]
        eventval = soup.find("input", {"id": "__EVENTVALIDATION"})["value"]
        generator = soup.find("input", {"id": "__VIEWSTATEGENERATOR"})["value"]
        return viewstate, eventval, generator
    except Exception as e:
        print(f"Error finding tokens: {e}")
        return None, None, None


def download_month_data(session, year, month, output_dir, vs, ev, gen):
    """Attempt to download data for a specific year and month.
    Returns True if successful, False otherwise."""
    month_name = calendar.month_name[month]
    filename = f"{month}_{year}_ontime.csv"
    filepath = os.path.join(output_dir, filename)

    # Build Payload
    payload = {
        "__VIEWSTATE": vs,
        "__VIEWSTATEGENERATOR": gen,
        "__EVENTVALIDATION": ev,
        "__EVENTTARGET": "",
        "__EVENTARGUMENT": "",
        "cboYear": str(year),
        "cboPeriod": str(month),
        "cboGeography": "All",
        "btnDownload": "Download",
    }
    for col in COLUMNS:
        payload[col] = "on"

    try:
        r = session.post(URL, data=payload, verify=False, stream=True)

        # Check for ZIP content
        if "zip" in r.headers.get("Content-Type", "").lower():
            # Get total file size
            total_size = int(r.headers.get("content-length", 0))

            # Download with progress bar
            content = io.BytesIO()
            with tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        content.write(chunk)
                        pbar.update(len(chunk))

            # Extract CSV from ZIP
            with zipfile.ZipFile(content) as z:
                csv_name = [f for f in z.namelist() if f.endswith(".csv")][0]
                z.extract(csv_name, output_dir)
                os.rename(os.path.join(output_dir, csv_name), filepath)
            print(f"✓ Downloaded {month_name} {year}")
            return True
        else:
            return False

    except Exception as e:
        print(f"Error downloading {month_name} {year}: {e}")
        return False


def download_all_months(year, output_dir, num_months=12):
    os.makedirs(output_dir, exist_ok=True)
    session = requests.Session()

    print(f"--- Downloading {year} Data ({num_months} months) ---")

    # 1. Handshake (Get Tokens)
    vs, ev, gen = get_tokens(session)
    if not vs:
        return

    # 2. Loop through months
    for month in range(1, num_months + 1):
        month_name = calendar.month_name[month]
        filename = f"{month}_{year}_ontime.csv"
        filepath = os.path.join(output_dir, filename)

        if os.path.exists(filepath):
            print(f"⊘ Skipping {month_name} {year} (already exists)")
            continue

        print(f"Downloading {month_name} {year}...")

        # Try to download current year
        success = download_month_data(session, year, month, output_dir, vs, ev, gen)

        # If not available, try previous year
        if not success:
            print(f"  {month_name} {year} not available, trying {year - 1}...")
            fallback_filename = f"{month}_{year - 1}_ontime.csv"
            fallback_filepath = os.path.join(output_dir, fallback_filename)

            if os.path.exists(fallback_filepath):
                print(f"  ⊘ Skipping {month_name} {year - 1} (already exists)")
            else:
                success = download_month_data(
                    session, year - 1, month, output_dir, vs, ev, gen
                )
                if not success:
                    print(f"  ✗ {month_name} {year - 1} also not available")

        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download flight data CSV files from BTS"
    )
    parser.add_argument(
        "num_months",
        type=int,
        nargs="?",
        default=12,
        help="Number of months to download (default: 12)",
    )
    args = parser.parse_args()

    # Validate input
    if args.num_months < 1 or args.num_months > 12:
        print("Error: num_months must be between 1 and 12")
        exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src/
    output_dir = os.path.join(project_root, "data", "raw")
    download_all_months(2025, output_dir, args.num_months)
