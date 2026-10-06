"""STEP 4 - Download the raw GCMT catalogue.

LEVEL 3 input, part 1 of 2. Before any filtering, we need the catalogue
itself: every earthquake with a Global Centroid Moment Tensor solution,
worldwide, for the PTHA18 observation window.

Where the data comes from
--------------------------
The Global CMT project (globalcmt.org), which publishes monthly "NDK" text
catalogue files (one 5-line record per event: hypocentre, centroid, moment
tensor, and the derived strike/dip/rake of both nodal planes). We pull the
pre-built quarterly cumulative files from their `CMT5` archive, which is the
same catalogue PTHA18's own gcmt_subsetter.R was run against.

Window
------
1976-01-01 to 2017-03-01, PTHA18's own observation window (41.1636 years,
report Section 3.7.3). v1 of this pipeline set WINDOW_END to today instead,
which extended the catalogue nearly a decade past PTHA18's cutoff and made
the LEVEL 3 event count incomparable to the official 9-event figure by
construction: "official" duration and "official" event count are a matched
pair (see official_gcmt_observations.csv's provenance notes), and a from-
scratch run needs the SAME pair to be judged against it. v2 uses the official
end date so this run's LEVEL 3 update is actually comparable to PTHA18's.
Steps 5 and 6 read WINDOW_END from here, so the whole pipeline stays
consistent with whatever window is set below.

What this script does NOT do
-----------------------------
It does not filter by location, focal mechanism or magnitude. That is step 5.
This step's only job is: get the full worldwide catalogue onto disk, parsed
into one row per event, so step 5 has something to filter.

Run from ptha18_logic_tree_test/:
    .venv/Scripts/python.exe examples/kermadectonga2/steps/step4_fetch_gcmt.py
"""

import datetime
import os
import re
import ssl
import sys
import urllib.request

import certifi
import pandas as pd

# On some Windows Python installs, ssl.create_default_context() falls back to
# a system cert store that doesn't complete ldeo.columbia.edu's chain (root:
# eMudhra emSign Root CA - G1), even though `certifi` ships that exact root
# and `openssl s_client` verifies fine with it. Point urllib at certifi's
# bundle explicitly rather than relying on the OS default.
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.abspath(os.path.join(HERE, ".."))
GCMT_DIR = os.path.join(EXAMPLE, "data", "gcmt")
OUT_CSV = os.path.join(GCMT_DIR, "gcmt_catalogue_1976_present.csv")

# PTHA18's own observation window (report Section 3.7.3). Steps 5-6 read this
# too, so LEVEL 3's count and duration match what the official run compares
# against. See the module docstring for why v1's WINDOW_END = today made
# this incomparable.
WINDOW_START = "1976-01-01"
WINDOW_END = "2017-03-01"

# GCMT ships the catalogue in two pieces: one large cumulative file (updated
# every few years) covering the older era, plus the current year's events in
# a separate "quick CMT" monthly-cumulative file. Both are downloaded and
# concatenated so WINDOW_END can reach today.
NDK_SOURCES = [
    # 1976 through the last full cumulative release.
    ("https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/jan76_dec20.ndk",
     "jan76_dec20.ndk"),
    # NDK monthly files for everything the cumulative file above does not
    # cover yet (2021 onward), one request per month up to the current one.
]


def _monthly_ndk_urls():
    start = datetime.date(2021, 1, 1)
    today = datetime.date.today()
    urls = []
    y, m = start.year, start.month
    months = "jan feb mar apr may jun jul aug sep oct nov dec".split()
    while (y, m) <= (today.year, today.month):
        mon = months[m - 1]
        yy = y % 100
        urls.append((
            f"https://www.ldeo.columbia.edu/~gcmt/projects/CMT/catalog/NEW_MONTHLY/"
            f"{y}/{mon}{yy:02d}.ndk",
            f"{mon}{yy:02d}.ndk"))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return urls


def download_ndk():
    """Download the cumulative file plus every monthly file since, cached."""
    os.makedirs(GCMT_DIR, exist_ok=True)
    all_sources = list(NDK_SOURCES) + _monthly_ndk_urls()

    combined = []
    for url, fname in all_sources:
        cached = os.path.join(GCMT_DIR, fname)
        if os.path.exists(cached):
            with open(cached, "r", encoding="latin-1") as f:
                combined.append(f.read())
            continue
        try:
            print(f"  downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180, context=SSL_CONTEXT) as resp:
                text = resp.read().decode("latin-1")
            with open(cached, "w", encoding="latin-1") as f:
                f.write(text)
            print(f"    {len(text):,} characters -> {cached}")
            combined.append(text)
        except Exception as e:  # noqa: BLE001 - a missing recent month is normal
            print(f"    skipped ({e})")

    if not combined:
        raise SystemExit(
            "could not download any part of the GCMT catalogue.\n"
            "If ldeo.columbia.edu is unreachable from this network, download "
            f"NDK files manually into:\n  {GCMT_DIR}")
    return "\n".join(combined)


# NDK format: 5 fixed-width lines per event. See globalcmt.org/CMTfiles.html.
#   line 1: hypocentre   - source, date, time, lat, lon, depth, mags, location
#   line 4: best-fit CMT - exponent + 6 moment tensor components
#   line 5: derived      - version, eigenvalues/vectors, then scalar moment
#                           and the two nodal planes: strike1 dip1 rake1
#                           strike2 dip2 rake2
LINE1_RE = re.compile(
    r"^\S+\s+(\d{4})/(\d{2})/(\d{2})\s+(\d{2}):(\d{2}):([\d.]+)\s+"
    r"(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)")


def parse_ndk(text):
    lines = text.splitlines()
    n_records = len(lines) // 5
    rows = []
    for i in range(n_records):
        block = lines[i * 5:(i + 1) * 5]
        if len(block) < 5 or not block[0].strip():
            continue
        m = LINE1_RE.match(block[0])
        if not m:
            continue
        yr, mo, da, hh, mm, ss, lat, lon, depth = m.groups()

        # Line 5, last 12 whitespace-separated fields: str1 dip1 rake1
        # str2 dip2 rake2 (each preceded by its uncertainty in the raw
        # field layout, but split() on whitespace still yields them as the
        # trailing tokens for every well-formed record).
        parts5 = block[4].split()
        if len(parts5) < 6:
            continue
        try:
            strike1, dip1, rake1, strike2, dip2, rake2 = (
                float(x) for x in parts5[-6:])
        except ValueError:
            continue

        rows.append({
            "date": f"{yr}-{mo}-{da}",
            "lat": float(lat), "lon": float(lon), "depth_km": float(depth),
            "strike1": strike1, "dip1": dip1, "rake1": rake1,
            "strike2": strike2, "dip2": dip2, "rake2": rake2,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("STEP 4 - Download the raw GCMT catalogue")
    print("=" * 70)

    text = download_ndk()

    print("\n  parsing NDK records ...")
    df = parse_ndk(text)
    print(f"  {len(df)} events parsed (worldwide, all magnitudes/depths)")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    before = len(df)
    df = df[(df["date"] >= WINDOW_START) & (df["date"] <= WINDOW_END)]
    df = df.drop_duplicates(subset=["date", "lat", "lon", "depth_km"])
    print(f"  {len(df)} of {before} fall in [{WINDOW_START}, {WINDOW_END}] "
          f"(deduplicated across the overlapping monthly/cumulative files)")

    os.makedirs(GCMT_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\n  saved -> {OUT_CSV}")
    print("\nNext:  step5_subset_gcmt.py")


if __name__ == "__main__":
    sys.exit(main())
