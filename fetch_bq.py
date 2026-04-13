"""
Standalone BigQuery fetcher for WO patent metadata.

Usage:
  # Single month (same as existing pipeline)
  python fetch_bq.py --year 2022 --month 2

  # Full year in one query
  python fetch_bq.py --year 2022

  # Multiple years in one query
  python fetch_bq.py --years 2020-2024

  # Limit rows (useful for testing)
  python fetch_bq.py --year 2022 --limit 100

Output goes to output/raw_patents_YYYY_MM.csv (per-month)
or output/raw_patents_YYYY.csv (per-year).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account


ROOT = Path(__file__).parent
QUERY_TEMPLATE = (ROOT / "query.sql").read_text()


def _build_client(cfg_bq: dict) -> bigquery.Client:
    project = cfg_bq["project_id"]
    if cfg_bq.get("credentials_path"):
        creds = service_account.Credentials.from_service_account_file(
            cfg_bq["credentials_path"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        return bigquery.Client(project=project, credentials=creds)
    return bigquery.Client(project=project)


def _build_sql(start_year: int, start_month: int,
               end_year: int, end_month: int,
               limit: int | None = None) -> str:
    """Build SQL with a date range replacing the single-month placeholders."""
    # Replace the original WHERE date clause with a range
    date_start = f"{start_year}{str(start_month).zfill(2)}00"
    date_end = f"{end_year}{str(end_month).zfill(2)}32"

    sql = QUERY_TEMPLATE.replace(
        "{yyyy}{mm}00 AND {yyyy}{mm}32",
        f"{date_start} AND {date_end}",
    )

    if limit:
        sql = sql.rstrip().rstrip(";")
        sql += f"\nLIMIT {limit}\n"

    return sql


def fetch(cfg_bq: dict, sql: str, label: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame."""
    import time

    print(f"[fetch_bq] Querying BigQuery: {label}")
    client = _build_client(cfg_bq)
    job = client.query(sql)
    print(f"[fetch_bq] Job created: {job.job_id}")
    print(f"[fetch_bq] Waiting for BigQuery to finish scanning (this can take 1-2 min)...")

    # Poll job status until rows start arriving
    t0 = time.time()
    while not job.done():
        elapsed = time.time() - t0
        print(f"  ... query running ({elapsed:.0f}s elapsed)", end="\r", flush=True)
        time.sleep(2)

    elapsed = time.time() - t0
    print(f"\n[fetch_bq] Query done in {elapsed:.1f}s. Streaming rows...")

    rows = []
    t1 = time.time()
    for row in job.result():
        rows.append(dict(row))
        if len(rows) % 1000 == 0:
            rate = len(rows) / (time.time() - t1)
            print(f"  ... {len(rows):,} rows fetched ({rate:.0f} rows/s)", end="\r", flush=True)

    print(f"\n[fetch_bq] Fetched {len(rows):,} rows for {label} in {time.time() - t0:.1f}s total")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch WO patents from BigQuery (standalone)."
    )
    parser.add_argument("--config", default="config.json",
                        help="Path to config.json (for BQ credentials).")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--year", type=int,
                       help="Single year. Add --month for a single month.")
    group.add_argument("--years", type=str,
                       help="Year range, e.g. 2020-2024.")

    parser.add_argument("--month", type=int, default=None,
                        help="Month (1-12). Only with --year. Omit for full year.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to fetch (applied in SQL).")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: from config).")
    parser.add_argument("--single-file", action="store_true",
                        help="Save everything in one CSV instead of per-year files.")

    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    cfg_bq = cfg["bigquery"]
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine date range
    if args.years:
        parts = args.years.split("-")
        if len(parts) != 2:
            parser.error("--years must be in format YYYY-YYYY, e.g. 2020-2024")
        start_year, end_year = int(parts[0]), int(parts[1])
        if start_year > end_year:
            parser.error("Start year must be <= end year")
        start_month, end_month = 1, 12
        label = f"WO patents {start_year}-{end_year}"
    elif args.month:
        start_year = end_year = args.year
        start_month = end_month = args.month
        label = f"WO patents {args.year}-{str(args.month).zfill(2)}"
    else:
        start_year = end_year = args.year
        start_month, end_month = 1, 12
        label = f"WO patents {args.year} (full year)"

    sql = _build_sql(start_year, start_month, end_year, end_month, args.limit)
    df = fetch(cfg_bq, sql, label)

    if df.empty:
        print("[fetch_bq] No rows returned.")
        sys.exit(0)

    # Save output
    if args.single_file or args.month:
        # Single file output
        if args.month:
            fname = f"raw_patents_{args.year}_{str(args.month).zfill(2)}.csv"
        elif args.years:
            fname = f"raw_patents_{start_year}_{end_year}.csv"
        else:
            fname = f"raw_patents_{start_year}.csv"
        path = out_dir / fname
        df.to_csv(path, index=False)
        print(f"[fetch_bq] Saved {len(df)} rows -> {path}")
    else:
        # Split by year and save separate files
        df["_year"] = df["publication_date"].astype(str).str[:4]
        for year, group in df.groupby("_year"):
            group = group.drop(columns=["_year"])
            fname = f"raw_patents_{year}.csv"
            path = out_dir / fname
            group.to_csv(path, index=False)
            print(f"[fetch_bq] Saved {len(group)} rows -> {path}")

    print("[fetch_bq] Done.")


if __name__ == "__main__":
    main()
