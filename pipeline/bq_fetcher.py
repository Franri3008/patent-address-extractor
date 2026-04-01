"""BigQuery patent fetcher.

Standalone usage:
  python pipeline/bq_fetcher.py --year 2025 --month 1          # batch
  python pipeline/bq_fetcher.py --patent WO2025086418          # individual
  python pipeline/bq_fetcher.py --year 2025 --month 1 --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from tqdm import tqdm

from utils.logger import get_logger

logger = get_logger("bq_fetcher");

_INDIVIDUAL_SQL = """
SELECT
  p.publication_number,
  tl.text AS title_text,
  tl.language AS title_language,
  p.application_number,
  p.country_code,
  p.publication_date,
  p.filing_date,
  p.grant_date,
  p.priority_date,
  (SELECT STRING_AGG(ih.name, ' & ')
     FROM UNNEST(p.inventor_harmonized) AS ih
  ) AS inventor_names,
  (SELECT STRING_AGG(ih.country_code, ' & ')
     FROM UNNEST(p.inventor_harmonized) AS ih
  ) AS inventor_countries,
  (SELECT STRING_AGG(ah.name, ' & ')
     FROM UNNEST(p.assignee_harmonized) AS ah
  ) AS assignee_names,
  (SELECT STRING_AGG(ah.country_code, ' & ')
     FROM UNNEST(p.assignee_harmonized) AS ah
  ) AS assignee_countries,
  (SELECT STRING_AGG(ipc.code, ' & ')
     FROM UNNEST(p.ipc) AS ipc
  ) AS ipc_codes,
  (SELECT STRING_AGG(cpc.code, ' & ')
     FROM UNNEST(p.cpc) AS cpc
  ) AS cpc_codes
FROM `patents-public-data.patents.publications` AS p
CROSS JOIN UNNEST(p.title_localized) AS tl
WHERE tl.language = 'en'
  AND REPLACE(p.publication_number, '-', '') LIKE @pat_pattern
ORDER BY p.publication_date ASC
""";


def _build_client(cfg_bq: dict) -> bigquery.Client:
    project = cfg_bq["project_id"];
    if cfg_bq.get("credentials_path"):
        creds = service_account.Credentials.from_service_account_file(
            cfg_bq["credentials_path"],
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        );
        return bigquery.Client(project=project, credentials=creds);
    return bigquery.Client(project=project);


def fetch_batch(config: dict, out_path: Path) -> list[dict]:
    """Query BigQuery for a full month of WO patents. Saves CSV and returns rows."""
    cfg_bq = config["bigquery"];
    batch = config["batch"];
    yyyy = str(batch["year"]);
    mm = str(batch["month"]).zfill(2);
    limit = batch.get("limit");

    sql_path = Path(__file__).parent.parent / "query.sql";
    sql = sql_path.read_text();
    sql = sql.replace("{yyyy}", yyyy).replace("{mm}", mm);
    if limit:
        sql = sql.rstrip().rstrip(";") + f"\nLIMIT {int(limit)}";

    logger.info(f"Querying BigQuery: WO patents {yyyy}-{mm} (limit={limit or 'none'})");
    client = _build_client(cfg_bq);
    job = client.query(sql);

    rows: list[dict] = [];
    pbar = tqdm(job.result(), desc="Fetching rows", unit="row", leave=False);
    for row in pbar:
        rows.append(dict(row));
        pbar.set_postfix(n=len(rows));

    df = pd.DataFrame(rows);
    out_path.parent.mkdir(parents=True, exist_ok=True);
    df.to_csv(out_path, index=False);
    logger.info(f"Saved {len(rows)} rows → {out_path}");
    return rows;


def fetch_individual(config: dict, patent_id: str) -> list[dict]:
    """Query BigQuery for a single patent by publication number."""
    cfg_bq = config["bigquery"];
    clean = patent_id.upper().replace("-", "");
    pattern = f"{clean}%";

    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("pat_pattern", "STRING", pattern)]
    );
    logger.info(f"Querying BigQuery for individual patent: {patent_id}");
    client = _build_client(cfg_bq);
    rows = [dict(r) for r in client.query(_INDIVIDUAL_SQL, job_config=job_config).result()];
    if not rows:
        logger.warning(f"Patent {patent_id} not found in BigQuery.");
    return rows;


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch WO patents from BigQuery.");
    parser.add_argument("--config", default="config.json");
    group = parser.add_mutually_exclusive_group(required=True);
    group.add_argument("--patent", metavar="ID", help="Single patent ID (individual mode).");
    group.add_argument("--year", type=int, help="Year for batch mode.");
    parser.add_argument("--month", type=int, help="Month for batch mode.");
    parser.add_argument("--limit", type=int, default=None);
    args = parser.parse_args();

    with open(args.config) as f:
        cfg = json.load(f);

    out_dir = Path(cfg["output"]["dir"]);

    if args.patent:
        rows = fetch_individual(cfg, args.patent);
        print(json.dumps(rows, indent=2, default=str));
    else:
        if not args.month:
            parser.error("--month is required with --year");
        cfg["batch"]["year"] = args.year;
        cfg["batch"]["month"] = args.month;
        if args.limit:
            cfg["batch"]["limit"] = args.limit;
        tmpl = cfg["output"]["filename_template"];
        fname = tmpl.format(yyyy=str(args.year), mm=str(args.month).zfill(2));
        fetch_batch(cfg, out_dir / f"raw_{fname}.csv");
