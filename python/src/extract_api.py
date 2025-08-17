# main.py
import os
import io
import json
import gzip
from datetime import datetime
from pathlib import Path
import logging
from logging.handlers import TimedRotatingFileHandler
import requests
import click

import boto3
from botocore.exceptions import ClientError
import psycopg2
import polars as pl
from polars.exceptions import ComputeError

try:
    BASE_DIR = Path(__file__).parent.parent.parent.resolve()
except NameError:
    # Fallback for interactive sessions (no __file__)
    BASE_DIR = Path(os.getcwd()).parent.parent.parent.resolve()
LOG_DIR = BASE_DIR / "logs"


def setup_logging(
    log_dir: str = LOG_DIR,
    logger_name: str = "dash_extract",
    when: str = "midnight",
    interval: int = 1,
    backup_count: int = 7,
    level: int = logging.INFO,
):
    """
    Setup logging with both console and time-based rotating file handlers.

    Args:
        log_dir (str): directory where logs are stored
        log_file (str): base log file name
        when (str): rotation interval type (e.g., 'S','M','H','D','midnight','W0'-'W6')
        interval (int): number of intervals between rotations
        backup_count (int): how many old logs to keep
        level (int): logging level
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = f"{logger_name}.log"
    log_path = os.path.join(log_dir, log_file)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Rotating file handler
    file_handler = TimedRotatingFileHandler(
        log_path,
        when=when,
        interval=interval,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Root logger config
    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    return logging.getLogger(logger_name)


def get_credentials():
    with open(BASE_DIR / "env.json", "r") as file:
        creds = json.load(file)
    return creds


credentials = get_credentials()

DAYSMART_CLIENT_ID = credentials["DAYSMART_CLIENT_ID"]
DAYSMART_CLIENT_SECRET = credentials["DAYSMART_CLIENT_SECRET"]
DAYSMART_API_GRANT_TYPE = "client_credentials"
POSTGRES_DB = credentials["POSTGRES_DB"]
POSTGRES_USER = credentials["POSTGRES_USER"]
POSTGRES_PASSWORD = credentials["POSTGRES_PASSWORD"]
S3_BUCKET = credentials["S3_BUCKET"]
BASE_URL = "https://api.dashplatform.com"
LOG = setup_logging()


def get_bearer_token():
    url = BASE_URL + "/v1/auth/token"
    headers = {"Content-Type": "application/vnd.api+json"}
    payload = {
        "grant_type": DAYSMART_API_GRANT_TYPE,
        "client_id": DAYSMART_CLIENT_ID,
        "client_secret": DAYSMART_CLIENT_SECRET,
    }
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get("access_token")
    else:
        raise Exception(response.json())


# -------------------------
# Fetcher
# -------------------------
def get_index_data(end_point, bearer_token):
    url = BASE_URL + "/v1/" + end_point
    headers = {
        "Content-Type": "application/vnd.api+json",
        "Authorization": f"Bearer {bearer_token}",
    }
    params = {"company": "allstars", "sort": "id"}
    all_data = []
    next_url = url
    while next_url:
        response = requests.get(next_url, headers=headers, params=params)

        if response.status_code == 200:
            data = response.json()

            # Append the current page's data (assuming it's under a 'data' key)
            if "data" in data:
                all_data.extend(data["data"])

            # Update the next_url for the next iteration
            next_url = data.get("links", {}).get("next")
        else:
            raise Exception(response.json())

    return all_data


# -------------------------
# JSON → Polars
# -------------------------
def records_to_polars(records: list[dict]) -> pl.DataFrame:
    """
    Lifts JSON:API objects (id/type + attributes.*).
    Builds a Polars DF with full-length schema inference.
    If mixed types are detected across rows, falls back to string-cast.
    """
    if not records:
        return pl.DataFrame([])

    def lift_one(r: dict):
        if isinstance(r, dict) and "attributes" in r:
            base = {
                k: v for k, v in r.items() if k not in ("attributes", "relationships")
            }
            # base.update(r.get("attributes", {}))
            return base
        return r

    lifted = [lift_one(r) for r in records]

    # First attempt: infer using full data scan
    try:
        df = pl.from_dicts(lifted, infer_schema_length=None)  # scan all rows
    except ComputeError:
        # Fallback: cast every value to string (JSON for nested), preserving None
        def to_str(v):
            if v is None:
                return None
            # stringify nested structures to preserve info for CSV/SQL
            if isinstance(v, (dict, list)):
                return json.dumps(v, separators=(",", ":"), default=str)
            return str(v)

        lifted_str = [{k: to_str(v) for k, v in row.items()} for row in lifted]
        df = pl.from_dicts(lifted_str, infer_schema_length=None)

    # Unnest any 1-level struct columns if they exist (after first path)
    struct_cols = [
        c for c, dt in zip(df.columns, df.dtypes) if isinstance(dt, pl.Struct)
    ]
    for c in struct_cols:
        df = df.unnest(c)

    # Put id first if present
    if "id" in df.columns:
        df = df.select(["id", *[c for c in df.columns if c != "id"]])

    now = datetime.now()  # timezone-aware UTC timestamp

    df = df.with_columns(
        [
            pl.lit(now).alias("inserted_dt"),
            pl.lit(now).alias("updated_dt"),
        ]
    )

    return df


# -------------------------
# S3 Writers
# -------------------------
def write_jsonl_to_s3(records: list[dict], bucket: str, key: str, s3_client=None):
    """
    Writes list-of-dicts as JSONL.gz to s3://bucket/key.
    """
    if not records:
        return
    if s3_client is None:
        s3_client = boto3.Session(profile_name="etl_user").client("s3")

    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for r in records:
            line = (json.dumps(r, separators=(",", ":"), default=str) + "\n").encode(
                "utf-8"
            )
            gz.write(line)
    body = buf.getvalue()

    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
        )
    except ClientError as e:
        raise RuntimeError(f"S3 JSONL upload failed: {e}")


def write_csv_to_s3_polars(
    df: pl.DataFrame,
    bucket: str,
    key: str,
    compress: bool = False,
    include_header: bool = True,
    s3_client=None,
    **to_csv_kwargs,
):
    """
    Write polars DataFrame to s3://bucket/key as CSV (optionally .gz).
    to_csv_kwargs → passed to Polars write_csv (e.g., separator=',', quote='\"').
    """
    if df.height == 0:
        return
    if s3_client is None:
        s3_client = boto3.Session(profile_name="etl_user").client("s3")

    # Polars write_csv expects a text buffer; wrap a BytesIO.
    raw_buf = io.BytesIO()
    text_buf = io.TextIOWrapper(raw_buf, encoding="utf-8", newline="")
    df.write_csv(text_buf, include_header=include_header, **to_csv_kwargs)
    text_buf.flush()
    raw_bytes = raw_buf.getvalue()

    extra = {"ContentType": "text/csv; charset=utf-8"}
    if compress:
        gz = io.BytesIO()
        with gzip.GzipFile(fileobj=gz, mode="wb") as z:
            z.write(raw_bytes)
        body = gz.getvalue()
        extra["ContentEncoding"] = "gzip"
    else:
        body = raw_bytes

    try:
        s3_client.put_object(Bucket=bucket, Key=key, Body=body, **extra)
    except ClientError as e:
        raise RuntimeError(f"S3 CSV upload failed: {e}")


# -------------------------
# Postgres Loader (TRUNCATE + INSERT)
# -------------------------
def load_polars_df_to_postgres(
    df: pl.DataFrame,
    table: str,
    conn_str: str,
    schema: str = "public",
):
    """
    Replaces target table: TRUNCATE then bulk INSERT via Polars write_to_database.
    Creates table if absent (all columns TEXT for simplicity).
    """
    if df.height == 0:
        LOG.info("No rows; skipping Postgres replace.")
        return

    conn = psycopg2.connect(conn_str)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            # Create schema
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema};")
            # DROP target
            cur.execute(f'DROP TABLE IF EXISTS "{schema}"."{table}";')
            
            # Create table with proper schema inference
            # Let Polars handle the table creation with proper data types
            df.write_database(
                table_name=f"{schema}.{table}",
                connection=conn,
                if_table_exists="replace"
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        raise
    finally:
        conn.close()


# -------------------------
# Process data
# -------------------------
def process(env):
    """Extract data from Dash API and load to S3 and Postgres."""
    # Setup logging
    log = setup_logging()

    # Get bearer token
    bearer_token = get_bearer_token()

    # Configure environment-specific settings
    s3_bucket = f"{S3_BUCKET}{env}"
    database_name = f"{POSTGRES_DB}{env}"
    pg_conn = f"dbname={database_name} user={POSTGRES_USER} password={POSTGRES_PASSWORD} host=localhost port=5432"
    keep_raw_jsonl = False
    csv_compress = True

    log.info(f"Starting extraction for environment: {env}")
    log.info(f"Using S3 bucket: {s3_bucket}")

    # Fetch data from API
    DASH_ENTITIES = [
        # "event-types",
        "events",
        # "customers",
        # "bookings",
        # "resources",
        # "stat-events",
        # "customer-events",
        # "addresses",
        # "invoices",
        # "event-comments",
        # "event-employees",
        # "customer-relationships",
    ]

    for entity in DASH_ENTITIES:
        try:
            log.info(f"Processing entity: {entity}")
            s3_target = f"""all_{entity.replace("-","_")}"""
            db_schema = "sch_raw"
            db_target = entity.replace("-","_")
            records = get_index_data(entity, bearer_token)
            log.info(f"Fetched {len(records)} records from /v1/{entity}")

            # 2) Optional: write raw JSONL.gz to S3
            run_date = datetime.now().strftime("%Y-%m-%d")
            s3_client = boto3.Session(profile_name="etl_user").client("s3")

            if keep_raw_jsonl and records:
                raw_key = f"raw/{s3_target}/ingest_date={run_date}/{s3_target}.jsonl.gz"
                log.info(f"Writing raw JSONL to s3://{s3_bucket}/{raw_key}")
                write_jsonl_to_s3(records, s3_bucket, raw_key, s3_client)

            # 3) Normalize to Polars
            df = records_to_polars(records)
            log.info(f"Normalized to Polars: {df.height} rows, {len(df.columns)} columns")

            # 4) Write CSV to S3
            csv_key = f"catalog/{s3_target}/ingest_date={run_date}/{s3_target}.csv"
            if csv_compress:
                csv_key += ".gz"

            log.info(f"Writing CSV to s3://{s3_bucket}/{csv_key}")
            write_csv_to_s3_polars(
                df, s3_bucket, csv_key, compress=csv_compress, s3_client=s3_client
            )

            # 5) Replace table in Postgres
            load_polars_df_to_postgres(
                df, schema=db_schema, table=db_target, conn_str=pg_conn
            )
            log.info(f"Replaced Postgres table {db_target}")

            break
        except Exception as e:
            log.error(f"Error processing entity: {entity}")
            raise e

@click.command()
@click.option(
    "--env",
    default="dev",
    help="Environment (dev, prd)",
    type=click.Choice(["dev", "prd"]),
)
def main(env):
    process(env)


if __name__ == "__main__":
    main()
