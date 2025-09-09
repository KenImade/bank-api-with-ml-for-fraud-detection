#!/bin/bash
set -o errexit
set -o nounset
set -o pipefail

python <<'PY'
import sys, time, os, psycopg

MAX_WAIT_SECONDS = 60
RETRY_INTERVAL = 5
start_time = time.time()

def check_database():
    try:
        psycopg.connect(
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
            host=os.environ["POSTGRES_HOST"],
            port=os.environ.get("POSTGRES_PORT", "5432"),
            connect_timeout=5,
        ).close()
        return True
    except psycopg.OperationalError as error:
        elapsed = int(time.time() - start_time)
        print(f"Database connection attempt failed after {elapsed} seconds: {error}", file=sys.stderr)
        return False

while True:
    if check_database():
        break
    if time.time() - start_time > MAX_WAIT_SECONDS:
        print(f"Error: Database connection could not be established after {MAX_WAIT_SECONDS} seconds", file=sys.stderr)
        sys.exit(1)
    print(f"Waiting {RETRY_INTERVAL} seconds before retrying...", file=sys.stderr)
    time.sleep(RETRY_INTERVAL)
PY

echo >&2 'PostgreSQL is ready to accept connections'
echo "Running: alembic upgrade head"
alembic upgrade head
echo "Migrations completed successfully"
echo >&2 'Migrations applied'

exec "$@"
