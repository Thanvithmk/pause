import csv
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / ".env")
load_dotenv(_ROOT / "backend" / ".env")
load_dotenv()

DATABASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
DEFAULT_SLANG_CSV = Path(os.getenv("SLANG_CSV_PATH", DATABASE_DIR / "slang_dictionary.csv"))


def get_connection():
    url = os.getenv("DATABASE_URL")
    if url:
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return psycopg2.connect(url)

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "pptis"),
        user=os.getenv("POSTGRES_USER", "pptis"),
        password=os.getenv("POSTGRES_PASSWORD", "pptis"),
    )


def create_tables(connection):
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(schema_sql)
    connection.commit()


def load_slang_dictionary(connection, csv_path=DEFAULT_SLANG_CSV):
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Slang dictionary CSV not found: {csv_path}")

    insert_sql = """
        INSERT INTO slang_dictionary (slang_term, meaning, category, confidence)
        VALUES (%s, %s, %s, %s)
    """

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            (
                row["slang_term"],
                row["meaning"],
                row.get("category") or None,
                float(row["confidence"]) if row.get("confidence") else None,
            )
            for row in reader
        ]

    with connection.cursor() as cursor:
        cursor.executemany(insert_sql, rows)
    connection.commit()
    return len(rows)


def init_db(slang_csv_path=DEFAULT_SLANG_CSV):
    connection = get_connection()
    try:
        create_tables(connection)
        loaded = load_slang_dictionary(connection, slang_csv_path)
        print(f"Tables created. Loaded {loaded} slang entries from {slang_csv_path}.")
    finally:
        connection.close()


if __name__ == "__main__":
    init_db()
