#!/usr/bin/env python3
"""Create the PPTIS++ database, load slang CSV, seed test data, and run tests."""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env")
DATABASE_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = DATABASE_DIR / "schema.sql"
SLANG_CSV = Path(os.getenv("SLANG_CSV_PATH", DATABASE_DIR / "slang_dictionary.csv"))

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))


def create_postgres():
    from init_db import init_db as init_postgres

    init_postgres(SLANG_CSV)


def create_sqlite(sqlite_path):
    from modules.database import SlangDictionary, close_session, get_session, init_db

    uri = f"sqlite:///{sqlite_path}"
    init_db(uri, create_tables=True)
    session = get_session()
    loaded = 0
    if SLANG_CSV.is_file():
        with SLANG_CSV.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                SlangDictionary.create(
                    session,
                    slang_term=row["slang_term"],
                    meaning=row["meaning"],
                    category=row.get("category") or None,
                    confidence=float(row["confidence"]) if row.get("confidence") else None,
                )
                loaded += 1
    close_session()
    print(f"SQLite schema ready at {sqlite_path}. Loaded {loaded} slang entries.")
    return uri


def seed_test_data():
    from modules.database import User, close_session, get_session
    from tests.fixtures import TEST_MESSAGES, seed_conversation, seed_message, seed_user

    session = get_session()
    already_present = (
        session.query(User).filter(User.username == "alexsafe").one_or_none() is not None
    )
    user = seed_user(session)
    conversation = seed_conversation(session, user)
    if already_present:
        print(
            f"Test user {user.username} already exists; leaving existing rows unchanged."
        )
        close_session()
        return
    for spec in TEST_MESSAGES:
        seed_message(session, conversation, user, spec)
    close_session()
    print(
        f"Seeded test user={user.username} conversation_id={conversation.conversation_id}"
    )


def run_tests():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(BACKEND)
    command = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        str(BACKEND / "tests"),
        "-p",
        "test_*.py",
        "-v",
    ]
    print("Running tests:", " ".join(command))
    return subprocess.call(command, cwd=str(BACKEND), env=env)


def main():
    parser = argparse.ArgumentParser(description="PPTIS++ database migration and tests")
    parser.add_argument("--sqlite", default=str(DATABASE_DIR / "pptis.db"))
    parser.add_argument("--postgres", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-seed", action="store_true")
    args = parser.parse_args()

    if args.postgres or os.getenv("DATABASE_URL", "").startswith("postgres"):
        print("Applying PostgreSQL schema.sql and slang CSV...")
        create_postgres()
    else:
        print("Creating SQLite database from SQLAlchemy models...")
        os.environ.setdefault("SQLITE_PATH", args.sqlite)
        create_sqlite(args.sqlite)

    if not args.skip_seed and not args.postgres:
        print("Seeding test data...")
        seed_test_data()
    elif not args.skip_seed and args.postgres:
        print("Skipping SQLAlchemy seed on PostgreSQL; slang CSV already loaded.")

    if args.skip_tests:
        return 0
    return run_tests()


if __name__ == "__main__":
    raise SystemExit(main())
