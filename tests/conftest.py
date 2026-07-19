import os

import psycopg
import pytest

from knowbase import db as db_mod

TEST_SERVER_DSN = os.environ.get(
    "KB_TEST_SERVER_DSN", "postgresql://knowbase:knowbase@localhost:5433/postgres"
)
TEST_DSN = os.environ.get(
    "KB_TEST_DSN", "postgresql://knowbase:knowbase@localhost:5433/knowbase_test"
)
TEST_DIMS = 384


@pytest.fixture(scope="session")
def db():
    admin = psycopg.connect(TEST_SERVER_DSN, autocommit=True)
    admin.execute("DROP DATABASE IF EXISTS knowbase_test")
    admin.execute("CREATE DATABASE knowbase_test")
    admin.close()
    conn = db_mod.connect(TEST_DSN)
    db_mod.init_db(conn, dims=TEST_DIMS)
    yield conn
    conn.close()


@pytest.fixture()
def clean_db(db):
    db.execute("TRUNCATE embeddings, sync_state")
    return db
