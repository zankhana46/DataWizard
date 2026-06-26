import sqlite3
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from functools import lru_cache
from langchain_core.tools import tool
import store

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "sample.db")


def _get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@lru_cache(maxsize=1)
def _fetch_schema() -> str:
    """Cached once per process — the schema never changes at runtime."""
    try:
        import pandas as _pd
        con = _get_connection()
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cur.fetchall()]
        if not tables:
            con.close()
            return "The database is empty — no tables found."

        parts = []
        for table in tables:
            cur.execute(f"PRAGMA table_info({table});")
            cols = cur.fetchall()
            col_defs = ", ".join(f"{c[1]} {c[2]}" for c in cols)
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            count = cur.fetchone()[0]
            parts.append(f"Table: {table} ({count} rows)\n  Columns: {col_defs}")

            # Queue a schema DataFrame so the UI can render it as a table widget
            schema_df = _pd.DataFrame(
                [(c[1], c[2], "YES" if not c[3] else "NO") for c in cols],
                columns=["column", "type", "nullable"]
            )
            store.queue_dataframe(f"Schema: {table} ({count} rows)", schema_df)

        con.close()
        return "\n\n".join(parts)
    except Exception as e:
        return f"Error reading schema: {e}"


@tool
def get_sql_schema() -> str:
    """
    Returns the schema of all tables in the SQLite database (sample.db).
    Always call this before writing any SQL query so you know the available
    tables and their column names and types.
    """
    # Re-fetch each call so schema DataFrames are always queued for display
    _fetch_schema.cache_clear()
    return _fetch_schema()


@tool
def run_sql_query(sql_query: str) -> str:
    """
    Executes a SQLite SELECT query against sample.db and returns the results.
    You must call get_sql_schema first to know the table and column names.
    Write valid SQLite SQL — only SELECT statements are allowed.

    Args:
        sql_query: A complete, valid SQLite SELECT statement.
    """
    sql = sql_query.strip().rstrip(";") + ";"

    # Strip accidental markdown fences
    if sql.startswith("```"):
        lines = sql.splitlines()
        sql = "\n".join(l for l in lines if not l.startswith("```")).strip()
        if not sql.endswith(";"):
            sql += ";"

    try:
        con = _get_connection()
        cur = con.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        col_names = [desc[0] for desc in cur.description] if cur.description else []
        con.close()

        store.queue_code("run_sql_query — SQL executed", f"""\
import sqlite3, pandas as pd

con = sqlite3.connect("data/sample.db")
df = pd.read_sql_query(\"\"\"
{sql}
\"\"\", con)
con.close()
print(df)
""")

        if not rows:
            return f"Query returned no results.\n\nSQL used:\n{sql}"

        import pandas as _pd
        result_df = _pd.DataFrame(rows[:50], columns=col_names)
        store.queue_dataframe(f"Query result ({len(rows)} rows)", result_df)

        suffix = f" (showing first 50 of {len(rows)})" if len(rows) > 50 else ""
        return f"Query returned {len(rows)} row(s){suffix}. Results are displayed as a table. SQL used:\n{sql}"
    except Exception as e:
        return (
            f"SQL execution failed: {e}\n\n"
            f"SQL attempted:\n{sql}\n\n"
            "Check the schema with get_sql_schema and try a corrected query."
        )
