import re

import duckdb
import pandas as pd

TABLE_NAME = "data"


def clean_column_names(columns) -> list[str]:
    """Make column names SQL-friendly: 'Order ID' -> 'order_id'. Keeps names unique."""
    cleaned, seen = [], {}
    for col in columns:
        name = re.sub(r"[^0-9a-zA-Z]+", "_", str(col).strip().lower()).strip("_")
        if not name:
            name = "column"
        if name[0].isdigit():
            name = f"col_{name}"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        cleaned.append(name)
    return cleaned


def read_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX (Streamlit UploadedFile) into a DataFrame."""
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)  # first sheet only in V1
    else:
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    if df.empty:
        raise ValueError("The file has no rows.")

    df.columns = clean_column_names(df.columns)
    return df


def load_into_duckdb(df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB database with the data in a table called `data`."""
    con = duckdb.connect(":memory:")
    con.register("uploaded_df", df)
    con.execute(f"CREATE TABLE {TABLE_NAME} AS SELECT * FROM uploaded_df")
    con.unregister("uploaded_df")
    return con


def get_schema(con) -> tuple[pd.DataFrame, str]:
    """Return (schema table for the UI, schema text for the LLM prompt)."""
    cur = con.cursor()
    info = cur.execute(f"DESCRIBE {TABLE_NAME}").df()[["column_name", "column_type"]]
    sample = cur.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 3").df()

    lines = [f"Table: {TABLE_NAME}", "Columns:"]
    lines += [f"- {r.column_name} ({r.column_type})" for r in info.itertuples()]
    lines += ["", "Sample rows:", sample.to_string(index=False)]
    return info, "\n".join(lines)


"""
Module: Data Loader (data_loader.py)
Description:
This module manages the complete data ingestion, preprocessing, and schema 
extraction pipeline for the AI Data Analysis Agent[cite: 1]. 

Key functionalities include:
1. Column Name Sanitation:
   - Converts raw, messy column headers (e.g., spaces, special characters, 
     mixed casing) into clean, SQL-friendly identifiers[cite: 1].
   - Prevents database syntax errors by handling digit-starting names (prefixed 
     with 'col_') and guarantees uniqueness by appending numeric suffixes 
     to duplicate column titles[cite: 1].

2. File Reading & Validation:
   - Reads uploaded CSV or Excel (.xlsx) files from Streamlit file uploaders 
     into pandas DataFrames while rejecting empty files or unsupported formats[cite: 1].

3. In-Memory Database Ingestion:
   - Spins up an ephemeral in-memory DuckDB database connection and materializes 
     the processed DataFrame into a structured SQL table named 'data'[cite: 1].

4. Schema & Context Generation:
   - Inspects the DuckDB table schema to generate a metadata summary for the 
     Streamlit UI sidebar and a formatted text block containing column types 
     and sample rows[cite: 1]. 
   - Providing these sample rows ensures the LLM can reference exact value 
     spellings and capitalization, drastically improving SQL generation accuracy[cite: 1].
"""