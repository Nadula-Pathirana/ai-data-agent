import re

import duckdb
import pandas as pd

import config

TABLE_NAME = "data"
SUPPORTED_TYPES = (".csv", ".xlsx")


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


def _read_csv(uploaded_file) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)  # rewind, then retry with a more forgiving encoding
        return pd.read_csv(uploaded_file, encoding="latin-1")


def read_file(uploaded_file) -> pd.DataFrame:
    """Read an uploaded CSV/XLSX (Streamlit UploadedFile) into a DataFrame.

    Every problem is raised as a ValueError with a message that is safe to show to the user.
    """
    filename = uploaded_file.name.lower()
    if not filename.endswith(SUPPORTED_TYPES):
        raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > config.MAX_FILE_MB:
        raise ValueError(
            f"The file is {size_mb:.1f} MB, which is over the {config.MAX_FILE_MB} MB limit."
        )

    try:
        if filename.endswith(".csv"):
            df = _read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)  # first sheet only until V5
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.") from None
    except Exception as exc:
        kind = "CSV" if filename.endswith(".csv") else "Excel"
        raise ValueError(
            f"Could not read this {kind} file. It may be corrupt, password protected "
            f"or not a valid {kind} file. ({type(exc).__name__})"
        ) from exc

    # Messy spreadsheets often contain blank rows and columns with a header but no values.
    # df = df.dropna(how="all").dropna(axis="columns", how="all")
    # if df.empty:
    #     raise ValueError("The file has no data rows.")

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