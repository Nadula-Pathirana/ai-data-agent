import pandas as pd

from data_loader import TABLE_NAME

NUMERIC_TYPES = {
    "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT",
    "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT",
    "FLOAT", "DOUBLE", "DECIMAL",
}
SMALL_CARDINALITY = 10   # text columns with this many distinct values or fewer: list them all
TOP_VALUES = 5           # otherwise list this many of the most common values


def _quote(name: str) -> str:
    """Quote a column name so odd names cannot break the SQL."""
    return '"' + name.replace('"', '""') + '"'


def _base_type(column_type: str) -> str:
    return column_type.split("(")[0].strip().upper()  # DECIMAL(10,2) -> DECIMAL


def _short(value, limit: int = 40) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def profile_column(cur, name: str, column_type: str, total_rows: int) -> dict:
    """Compute simple statistics for one column. All numbers come from DuckDB."""
    col = _quote(name)
    base = _base_type(column_type)
    info = {"column": name, "type": column_type, "nulls": None, "distinct": None,
            "min": None, "max": None, "mean": None, "top_values": []}
    try:
        non_null, distinct, min_v, max_v = cur.execute(
            f"SELECT COUNT({col}), COUNT(DISTINCT {col}), MIN({col}), MAX({col}) FROM {TABLE_NAME}"
        ).fetchone()
        info.update(nulls=total_rows - non_null, distinct=distinct)

        if base == "VARCHAR":
            if non_null:
                limit = distinct if distinct <= SMALL_CARDINALITY else TOP_VALUES
                rows = cur.execute(
                    f"SELECT {col}, COUNT(*) AS n FROM {TABLE_NAME} "
                    f"WHERE {col} IS NOT NULL GROUP BY {col} ORDER BY n DESC, {col} LIMIT {limit}"
                ).fetchall()
                info["top_values"] = [_short(r[0]) for r in rows]
        else:
            info.update(min=_short(min_v), max=_short(max_v))
            if base in NUMERIC_TYPES:
                mean = cur.execute(f"SELECT AVG({col}) FROM {TABLE_NAME}").fetchone()[0]
                info["mean"] = None if mean is None else round(float(mean), 2)
    except Exception:
        pass  # a column we cannot profile should never stop the app
    return info


def build_profile(con) -> tuple[pd.DataFrame, str]:
    """Return (profile table for the UI, compact profile text for the LLM prompt)."""
    cur = con.cursor()
    total_rows = cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    schema = cur.execute(f"DESCRIBE {TABLE_NAME}").fetchall()  # (name, type, ...)

    infos = [profile_column(cur, row[0], row[1], total_rows) for row in schema]

    # ---- table for the UI (everything as text so Streamlit never complains about mixed types) ----
    table = pd.DataFrame([
        {
            "column": i["column"],
            "type": i["type"],
            "missing": "" if i["nulls"] is None else str(i["nulls"]),
            "missing_%": "" if i["nulls"] is None or not total_rows
                         else f"{100 * i['nulls'] / total_rows:.0f}%",
            "distinct": "" if i["distinct"] is None else str(i["distinct"]),
            "min": "" if i["min"] is None else i["min"],
            "max": "" if i["max"] is None else i["max"],
            "mean": "" if i["mean"] is None else str(i["mean"]),
            "common_values": ", ".join(i["top_values"]),
        }
        for i in infos
    ])

    # ---- text for the prompt ----
    lines = [f"Dataset profile ({total_rows} rows):"]
    for i in infos:
        parts = []
        if i["distinct"] is not None:
            parts.append(f"{i['distinct']} distinct")
        if i["nulls"]:
            parts.append(f"{i['nulls']} missing (NULL)")
        if i["min"] is not None:
            parts.append(f"min {i['min']}, max {i['max']}")
        if i["mean"] is not None:
            parts.append(f"mean {i['mean']}")
        if i["top_values"]:
            label = "values" if (i["distinct"] or 0) <= SMALL_CARDINALITY else "most common values"
            parts.append(f"{label}: " + ", ".join(f"'{v}'" for v in i["top_values"]))
        lines.append(f"- {i['column']} ({i['type']}): " + "; ".join(parts))

    return table, "\n".join(lines)