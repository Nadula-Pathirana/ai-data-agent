"""Pure-pandas chart logic. No LLM and no Streamlit in this file, so it is easy to test.

It provides:
  - normalize_result : tidy up types in a query result (Decimal -> float, dates -> datetime)
  - column_kinds     : label every column "numeric", "datetime" or "categorical"
  - suggest_chart    : the rule-based fallback that picks a chart without any LLM
"""
import re
from datetime import date, datetime
from decimal import Decimal

import pandas as pd

ALLOWED_CHARTS = ("bar", "line", "scatter", "histogram", "none")
MIN_HISTOGRAM_ROWS = 10  # the rule-based chooser only proposes a histogram for 10+ values

# Column-name words that suggest "this column is an ordered axis" (used for numeric x columns)
_ORDERED_WORDS = {"year", "month", "quarter", "week", "day", "hour", "minute", "period", "date", "time"}

# '2025-03', '2025-03-15', '2025-03-15 10:00:00'  (strftime('%Y-%m') results look like the first one)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T].*)?$")


def _unique_names(columns) -> list[str]:
    """SELECT a.x, b.x gives two columns named x. Make the names unique."""
    seen: dict[str, int] = {}
    names = []
    for col in columns:
        name = str(col)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        names.append(name)
    return names


def normalize_result(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with chart-friendly types. Safe to call more than once.

    DuckDB hands Python plain objects, so a DataFrame built from query rows can hold:
      - decimal.Decimal values (object column)    -> converted to float
      - datetime.date values (object column)      -> converted to datetime64
      - ISO date strings such as '2025-03'        -> converted to datetime64
    """
    out = df.copy()
    out.columns = _unique_names(out.columns)

    for col in out.columns:
        series = out[col]
        if (pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series)
                or pd.api.types.is_datetime64_any_dtype(series)):
            continue
        values = series.dropna()
        if values.empty:
            continue
        if all(isinstance(v, Decimal) for v in values):
            out[col] = pd.to_numeric(series, errors="coerce")
        elif all(isinstance(v, (date, datetime)) for v in values):
            out[col] = pd.to_datetime(series)
        elif all(isinstance(v, str) and _ISO_DATE.match(v) for v in values):
            out[col] = pd.to_datetime(series, format="ISO8601", errors="coerce")
    return out


def column_kinds(df: pd.DataFrame) -> dict[str, str]:
    """Map each column name to 'numeric', 'datetime' or 'categorical'."""
    kinds = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            kinds[col] = "categorical"
        elif pd.api.types.is_numeric_dtype(series):
            kinds[col] = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(series):
            kinds[col] = "datetime"
        else:
            kinds[col] = "categorical"
    return kinds


# ---------- the rule-based chooser ----------

def _spec(chart: str, x: str, y: str | None = None) -> dict:
    title = f"Distribution of {x}" if chart == "histogram" else f"{y} by {x}"
    return {"chart": chart, "x": x, "y": y, "title": title}


def _preferred_type(question: str) -> str | None:
    """If the user named a chart type or relationship, try that type first."""
    q = question.lower()
    if re.search(r"histogram|distribution", q):
        return "histogram"
    if re.search(r"scatter|relationship|correlat|\bvs\.?\b|versus|against", q):
        return "scatter"
    if re.search(r"\bline\b|trend|over time", q):
        return "line"
    if re.search(r"\bbar\b", q):
        return "bar"
    return None


def _possible_charts(data: pd.DataFrame) -> dict[str, dict]:
    """Every chart type these columns can sensibly support, as ready-made specs."""
    kinds = column_kinds(data)
    numeric = [c for c in data.columns if kinds[c] == "numeric"]
    dates = [c for c in data.columns if kinds[c] == "datetime"]
    cats = [c for c in data.columns if kinds[c] == "categorical"]
    rows = len(data)
    possible: dict[str, dict] = {}

    def unique(col: str) -> bool:
        return not data[col].duplicated().any()

    if rows >= 2:
        # A date axis: line (or bars), as long as there is one point per date.
        if len(dates) == 1 and not cats and numeric and unique(dates[0]):
            possible["line"] = _spec("line", dates[0], numeric[0])
            possible["bar"] = _spec("bar", dates[0], numeric[0])
        # A category axis: bars, as long as there is one row per category.
        elif len(cats) == 1 and not dates and numeric and unique(cats[0]):
            possible["bar"] = _spec("bar", cats[0], numeric[0])
        # Only numbers: a line if the first column is an ordered axis (year, month...), else scatter.
        if not cats and not dates and len(numeric) >= 2:
            first, second = numeric[0], numeric[1]
            if set(first.lower().split("_")) & _ORDERED_WORDS and unique(first):
                possible["line"] = _spec("line", first, second)
            possible["scatter"] = _spec("scatter", first, second)
        elif len(numeric) >= 2 and (cats or dates):
            possible.setdefault("scatter", _spec("scatter", numeric[0], numeric[1]))
    if numeric and rows >= MIN_HISTOGRAM_ROWS:
        possible["histogram"] = _spec("histogram", numeric[0])
    return possible


def suggest_chart(df: pd.DataFrame, question: str = "") -> dict:
    """Pick a chart using simple rules. Returns a spec dict; {'chart': 'none'} if nothing fits."""
    data = normalize_result(df)
    possible = _possible_charts(data)
    if not possible:
        return {"chart": "none", "x": None, "y": None, "title": ""}

    preferred = _preferred_type(question)
    if preferred in possible:
        return possible[preferred]

    # Default order. A histogram is the default only for a single numeric column.
    if data.shape[1] == 1:
        order = ["histogram"]
    else:
        order = ["line", "bar", "scatter"]
    for chart in order:
        if chart in possible:
            return possible[chart]
    return {"chart": "none", "x": None, "y": None, "title": ""}