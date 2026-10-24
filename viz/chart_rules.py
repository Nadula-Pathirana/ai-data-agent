"""Pure-pandas chart logic. No LLM and no Streamlit in this file, so it is easy to test.

It provides:
  - requested_chart_type : which chart type did the user explicitly name? ("pie", "line", ...)
  - normalize_result     : tidy up types in a query result (Decimal -> float, dates -> datetime, 'year' text -> int)
  - column_kinds         : label every column "numeric", "datetime" or "categorical"
  - histogram_bins       : bin raw numbers into (bin_start, bin_end, count)
  - suggest_chart        : the rule-based fallback that picks a chart without any LLM
"""
import re
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

ALLOWED_CHARTS = ("bar", "line", "scatter", "histogram", "pie", "none")
MIN_HISTOGRAM_ROWS = 10  # the rule-based chooser only proposes a histogram for 10+ values

# Column-name words that suggest "this column is an ordered axis" (used for numeric x columns)
_ORDERED_WORDS = {"year", "month", "quarter", "week", "day", "hour", "minute", "period", "date", "time"}

# '2025-03', '2025-03-15', '2025-03-15 10:00:00'  (strftime('%Y-%m') results look like the first one)
_ISO_DATE = re.compile(r"^\d{4}-\d{2}(-\d{2})?([ T].*)?$")

# FIX 1: '1903' (what strftime(..., '%Y') returns) is NOT an ISO date, so it used to stay text.
_YEAR_TEXT = re.compile(r"^\d{4}$")
_YEAR_COLUMN_NAME = re.compile(r"(^|_)(year|years|yr)($|_)")

# ---------- which chart did the user ask for? ----------

# Order matters: the first pattern that matches wins.
_EXPLICIT_TYPE_PATTERNS = (
    ("pie", re.compile(
        r"\b(pie|donut|doughnut)[\s-]*(chart|graph|plot)s?\b|\b(as|in|into|with)\s+an?\s+(pie|donut|doughnut)\b",
        re.IGNORECASE)),
    ("histogram", re.compile(r"\bhistograms?\b", re.IGNORECASE)),
    ("scatter", re.compile(r"\bscatter\w*", re.IGNORECASE)),
    ("line", re.compile(r"\bline[\s-]*(chart|graph|plot)s?\b", re.IGNORECASE)),
    ("bar", re.compile(r"\b(bar|column)[\s-]*(chart|graph|plot)s?\b", re.IGNORECASE)),
)


def requested_chart_type(question: str) -> str | None:
    """Return 'pie' / 'histogram' / 'scatter' / 'line' / 'bar' if the user NAMED a chart type, else None."""
    for chart, pattern in _EXPLICIT_TYPE_PATTERNS:
        if pattern.search(question or ""):
            return chart
    return None


# ---------- tidy types ----------

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
      - '1903' style text in a column named *year* -> converted to a number (FIX 1)
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
        elif (_YEAR_COLUMN_NAME.search(str(col).lower())
              and all(isinstance(v, str) and _YEAR_TEXT.match(v.strip()) for v in values)):
            out[col] = pd.to_numeric(series.str.strip(), errors="coerce")
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


def is_year_like(series: pd.Series) -> bool:
    """True for a numeric column that only holds whole years (1000-2999). Used to avoid '1,903' on axes."""
    if not pd.api.types.is_numeric_dtype(series):
        return False
    s = series.dropna()
    return bool(len(s)) and bool((s == s.round()).all()) and bool(s.between(1000, 2999).all())


def histogram_bins(values, max_bins: int = 20) -> pd.DataFrame:
    """Bin raw numbers. Returns columns bin_start, bin_end, count. (FIX 3: the app bins, not the SQL.)"""
    vals = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return pd.DataFrame({"bin_start": [], "bin_end": [], "count": []})
    bins = int(min(max_bins, max(5, np.ceil(np.sqrt(len(vals))))))
    counts, edges = np.histogram(vals.to_numpy(dtype=float), bins=bins)
    return pd.DataFrame({"bin_start": edges[:-1], "bin_end": edges[1:], "count": counts})


# ---------- the rule-based chooser ----------

def _spec(chart: str, x: str, y: str | None = None) -> dict:
    title = f"Distribution of {x}" if chart == "histogram" else f"{y} by {x}"
    return {"chart": chart, "x": x, "y": y, "title": title}


def _soft_hint(question: str) -> str | None:
    """Looser hints for when the user did NOT name a chart type ('trend', 'vs', 'distribution')."""
    q = question.lower()
    if re.search(r"distribution", q):
        return "histogram"
    if re.search(r"relationship|correlat|\bvs\.?\b|versus|against", q):
        return "scatter"
    if re.search(r"\bline\b|trend|over time", q):
        return "line"
    if re.search(r"\bbar\b", q):
        return "bar"
    return None


def _is_ordered_name(name: str) -> bool:
    return bool(set(name.lower().split("_")) & _ORDERED_WORDS)


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
        # A category axis. FIX 2: repeated labels (two 'Titanic' rows) no longer block a bar chart;
        # the renderer gives repeated labels a '(2)' suffix instead.
        elif len(cats) == 1 and not dates and numeric:
            possible["bar"] = _spec("bar", cats[0], numeric[0])
            values = data[numeric[0]].dropna()
            # NEW: pie chart = one label + one non-negative number with a positive total.
            if len(values) >= 2 and (values >= 0).all() and values.sum() > 0:
                possible["pie"] = _spec("pie", cats[0], numeric[0])
        # Only numbers: a line/bar if the first column is an ordered axis (year, month...), else scatter.
        if not cats and not dates and len(numeric) >= 2:
            first, second = numeric[0], numeric[1]
            if _is_ordered_name(first) and unique(first):
                possible["line"] = _spec("line", first, second)
                possible["bar"] = _spec("bar", first, second)
            possible["scatter"] = _spec("scatter", first, second)
        elif len(numeric) >= 2 and (cats or dates):
            possible.setdefault("scatter", _spec("scatter", numeric[0], numeric[1]))

    # FIX 3: a histogram needs RAW numbers: exactly one numeric column and nothing else.
    # (Before, a table of "score_range, movie_count" counted as a histogram candidate of the counts.)
    if len(numeric) == 1 and not cats and not dates and rows >= MIN_HISTOGRAM_ROWS:
        possible["histogram"] = _spec("histogram", numeric[0])
    return possible


def suggest_chart(df: pd.DataFrame, question: str = "") -> dict:
    """Pick a chart using simple rules. Returns a spec dict; {'chart': 'none'} if nothing fits.

    The chart type the user NAMED wins whenever the data can support it.
    """
    data = normalize_result(df)
    possible = _possible_charts(data)
    if not possible:
        return {"chart": "none", "x": None, "y": None, "title": ""}

    preferred = requested_chart_type(question) or _soft_hint(question)
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