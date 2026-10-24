"""Asks the LLM for a chart spec, then treats the answer as UNTRUSTED input.

The LLM only *proposes*. validate_spec() decides whether the proposal is usable.
"""
import json
import re

import pandas as pd

from llm import get_llm
from prompts import CHART_PROMPT
from viz.chart_rules import ALLOWED_CHARTS, column_kinds, normalize_result

MIN_ROWS = 2            # bar / line / scatter need at least this many rows
MIN_HISTOGRAM_ROWS = 5  # a histogram of fewer values is meaningless

_EXPLICIT_CHART_WORDS = re.compile(
    r"\b(plot|chart|graph|visuali[sz]\w*|histogram|scatter\w*)\b", re.IGNORECASE
)


def wants_chart(question: str) -> bool:
    """True when the user explicitly asked for a visual ('plot sales by month')."""
    return bool(_EXPLICIT_CHART_WORDS.search(question))


def _response_text(response) -> str:
    """Same idea as _as_text in agent_graph.py (kept here to avoid a circular import)."""
    content = response.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def parse_spec_json(text: str) -> dict:
    """Pull the first {...} object out of the model's reply. Raises ValueError if there is none."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)  # also copes with ```json fences and chatter
    if not match:
        raise ValueError("The model did not return a JSON object.")
    try:
        spec = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"The model returned invalid JSON ({exc.msg}).") from exc
    if not isinstance(spec, dict):
        raise ValueError("The chart spec must be a JSON object.")
    return spec


def _match_column(name, columns) -> str | None:
    """Find a column by exact name, then ignoring case and surrounding spaces."""
    if not isinstance(name, str):
        return None
    if name in columns:
        return name
    lowered = {c.lower(): c for c in columns}
    return lowered.get(name.strip().lower())


def validate_spec(spec: dict, df: pd.DataFrame) -> dict:
    """Check a chart spec against the real result. Returns a clean spec or raises ValueError.

    Nothing the model says is trusted: the chart type must be allowed, the columns must
    exist, the column types must suit the chart, and the data must be drawable.
    """
    if not isinstance(spec, dict):
        raise ValueError("The chart spec must be a JSON object.")

    data = normalize_result(df)
    kinds = column_kinds(data)
    columns = list(data.columns)

    chart = str(spec.get("chart")).strip().lower()  # a JSON null becomes "none"
    if chart not in ALLOWED_CHARTS:
        raise ValueError(f"'{chart}' is not an allowed chart type ({', '.join(ALLOWED_CHARTS)}).")
    if chart == "none":
        return {"chart": "none", "x": None, "y": None, "title": ""}

    x = _match_column(spec.get("x"), columns)
    if x is None:
        raise ValueError(f"Column '{spec.get('x')}' is not in the result. Columns: {', '.join(columns)}.")

    y = None
    if chart == "histogram":
        if kinds[x] != "numeric":
            raise ValueError(f"A histogram needs a numeric column, but '{x}' is {kinds[x]}.")
        if len(data) < MIN_HISTOGRAM_ROWS:
            raise ValueError("Too few rows for a histogram.")
    else:
        y = _match_column(spec.get("y"), columns)
        if y is None:
            raise ValueError(f"Column '{spec.get('y')}' is not in the result. Columns: {', '.join(columns)}.")
        if x == y:
            raise ValueError("x and y must be different columns.")
        if len(data) < MIN_ROWS:
            raise ValueError("Too few rows to draw this chart.")

        if chart == "bar":
            if kinds[y] != "numeric":
                raise ValueError(f"Bar heights must be numeric, but '{y}' is {kinds[y]}.")
        elif chart == "line":
            if kinds[y] != "numeric":
                raise ValueError(f"A line needs a numeric y column, but '{y}' is {kinds[y]}.")
            if kinds[x] == "categorical":
                raise ValueError(f"A line chart needs a date or number on the x axis, but '{x}' is categorical.")
        elif chart == "scatter":
            if kinds[x] != "numeric" or kinds[y] != "numeric":
                raise ValueError("A scatter plot needs two numeric columns.")

        if chart in ("bar", "line") and data[x].duplicated().any():
            raise ValueError(f"'{x}' has repeated values, so a {chart} chart would draw several points per x.")

    title = str(spec.get("title") or "").strip()[:80]
    if not title:
        title = f"Distribution of {x}" if chart == "histogram" else f"{y} by {x}"
    return {"chart": chart, "x": x, "y": y, "title": title}


def plan_chart_with_llm(question: str, df: pd.DataFrame, requested: bool) -> dict:
    """Ask the model for a chart spec. Returns the RAW parsed JSON (not yet validated)."""
    data = normalize_result(df)
    kinds = column_kinds(data)
    chain = CHART_PROMPT | get_llm()
    response = chain.invoke({
        "question": question,
        "requested": ("YES. The user explicitly asked for a chart." if requested
                      else "No. Decide whether a chart would really help."),
        "columns": "\n".join(f"- {c} ({kinds[c]})" for c in data.columns),
        "row_count": len(data),
        "sample": data.head(3).to_string(index=False),
    })
    return parse_spec_json(_response_text(response))