"""Asks the LLM for a chart spec, then treats the answer as UNTRUSTED input.

The LLM only *proposes*. validate_spec() decides whether the proposal is usable.
"""
import json
import re

import pandas as pd

from llm import get_llm
from prompts import CHART_PROMPT
from viz.chart_rules import ALLOWED_CHARTS, column_kinds, normalize_result, requested_chart_type

MIN_ROWS = 2            # bar / line / scatter / pie need at least this many rows
MIN_HISTOGRAM_ROWS = 5  # a histogram of fewer values is meaningless

_EXPLICIT_CHART_WORDS = re.compile(
    r"\b(plot|chart|graph|visuali[sz]\w*|histogram|scatter\w*|pie|donut|doughnut)\b", re.IGNORECASE
)


def wants_chart(question: str) -> bool:
    """True when the user explicitly asked for a visual ('plot sales by month')."""
    return bool(_EXPLICIT_CHART_WORDS.search(question)) or requested_chart_type(question) is not None


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


def validate_spec(spec: dict, df: pd.DataFrame, requested_type: str | None = None) -> dict:
    """Check a chart spec against the real result. Returns a clean spec or raises ValueError.

    Nothing the model says is trusted: the chart type must be allowed, the columns must
    exist, the column types must suit the chart, and the data must be drawable.

    requested_type: the chart type the USER named ("line", "pie"...). If given, it overrides
    whatever type the model chose (FIX 1), and the usual checks then decide if it is drawable.
    """
    if not isinstance(spec, dict):
        raise ValueError("The chart spec must be a JSON object.")

    data = normalize_result(df)
    kinds = column_kinds(data)
    columns = list(data.columns)

    chart = str(spec.get("chart")).strip().lower()  # a JSON null becomes "none"
    if chart not in ALLOWED_CHARTS:
        raise ValueError(f"'{chart}' is not an allowed chart type ({', '.join(ALLOWED_CHARTS)}).")
    if requested_type in ALLOWED_CHARTS and requested_type != "none":
        chart = requested_type  # the user's word beats the model's guess
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
        # FIX 3: a histogram of a 'count' column that came out of a GROUP BY is not a histogram of the data.
        if data.shape[1] > 1 and any(k == "categorical" for k in kinds.values()):
            raise ValueError("The result is already grouped (label + number), so it cannot be a raw histogram.")
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
            # FIX 2: repeated x labels are allowed now (the renderer disambiguates them).
        elif chart == "line":
            if kinds[y] != "numeric":
                raise ValueError(f"A line needs a numeric y column, but '{y}' is {kinds[y]}.")
            if kinds[x] == "categorical":
                raise ValueError(f"A line chart needs a date or number on the x axis, but '{x}' is categorical.")
            if data[x].duplicated().any():
                raise ValueError(f"'{x}' has repeated values, so a line chart would draw several points per x.")
        elif chart == "scatter":
            if kinds[x] != "numeric" or kinds[y] != "numeric":
                raise ValueError("A scatter plot needs two numeric columns.")
        elif chart == "pie":
            if kinds[x] != "categorical":
                raise ValueError(f"A pie chart needs a category for its slices, but '{x}' is {kinds[x]}.")
            if kinds[y] != "numeric":
                raise ValueError(f"Pie slice sizes must be numeric, but '{y}' is {kinds[y]}.")
            values = data[y].dropna()
            if (values < 0).any():
                raise ValueError(f"'{y}' has negative values, which a pie chart cannot show.")
            if values.sum() <= 0:
                raise ValueError(f"'{y}' adds up to zero, so there is nothing to slice.")

    title = str(spec.get("title") or "").strip()[:80]
    if not title:
        title = f"Distribution of {x}" if chart == "histogram" else f"{y} by {x}"
    return {"chart": chart, "x": x, "y": y, "title": title}


def plan_chart_with_llm(question: str, df: pd.DataFrame, requested: bool,
                        requested_type: str | None = None) -> dict:
    """Ask the model for a chart spec. Returns the RAW parsed JSON (not yet validated)."""
    data = normalize_result(df)
    kinds = column_kinds(data)
    chain = CHART_PROMPT | get_llm()
    response = chain.invoke({
        "question": question,
        "requested": ("YES. The user explicitly asked for a chart." if requested
                      else "No. Decide whether a chart would really help."),
        "requested_type": requested_type or "not specified",
        "columns": "\n".join(f"- {c} ({kinds[c]})" for c in data.columns),
        "row_count": len(data),
        "sample": data.head(3).to_string(index=False),
    })
    return parse_spec_json(_response_text(response))