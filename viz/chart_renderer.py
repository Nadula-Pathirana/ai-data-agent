"""Turns a validated chart spec plus a result DataFrame into a Streamlit chart."""
import re
from typing import NamedTuple

import altair as alt  # ships with Streamlit, no extra install
import pandas as pd
import streamlit as st

import config
from viz.chart_rules import normalize_result


class ChartData(NamedTuple):
    data: pd.DataFrame   # only the needed columns, safe names, cleaned and ordered
    x: str               # safe name of the x column in `data`
    y: str | None        # safe name of the y column in `data` (None for histograms)
    x_label: str         # original names, used for axis titles
    y_label: str | None
    note: str            # e.g. "Showing the top 20 of 57 categories."


def _safe_name(name: str) -> str:
    """'Total Sales ($)' -> 'Total_Sales'. Odd characters can confuse chart libraries."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or "col"


def prepare_chart_data(df: pd.DataFrame, spec: dict, max_categories: int | None = None) -> ChartData:
    """Select, clean, order and limit the data for one chart. No Streamlit calls, so it is testable."""
    max_categories = max_categories or config.MAX_CHART_CATEGORIES
    kind, x_col, y_col = spec["chart"], spec["x"], spec.get("y")

    data = normalize_result(df)
    wanted = [x_col] + ([y_col] if y_col else [])
    subset = data[wanted].dropna().copy()

    x = _safe_name(x_col)
    y = _safe_name(y_col) if y_col else None
    if y and y == x:
        y = f"{y}_2"
    subset.columns = [x] + ([y] if y else [])

    note = ""
    if kind == "bar":
        if len(subset) > max_categories:
            note = f"Showing the top {max_categories} of {len(subset)} categories."
            subset = subset.nlargest(max_categories, y)
        subset = subset.sort_values(y, ascending=False)  # tallest bar first
        if pd.api.types.is_datetime64_any_dtype(subset[x]):
            subset = subset.sort_values(x)               # dates read best in time order
            subset[x] = subset[x].dt.strftime("%Y-%m-%d")
    elif kind == "line":
        subset = subset.sort_values(x)

    return ChartData(subset, x, y, x_col, y_col, note)


def render_chart(spec: dict, df: pd.DataFrame) -> None:
    """Draw the chart. Callers should wrap this in try/except: a chart must never crash the app."""
    prepared = prepare_chart_data(df, spec)
    data, x, y = prepared.data, prepared.x, prepared.y
    kind = spec["chart"]

    title = spec["title"].replace("$", "\\$")  # Streamlit would treat $...$ as math
    st.markdown(f"**{title}**")

    if kind == "line":
        st.line_chart(data, x=x, y=y)
    elif kind == "scatter":
        st.scatter_chart(data, x=x, y=y)
    elif kind == "bar":
        # Altair instead of st.bar_chart: it keeps OUR order (tallest first) instead of re-sorting
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{x}:N", sort=None, title=prepared.x_label, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y(f"{y}:Q", title=prepared.y_label),
            tooltip=[alt.Tooltip(f"{x}:N", title=prepared.x_label),
                     alt.Tooltip(f"{y}:Q", title=prepared.y_label)],
        )
        st.altair_chart(chart, width="stretch")
    elif kind == "histogram":
        # Streamlit has no built-in histogram. Altair bins the numbers and counts rows per bin.
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{x}:Q", bin=alt.Bin(maxbins=20), title=prepared.x_label),
            y=alt.Y("count()", title="Number of rows"),
        )
        st.altair_chart(chart, width="stretch")

    if prepared.note:
        st.caption(prepared.note)