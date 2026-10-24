"""Turns a validated chart spec plus a result DataFrame into a Streamlit chart."""
import re
from typing import NamedTuple

import altair as alt  # ships with Streamlit, no extra install
import pandas as pd
import streamlit as st

import config
from viz.chart_rules import histogram_bins, is_year_like, normalize_result


class ChartData(NamedTuple):
    data: pd.DataFrame   # only the needed columns, safe names, cleaned and ordered
    x: str               # safe name of the x column in `data`
    y: str | None        # safe name of the y column in `data` (None for histograms)
    x_label: str         # original names, used for axis titles
    y_label: str | None
    note: str            # e.g. "Showing the first 20 of 57 categories."


def _safe_name(name: str) -> str:
    """'Total Sales ($)' -> 'Total_Sales'. Odd characters can confuse chart libraries."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or "col"


def _as_labels(series: pd.Series) -> pd.Series:
    """Turn an x column into clean text labels: 1903.0 -> '1903', timestamps -> '2024-03-15'."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.dt.strftime("%Y-%m-%d")
    if pd.api.types.is_numeric_dtype(series):
        return series.map(lambda v: str(int(v)) if float(v).is_integer() else str(v))
    return series.astype(str)


def _unique_labels(labels: pd.Series) -> pd.Series:
    """Two rows called 'Titanic' become 'Titanic' and 'Titanic (2)' so each keeps its own bar/slice."""
    seen = labels.groupby(labels).cumcount()
    return pd.Series([f"{lbl} ({n + 1})" if n else lbl for lbl, n in zip(labels, seen)], index=labels.index)


def prepare_chart_data(df: pd.DataFrame, spec: dict, max_categories: int | None = None) -> ChartData:
    """Select, clean, order and limit the data for one chart. No Streamlit calls, so it is testable."""
    max_categories = max_categories or config.MAX_CHART_CATEGORIES
    kind, x_col, y_col = spec["chart"], spec["x"], spec.get("y")
    data = normalize_result(df)

    # FIX 3: histograms are binned HERE from the raw numbers (never from SQL-made ranges).
    if kind == "histogram":
        bins = histogram_bins(data[x_col], config.HISTOGRAM_BINS)
        return ChartData(bins, "bin_start", "count", x_col, "Number of rows", "")

    wanted = [x_col] + ([y_col] if y_col else [])
    subset = data[wanted].dropna().copy()

    x = _safe_name(x_col)
    y = _safe_name(y_col) if y_col else None
    if y and y == x:
        y = f"{y}_2"
    subset.columns = [x] + ([y] if y else [])

    note = ""
    if kind == "bar":
        # FIX 2 + ordering: keep the order the SQL gave us (it already did ORDER BY). The old code
        # re-sorted by height, which scrambled bins like '60-70, 70-80, 50-60' and years.
        subset[x] = _unique_labels(_as_labels(subset[x]))
        if len(subset) > max_categories:
            note = (f"Showing the first {max_categories} of {len(subset)} bars "
                    "(in the order of the query result).")
            subset = subset.head(max_categories)
    elif kind == "line":
        subset = subset.sort_values(x)
    elif kind == "pie":
        subset[x] = _unique_labels(_as_labels(subset[x]))
        subset = subset[subset[y] > 0].sort_values(y, ascending=False)
        max_slices = config.MAX_PIE_SLICES
        if len(subset) > max_slices:
            top = subset.head(max_slices - 1)
            rest = subset.iloc[max_slices - 1:]
            other = pd.DataFrame({x: ["Other"], y: [rest[y].sum()]})
            note = f"The {len(rest)} smallest categories are grouped into 'Other'."
            subset = pd.concat([top, other], ignore_index=True)
        total = subset[y].sum()
        subset["pie_share"] = subset[y] / total
        subset["pie_label"] = [f"{lbl} ({share:.1%})" for lbl, share in zip(subset[x], subset["pie_share"])]

    return ChartData(subset, x, y, x_col, y_col, note)


def render_chart(spec: dict, df: pd.DataFrame) -> None:
    """Draw the chart. Callers should wrap this in try/except: a chart must never crash the app."""
    prepared = prepare_chart_data(df, spec)
    data, x, y = prepared.data, prepared.x, prepared.y
    kind = spec["chart"]

    title = spec["title"].replace("$", "\\$")  # Streamlit would treat $...$ as math
    st.markdown(f"**{title}**")

    if kind == "line":
        # Altair instead of st.line_chart so a year axis reads 1903, not 1,903.
        is_date = pd.api.types.is_datetime64_any_dtype(data[x])
        year_like = (not is_date) and is_year_like(data[x])
        x_type = "T" if is_date else "Q"
        fmt = {"format": "d"} if year_like else {}
        chart = alt.Chart(data).mark_line(point=True).encode(
            x=alt.X(f"{x}:{x_type}", title=prepared.x_label, axis=alt.Axis(**fmt)),
            y=alt.Y(f"{y}:Q", title=prepared.y_label, scale=alt.Scale(zero=False)),
            tooltip=[alt.Tooltip(f"{x}:{x_type}", title=prepared.x_label, **fmt),
                     alt.Tooltip(f"{y}:Q", title=prepared.y_label)],
        )
        st.altair_chart(chart, width="stretch")
    elif kind == "scatter":
        st.scatter_chart(data, x=x, y=y)
    elif kind == "bar":
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X(f"{x}:N", sort=None, title=prepared.x_label, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y(f"{y}:Q", title=prepared.y_label),
            tooltip=[alt.Tooltip(f"{x}:N", title=prepared.x_label),
                     alt.Tooltip(f"{y}:Q", title=prepared.y_label)],
        )
        st.altair_chart(chart, width="stretch")
    elif kind == "histogram":
        # Bars touch each other (a real histogram): each bar spans bin_start -> bin_end.
        big = data["bin_end"].abs().max() >= 10_000
        axis = alt.Axis(format="~s") if big else alt.Axis()
        chart = alt.Chart(data).mark_bar().encode(
            x=alt.X("bin_start:Q", bin="binned", title=prepared.x_label, axis=axis),
            x2="bin_end:Q",
            y=alt.Y("count:Q", title="Number of rows"),
            tooltip=[alt.Tooltip("bin_start:Q", title="From", format=",.2f"),
                     alt.Tooltip("bin_end:Q", title="To", format=",.2f"),
                     alt.Tooltip("count:Q", title="Rows")],
        )
        st.altair_chart(chart, width="stretch")
    elif kind == "pie":
        slices = list(data["pie_label"])
        chart = alt.Chart(data).mark_arc(outerRadius=130).encode(
            theta=alt.Theta(f"{y}:Q", stack=True),
            color=alt.Color("pie_label:N", sort=slices, title=prepared.x_label),
            order=alt.Order(f"{y}:Q", sort="descending"),
            tooltip=[alt.Tooltip(f"{x}:N", title=prepared.x_label),
                     alt.Tooltip(f"{y}:Q", title=prepared.y_label, format=",.2f"),
                     alt.Tooltip("pie_share:Q", title="Share", format=".1%")],
        ).properties(height=340)
        st.altair_chart(chart, width="stretch")

    if prepared.note:
        st.caption(prepared.note)