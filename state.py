from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    # ---- inputs (set once by run_agent) ----
    question: str
    schema_text: str      # table, columns, types, sample rows
    profile_text: str     # compact dataset profile for the prompt
    columns: list[str]    # column names, used for "closest column" hints
    con: Any              # DuckDB connection (already locked down)

    # ---- working fields (change while the graph loops) ----
    sql: str              # the current candidate query
    last_error: str       # why the current query is not good ("" = no problem)
    retry_count: int      # how many repairs have been attempted
    attempts: list[dict]  # history of failed attempts: {"sql": ..., "error": ...}
    refusal: str          # set when the model says the question cannot be answered

    # ---- outputs ----
    result_df: Any        # pandas DataFrame
    truncated: bool       # True if the result was cut at MAX_RESULT_ROWS
    explanation: str
    error: str            # final, user-friendly failure message (only set on failure)

    # ---- V3: visualization ----
    chart_spec: dict | None  # validated {"chart","x","y","title"}, or None = no chart
    chart_source: str        # "llm" | "skipped" | "needs_fallback" | "rules"
    chart_issue: str         # why the AI's chart plan was rejected ("" = nothing wrong)
    chart_requested: bool    # True if the user explicitly asked for a chart

    # ---- V4 ----
    row_limit: int           # how many rows this query was allowed to return (200 normally, more for histogram/scatter)
    chart_note: str          # message for the user, e.g. "You asked for a line chart, but ..."