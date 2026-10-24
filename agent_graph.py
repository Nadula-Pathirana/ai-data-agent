import re
import threading

import pandas as pd
from langgraph.graph import END, START, StateGraph

import config
from llm import get_llm
from prompts import EXPLAIN_PROMPT, REPAIR_PROMPT, SQL_PROMPT
from sql_validator import closest_columns, find_missing_column, validate_sql
from state import AgentState
from viz.chart_planner import plan_chart_with_llm, validate_spec, wants_chart
from viz.chart_rules import normalize_result, suggest_chart

REFUSAL_MARKER = "CANNOT_ANSWER"


# ---------- helpers ----------

def _as_text(response) -> str:
    """Some providers return a list of content blocks instead of a plain string."""
    content = response.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def clean_sql(text: str) -> str:
    """Remove markdown fences (and any chatter around them) the model sometimes adds."""
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    return (match.group(1) if match else text).strip()


def parse_model_output(text: str) -> tuple[str, str]:
    """Return (sql, refusal_reason). Exactly one of them is non-empty."""
    text = clean_sql(text)
    if text.upper().startswith(REFUSAL_MARKER):
        reason = text[len(REFUSAL_MARKER):].lstrip(" :-").strip()
        return "", reason or "The question cannot be answered from this dataset."
    return text, ""


def run_query(con, sql: str, max_rows: int, timeout: int) -> tuple[pd.DataFrame, bool]:
    """Run one query on its own cursor. Returns (DataFrame, was_truncated).

    - fetchmany(max_rows + 1) means we never pull a giant result into memory,
      and the one extra row tells us whether the result was cut off.
    - A timer calls interrupt() if the query runs too long.
    """
    cur = con.cursor()
    timed_out = threading.Event()

    def stop_query() -> None:
        timed_out.set()
        cur.interrupt()

    timer = threading.Timer(timeout, stop_query)
    timer.start()
    try:
        cur.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
        columns = [d[0] for d in cur.description]
    except Exception as exc:
        if timed_out.is_set():
            raise TimeoutError(
                f"The query ran for more than {timeout} seconds and was stopped."
            ) from exc
        raise
    finally:
        timer.cancel()
        cur.close()

    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=columns), truncated


# ---------- nodes ----------

def generate_sql(state: AgentState) -> dict:
    chain = SQL_PROMPT | get_llm()
    response = chain.invoke({
        "schema": state["schema_text"],
        "profile": state.get("profile_text", ""),
        "question": state["question"],
    })
    sql, refusal = parse_model_output(_as_text(response))
    return {"sql": sql, "refusal": refusal}


def check_sql(state: AgentState) -> dict:
    ok, message = validate_sql(state["sql"])
    return {"last_error": "" if ok else message}


def execute_sql(state: AgentState) -> dict:
    try:
        df, truncated = run_query(
            state["con"], state["sql"], config.MAX_RESULT_ROWS, config.QUERY_TIMEOUT_SECONDS
        )
        return {"result_df": df, "truncated": truncated, "last_error": ""}
    except Exception as exc:
        return {"last_error": f"The SQL query failed: {exc}"}


def repair_sql(state: AgentState) -> dict:
    """Show the model its own failed SQL plus the error, and ask for a fix."""
    chain = REPAIR_PROMPT | get_llm()
    response = chain.invoke({
        "schema": state["schema_text"],
        "profile": state.get("profile_text", ""),
        "question": state["question"],
        "failed_sql": state["sql"],
        "error": state["last_error"],
    })
    sql, refusal = parse_model_output(_as_text(response))
    history = state.get("attempts", []) + [{"sql": state["sql"], "error": state["last_error"]}]
    return {
        "sql": sql,
        "refusal": refusal,
        "retry_count": state.get("retry_count", 0) + 1,
        "attempts": history,
    }


def plan_visualization(state: AgentState) -> dict:
    """Ask the LLM for a chart, but never trust the answer.

    chart_source tells the router what happened:
      "skipped"        nothing worth drawing (0 or 1 rows): no LLM call was made
      "llm"            the model's spec passed validation (the spec may legitimately be "no chart")
      "needs_fallback" the model failed, or its spec was unusable: go to rule_based_fallback
    """
    question = state["question"]
    requested = wants_chart(question)
    data = normalize_result(state["result_df"])

    def outcome(spec, source, issue=""):
        return {"chart_spec": spec, "chart_source": source,
                "chart_issue": issue, "chart_requested": requested}

    # A single value or a single row never needs a chart. Skip the LLM call entirely.
    if len(data) < 2:
        return outcome(None, "skipped")

    if not config.USE_LLM_CHART_PLANNER:  # rules-only mode
        return outcome(None, "needs_fallback")

    try:
        spec = validate_spec(plan_chart_with_llm(question, data, requested), data)
    except Exception as exc:  # bad JSON, unknown column, network error... all end up here
        return outcome(None, "needs_fallback", str(exc)[:200])

    if spec["chart"] == "none":
        if requested:  # the user asked for a chart, so "none" is not acceptable
            return outcome(None, "needs_fallback", "The model chose no chart although one was requested.")
        return outcome(None, "llm")
    return outcome(spec, "llm")


def rule_based_fallback(state: AgentState) -> dict:
    """Pick a chart with plain rules. The result goes through the same validation as the AI's."""
    data = normalize_result(state["result_df"])
    try:
        spec = validate_spec(suggest_chart(data, state["question"]), data)
    except ValueError:
        spec = {"chart": "none"}
    return {"chart_spec": None if spec["chart"] == "none" else spec, "chart_source": "rules"}


def explain(state: AgentState) -> dict:
    df = state["result_df"]

    # No LLM call needed (and no chance of invented facts) when there is nothing to explain.
    if df.empty:
        return {"explanation": (
            "The query ran fine, but no rows matched. Try loosening a filter, and check the "
            "spelling of text values in the dataset profile."
        )}

    notes = []
    if state.get("truncated"):
        notes.append(f"The result was cut off at {config.MAX_RESULT_ROWS} rows.")
    if len(df) > 50:
        notes.append("Only the first 50 rows are shown to you here.")

    try:
        chain = EXPLAIN_PROMPT | get_llm()
        response = chain.invoke({
            "question": state["question"],
            "sql": state["sql"],
            "result": df.head(50).to_string(index=False),
            "note": " ".join(notes) or "None",
        })
        return {"explanation": _as_text(response)}
    except Exception as exc:
        # The query worked, so don't throw the result away just because the summary failed.
        return {"explanation": (
            f"I ran the query successfully but could not write a summary ({type(exc).__name__}). "
            "The result table is shown below."
        )}


def fail_gracefully(state: AgentState) -> dict:
    """Turn whatever went wrong into one friendly message. No stack traces."""
    if state.get("refusal"):
        return {"error": f"I can't answer that from this dataset. {state['refusal']}"}

    attempts_made = state.get("retry_count", 0) + 1
    message = f"I couldn't produce a working query after {attempts_made} attempt(s)."

    # Look through every error we saw for an unknown column name.
    errors = [a["error"] for a in state.get("attempts", [])] + [state.get("last_error", "")]
    missing = next((m for m in map(find_missing_column, errors) if m), None)
    if missing:
        columns = state.get("columns", [])
        close = closest_columns(missing, columns)
        message += f" Your question seems to need a column called '{missing}', which doesn't exist."
        if close:
            message += f" The closest columns are: {', '.join(close)}."
        elif columns:
            message += f" Available columns: {', '.join(columns)}."
    else:
        message += " Try rephrasing the question, or ask about specific columns."

    return {"error": message}


# ---------- routing ----------

def _retry_or_fail(state: AgentState) -> str:
    return "repair_sql" if state.get("retry_count", 0) < config.MAX_RETRIES else "fail_gracefully"


def route_after_generation(state: AgentState) -> str:
    """Used after generate_sql AND after repair_sql (both can answer CANNOT_ANSWER)."""
    return "fail_gracefully" if state.get("refusal") else "check_sql"


def route_after_validation(state: AgentState) -> str:
    return _retry_or_fail(state) if state.get("last_error") else "execute_sql"


def route_after_execution(state: AgentState) -> str:
    return _retry_or_fail(state) if state.get("last_error") else "plan_visualization"


def route_after_planning(state: AgentState) -> str:
    return "rule_based_fallback" if state.get("chart_source") == "needs_fallback" else "explain"


# ---------- graph ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_sql", generate_sql)
    graph.add_node("check_sql", check_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("repair_sql", repair_sql)
    graph.add_node("plan_visualization", plan_visualization)
    graph.add_node("rule_based_fallback", rule_based_fallback)
    graph.add_node("explain", explain)
    graph.add_node("fail_gracefully", fail_gracefully)

    graph.add_edge(START, "generate_sql")
    graph.add_conditional_edges(
        "generate_sql", route_after_generation,
        {"check_sql": "check_sql", "fail_gracefully": "fail_gracefully"},
    )
    graph.add_conditional_edges(
        "check_sql", route_after_validation,
        {"execute_sql": "execute_sql", "repair_sql": "repair_sql", "fail_gracefully": "fail_gracefully"},
    )
    graph.add_conditional_edges(
        "execute_sql", route_after_execution,
        {"plan_visualization": "plan_visualization", "repair_sql": "repair_sql",
         "fail_gracefully": "fail_gracefully"},
    )
    graph.add_conditional_edges(
        "plan_visualization", route_after_planning,
        {"rule_based_fallback": "rule_based_fallback", "explain": "explain"},
    )
    graph.add_edge("rule_based_fallback", "explain")
    graph.add_conditional_edges(
        "repair_sql", route_after_generation,
        {"check_sql": "check_sql", "fail_gracefully": "fail_gracefully"},
    )
    graph.add_edge("explain", END)
    graph.add_edge("fail_gracefully", END)

    return graph.compile()


agent = build_graph()


def run_agent(question: str, schema_text: str, profile_text: str, columns: list[str], con) -> AgentState:
    initial: AgentState = {
        "question": question,
        "schema_text": schema_text,
        "profile_text": profile_text,
        "columns": columns,
        "con": con,
        "retry_count": 0,
        "attempts": [],
        "last_error": "",
        "refusal": "",
        "chart_spec": None,
        "chart_source": "",
        "chart_issue": "",
        "chart_requested": False,
    }
    # Each retry loop costs about 3 graph steps, so raise LangGraph's default limit of 25 if needed.
    return agent.invoke(initial, config={"recursion_limit": 10 + 4 * config.MAX_RETRIES})