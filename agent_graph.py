import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import config
from llm import get_llm
from prompts import EXPLAIN_PROMPT, SQL_PROMPT
from sql_validator import validate_sql


class AgentState(TypedDict, total=False):
    question: str
    schema_text: str
    con: Any          # DuckDB connection
    sql: str
    result_df: Any    # pandas DataFrame
    explanation: str
    error: str


def clean_sql(text: str) -> str:
    """Remove markdown fences the model sometimes adds."""
    text = text.strip()
    text = re.sub(r"^```(?:sql)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


# ---------- nodes ----------

def generate_sql(state: AgentState) -> dict:
    chain = SQL_PROMPT | get_llm()
    response = chain.invoke({"schema": state["schema_text"], "question": state["question"]})
    return {"sql": clean_sql(response.content)}


def check_sql(state: AgentState) -> dict:
    ok, message = validate_sql(state["sql"])
    return {"error": "" if ok else message}


def execute_sql(state: AgentState) -> dict:
    try:
        df = state["con"].cursor().execute(state["sql"]).df()
        return {"result_df": df.head(config.MAX_RESULT_ROWS), "error": ""}
    except Exception as exc:
        return {"error": f"The SQL query failed: {exc}"}


def explain(state: AgentState) -> dict:
    df = state["result_df"]
    result_text = "(the query returned no rows)" if df.empty else df.head(50).to_string(index=False)
    chain = EXPLAIN_PROMPT | get_llm()
    response = chain.invoke({
        "question": state["question"],
        "sql": state["sql"],
        "result": result_text,
    })
    return {"explanation": response.content}


# ---------- routing ----------

def route_after_validation(state: AgentState) -> str:
    return "stop" if state.get("error") else "execute_sql"


def route_after_execution(state: AgentState) -> str:
    return "stop" if state.get("error") else "explain"


# ---------- graph ----------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("generate_sql", generate_sql)
    graph.add_node("check_sql", check_sql)
    graph.add_node("execute_sql", execute_sql)
    graph.add_node("explain", explain)

    graph.add_edge(START, "generate_sql")
    graph.add_edge("generate_sql", "check_sql")
    graph.add_conditional_edges(
        "check_sql", route_after_validation, {"execute_sql": "execute_sql", "stop": END}
    )
    graph.add_conditional_edges(
        "execute_sql", route_after_execution, {"explain": "explain", "stop": END}
    )
    graph.add_edge("explain", END)

    return graph.compile()


agent = build_graph()


def run_agent(question: str, schema_text: str, con) -> AgentState:
    return agent.invoke({"question": question, "schema_text": schema_text, "con": con})