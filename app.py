import streamlit as st

import config
from agent_graph import run_agent
from data_loader import get_schema, load_into_duckdb, read_file
from db_security import lock_down
from profiler import build_profile
from viz.chart_renderer import render_chart

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("AI Data Analysis Agent")

if not config.OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()


# ---------- per-session state ----------
# Streamlit gives every browser session its own st.session_state, so each user
# automatically gets their own DuckDB connection, data and chat history.

def fresh_state() -> dict:
    return {
        "con": None, "df": None, "schema_df": None, "schema_text": "",
        "profile_df": None, "profile_text": "", "columns": [],
        "file_id": None, "messages": [],
    }


for key, value in fresh_state().items():
    st.session_state.setdefault(key, value)
st.session_state.setdefault("uploader_key", 0)


def close_connection() -> None:
    con = st.session_state.get("con")
    if con is not None:
        try:
            con.close()
        except Exception:
            pass


def reset_dataset() -> None:
    """Close the old connection, clear everything and empty the upload box."""
    close_connection()
    next_uploader_key = st.session_state.uploader_key + 1
    st.session_state.update(fresh_state())
    st.session_state.uploader_key = next_uploader_key  # a new key gives a brand-new, empty uploader


def clear_chat() -> None:
    st.session_state.messages = []


def load_dataset(uploaded) -> dict:
    """Read the file, build the database, profile it, then lock it down."""
    df = read_file(uploaded)
    con = load_into_duckdb(df)
    try:
        schema_df, schema_text = get_schema(con)
        profile_df, profile_text = build_profile(con)
        lock_down(con)  # after the data is loaded, before any generated SQL can run
    except Exception:
        con.close()
        raise
    return {
        "con": con, "df": df, "schema_df": schema_df, "schema_text": schema_text,
        "profile_df": profile_df, "profile_text": profile_text,
        "columns": schema_df["column_name"].tolist(),
    }


# ---------- sidebar ----------
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader(
        "Upload a CSV or Excel file",
        type=["csv", "xlsx"],
        key=f"uploader_{st.session_state.uploader_key}",
    )
    st.caption(f"Maximum file size: {config.MAX_FILE_MB} MB")
    st.button("Clear chat", on_click=clear_chat)
    st.button("Load new dataset", on_click=reset_dataset)

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.file_id != file_id:  # new file -> (re)load everything
        try:
            loaded = load_dataset(uploaded)
        except ValueError as exc:  # friendly messages raised by read_file
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Could not load the file: {exc}")
            st.stop()
        close_connection()  # release the previous dataset's connection, if any
        st.session_state.update(loaded)
        st.session_state.update(file_id=file_id, messages=[])

if st.session_state.con is None:
    st.info("Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

# ---------- dataset overview ----------
with st.expander("Dataset preview and schema", expanded=True):
    left, right = st.columns([2, 1])
    with left:
        st.caption(f"{len(st.session_state.df)} rows × {len(st.session_state.df.columns)} columns")
        st.dataframe(st.session_state.df.head(10), width="stretch")
    with right:
        st.caption("Columns and types")
        st.dataframe(st.session_state.schema_df, width="stretch", hide_index=True)

with st.expander("Dataset profile"):
    st.caption("Calculated by DuckDB. The agent reads this summary before writing SQL.")
    st.dataframe(st.session_state.profile_df, width="stretch", hide_index=True)


# ---------- chat ----------
def safe_text(text: str) -> str:
    """Streamlit treats $...$ as math, which garbles answers like '$3,140 and $2,305'."""
    return text.replace("$", r"\$")


def render_chart_section(msg: dict) -> None:
    """Draw the chart (if any) and explain where it came from. Never lets a chart error escape."""
    spec = msg.get("chart")
    if spec:
        try:
            render_chart(spec, msg["df"])
        except Exception as exc:  # a broken chart must never break the chat
            st.caption(f"The chart could not be drawn ({type(exc).__name__}).")
        else:
            if msg.get("chart_source") == "llm":
                st.caption("Chart planned by the AI and checked by the app.")
            else:
                reason = f" (the AI's plan was not usable: {msg['chart_issue']})" if msg.get("chart_issue") else ""
                st.caption(f"Chart chosen by simple rules{reason}.")
        with st.expander("Chart spec"):
            st.json(spec)
    elif msg.get("chart_requested"):
        st.caption("You asked for a chart, but I could not find a sensible one for this result.")


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(safe_text(msg["content"]))

        retries = msg.get("retries", 0)
        if retries and not msg.get("failed"):
            plural = "y" if retries == 1 else "ies"
            st.caption(f"The first query did not work, so the agent repaired it automatically ({retries} retr{plural}).")

        if msg.get("sql"):
            with st.expander("SQL used"):
                st.code(msg["sql"], language="sql")

        if msg.get("attempts"):
            with st.expander(f"Failed attempts ({len(msg['attempts'])})"):
                for number, attempt in enumerate(msg["attempts"], start=1):
                    st.markdown(f"**Attempt {number}**")
                    st.code(attempt["sql"] or "(no SQL)", language="sql")
                    st.caption(attempt["error"])

        if msg.get("df") is not None:
            st.dataframe(msg["df"], width="stretch")
            if msg.get("truncated"):
                st.caption(f"Only the first {config.MAX_RESULT_ROWS} rows are shown.")
            render_chart_section(msg)  # the chart goes under the table


for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input("Ask a question about your data"):
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner("Thinking..."):
        try:
            result = run_agent(
                question,
                st.session_state.schema_text,
                st.session_state.profile_text,
                st.session_state.columns,
                st.session_state.con,
            )
            if result.get("error"):
                reply = {
                    "role": "assistant",
                    "content": result["error"],
                    "sql": result.get("sql"),
                    "attempts": result.get("attempts", []),
                    "retries": result.get("retry_count", 0),
                    "failed": True,
                }
            else:
                reply = {
                    "role": "assistant",
                    "content": result["explanation"],
                    "sql": result["sql"],
                    "df": result["result_df"],
                    "truncated": result.get("truncated", False),
                    "attempts": result.get("attempts", []),
                    "retries": result.get("retry_count", 0),
                    "chart": result.get("chart_spec"),
                    "chart_source": result.get("chart_source", ""),
                    "chart_issue": result.get("chart_issue", ""),
                    "chart_requested": result.get("chart_requested", False),
                }
        except Exception as exc:  # e.g. bad API key, wrong model name, network problem
            reply = {
                "role": "assistant",
                "content": (
                    "Something went wrong while talking to the model. Check your API key, "
                    f"MODEL_NAME and internet connection. ({type(exc).__name__}: {exc})"
                ),
                "failed": True,
            }

    st.session_state.messages.append(reply)
    render_message(reply)