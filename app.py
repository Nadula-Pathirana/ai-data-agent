import streamlit as st

import config
from agent_graph import run_agent
from data_loader import get_schema, load_into_duckdb, read_file

st.set_page_config(page_title="AI Data Analysis Agent", layout="wide")
st.title("AI Data Analysis Agent")

if not config.OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

# ---- session state defaults ----
for key, default in [
    ("con", None), ("df", None), ("schema_df", None),
    ("schema_text", ""), ("file_id", None), ("messages", []),
]:
    st.session_state.setdefault(key, default)

# ---- sidebar: upload ----
with st.sidebar:
    st.header("Dataset")
    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
    if st.button("Clear chat"):
        st.session_state.messages = []

if uploaded is not None:
    file_id = (uploaded.name, uploaded.size)
    if st.session_state.file_id != file_id:  # new file -> (re)load everything
        try:
            df = read_file(uploaded)
            con = load_into_duckdb(df)
            schema_df, schema_text = get_schema(con)
        except Exception as exc:
            st.error(f"Could not load the file: {exc}")
            st.stop()
        st.session_state.update(
            con=con, df=df, schema_df=schema_df, schema_text=schema_text,
            file_id=file_id, messages=[],
        )

if st.session_state.con is None:
    st.info("Upload a CSV or Excel file in the sidebar to get started.")
    st.stop()

# ---- dataset overview ----
with st.expander("Dataset preview and schema", expanded=True):
    left, right = st.columns([2, 1])
    with left:
        st.caption(f"{len(st.session_state.df)} rows × {len(st.session_state.df.columns)} columns")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
    with right:
        st.caption("Columns and types")
        st.dataframe(st.session_state.schema_df, use_container_width=True, hide_index=True)


# ---- chat ----
def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sql"):
            with st.expander("SQL used"):
                st.code(msg["sql"], language="sql")
        if msg.get("df") is not None:
            st.dataframe(msg["df"], use_container_width=True)


for message in st.session_state.messages:
    render_message(message)

if question := st.chat_input("Ask a question about your data"):
    user_msg = {"role": "user", "content": question}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.spinner("Thinking..."):
        try:
            result = run_agent(question, st.session_state.schema_text, st.session_state.con)
            if result.get("error"):
                reply = {
                    "role": "assistant",
                    "content": f"Sorry, I couldn't answer that. {result['error']}",
                    "sql": result.get("sql"),
                }
            else:
                reply = {
                    "role": "assistant",
                    "content": result["explanation"],
                    "sql": result["sql"],
                    "df": result["result_df"],
                }
        except Exception as exc:  # e.g. bad API key, network problem
            reply = {"role": "assistant", "content": f"Something went wrong: {exc}"}

    st.session_state.messages.append(reply)
    render_message(reply)