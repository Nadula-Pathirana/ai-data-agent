from langchain_core.prompts import ChatPromptTemplate

# Rules shared by the first attempt and the repair attempt.
_SQL_RULES = (
    "Rules:\n"
    "- Use only the table and columns listed above. Never invent columns.\n"
    "- Write a single read-only SELECT query (a WITH ... SELECT is fine).\n"
    "- Use DuckDB SQL syntax.\n"
    "- Use the dataset profile to spell text values exactly as they appear (for example 'North').\n"
    "- If a column holds dates but its type is VARCHAR, convert it with CAST(column AS DATE) "
    "before using date functions.\n"
    "- Give computed columns clear aliases (for example total_revenue).\n"
    "- If the question needs a column that does not exist, is unrelated to this data, or is too vague "
    "to answer (for example 'show me the best ones' with no metric), do NOT guess. "
    "Reply with exactly one line: CANNOT_ANSWER: <short reason, naming the closest "
    "available columns if relevant>.\n"
    "- Otherwise return ONLY the SQL. No explanation, no markdown."
)

_CONTEXT = "{schema}\n\n{profile}\n\n"

SQL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert data analyst who writes DuckDB SQL.\n"
     "Write ONE SQL query that answers the user's question using only the table below.\n\n"
     + _CONTEXT + _SQL_RULES),
    ("human", "{question}"),
])

REPAIR_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert data analyst who writes DuckDB SQL.\n"
     "A query you wrote failed. Fix it so it answers the user's question, "
     "using only the table below.\n\n"
     + _CONTEXT + _SQL_RULES),
    ("human",
     "Question: {question}\n\n"
     "Failed SQL:\n{failed_sql}\n\n"
     "Error message:\n{error}\n\n"
     "Write the corrected query."),
])

EXPLAIN_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful data analyst explaining query results to a non-technical user.\n"
     "Rules:\n"
     "- Use ONLY the numbers and values in the result table. Never invent or estimate numbers.\n"
     "- If a note says the result was cut off, mention that only the first rows were considered.\n"
     "- Keep it short: 2 to 4 sentences, in plain English."),
    ("human",
     "Question: {question}\n\n"
     "SQL used:\n{sql}\n\n"
     "Result table:\n{result}\n\n"
     "Notes: {note}"),
])





"""
Module: Prompts Configuration (prompts.py)
Description:
This module defines the core ChatPromptTemplates used by the AI Data Analysis 
Agent to interact with LLMs via LangChain[cite: 1]. It structures system personas 
and human queries to ensure safe, accurate, and structured outputs[cite: 1].

Key components include:
1. SQL_PROMPT:
   - Role: Directs the LLM to act as an expert data analyst writing DuckDB SQL[cite: 1].
   - Inputs: Injects the database schema metadata and the user's natural language question[cite: 1].
   - Rules: Enforces read-only SELECT statements, restricts column usage to prevent 
     hallucinations, requires clear aliases for computed columns, and demands 
     raw SQL output with zero conversational explanation or markdown formatting[cite: 1].

2. EXPLAIN_PROMPT:
   - Role: Directs the LLM to act as a careful data analyst explaining data query 
     results in plain, non-technical English[cite: 1].
   - Inputs: Combines the original user question, the generated SQL query, and the 
     resulting data table[cite: 1].
   - Rules: Enforces strict adherence to table values (preventing invented numbers), 
     handles empty tables gracefully, and restricts answers to concise 2-to-4-sentence explanations[cite: 1].
"""