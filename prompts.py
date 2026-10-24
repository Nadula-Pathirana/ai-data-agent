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
    "- To group by month, use strftime(CAST(date_column AS DATE), '%Y-%m').\n"
    "- Give computed columns clear aliases (for example total_revenue).\n"
    "- If the user asks for a chart or plot, or for a trend over time, return the data needed to draw it: "
    "one row per point or category, label or date column first, dates in ascending order. "
    "Do not collapse it into a single number.\n"
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

# V3. NOTE: the JSON example uses doubled braces {{ }} because single braces mean
# "template variable" in a ChatPromptTemplate. They come out as normal braces.
CHART_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decide whether a query result should be shown as a chart, and which one.\n\n"
     "Reply with ONLY one JSON object, no markdown and no explanation:\n"
     '{{"chart": "bar", "x": "region", "y": "total_sales", "title": "Sales by Region"}}\n\n'
     "Allowed values for \"chart\": bar, line, scatter, histogram, none.\n"
     "Rules:\n"
     "- \"x\" and \"y\" must be exact column names from the column list you are given.\n"
     "- bar: x is a category (or a label), y is a number. One row per category.\n"
     "- line: x is a date or an ordered number (year, month), y is a number.\n"
     "- scatter: x and y are both numbers, to show a relationship.\n"
     "- histogram: x is a number and y is left out. Use it for the spread of one numeric column.\n"
     "- none: a single value, a single row, a plain list of text, or any result where a chart would not help.\n"
     "- If the user explicitly asked for a chart, do not answer none unless it is truly impossible. "
     "If they named a chart type and the columns allow it, use that type.\n"
     "- The title is short (under 60 characters) and plain English."),
    ("human",
     "Question: {question}\n"
     "Explicit chart request: {requested}\n\n"
     "Result columns:\n{columns}\n\n"
     "Number of rows: {row_count}\n\n"
     "First rows:\n{sample}"),
])