from langchain_core.prompts import ChatPromptTemplate

# Rules shared by the first attempt and the repair attempt.
_SQL_RULES = (
    "Rules:\n"
    "- Use only the table and columns listed above. Never invent columns.\n"
    "- Write a single read-only SELECT query (a WITH ... SELECT is fine).\n"
    "- Use DuckDB SQL syntax.\n"
    "- Use the dataset profile to spell text values exactly as they appear (for example 'North').\n"
    "- If a column holds dates but its type is VARCHAR, look at the sample rows and the profile to see the "
    "format. If it is not ISO (YYYY-MM-DD), parse it with try_strptime(trim(column), '%m/%d/%Y') "
    "(or the matching format). Otherwise use CAST(column AS DATE).\n"
    "- To group by month, use strftime(<parsed date>, '%Y-%m').\n"
    "- To group by year, return the year as an INTEGER: CAST(EXTRACT(year FROM <parsed date>) AS INTEGER) "
    "AS release_year. Never return the year as text.\n"
    "- Give computed columns clear aliases (for example total_revenue).\n"
    "- If the user asks for a chart or plot, or for a trend over time, return the data needed to draw it: "
    "label or date column first, then the number column(s). Do not collapse it into a single number.\n"
    "- Order matters for charts. Rankings: ORDER BY the number DESC. Time: ORDER BY the date or year ASC. "
    "Ranges or bins: order them naturally (sort by the numeric start of the range, not by the count).\n"
    "- Bar chart or pie chart: ONE row per category, label first, one number second. "
    "For a pie chart the number must not be negative and there should be at most 10 categories.\n"
    "- Line chart: one row per date or year, the date or year first, ascending.\n"
    "- Histogram (the user says 'histogram'): do NOT group, bin or count in SQL. Return ONLY the raw numeric "
    "column: SELECT column FROM data WHERE column IS NOT NULL. No LIMIT. The app draws the bins itself.\n"
    "- Scatter plot: return the two numeric columns (plus a name column if useful), WHERE both are NOT NULL, "
    "with no LIMIT and no ORDER BY unless the user asked for one.\n"
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

# V3/V4. NOTE: the JSON example uses doubled braces {{ }} because single braces mean
# "template variable" in a ChatPromptTemplate. They come out as normal braces.
CHART_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You decide whether a query result should be shown as a chart, and which one.\n\n"
     "Reply with ONLY one JSON object, no markdown and no explanation:\n"
     '{{"chart": "bar", "x": "region", "y": "total_sales", "title": "Sales by Region"}}\n\n'
     "Allowed values for \"chart\": bar, line, scatter, histogram, pie, none.\n"
     "Rules:\n"
     "- \"x\" and \"y\" must be exact column names from the column list you are given.\n"
     "- bar: x is a category (or a label), y is a number. One row per category.\n"
     "- line: x is a date or an ordered number (year, month), y is a number.\n"
     "- scatter: x and y are both numbers, to show a relationship.\n"
     "- histogram: x is a number and y is left out. Use it for the spread of one numeric column.\n"
     "- pie: x is a category (the slices), y is a non-negative number (the slice size). "
     "Use it for parts of a whole.\n"
     "- none: a single value, a single row, a plain list of text, or any result where a chart would not help.\n"
     "- If the user named a chart type, you MUST use exactly that type. Never swap it for another one.\n"
     "- If the user asked for a chart without naming a type, pick the best one. "
     "Do not answer none unless it is truly impossible.\n"
     "- The title is short (under 60 characters) and plain English."),
    ("human",
     "Question: {question}\n"
     "Explicit chart request: {requested}\n"
     "Chart type named by the user: {requested_type}\n\n"
     "Result columns:\n{columns}\n\n"
     "Number of rows: {row_count}\n\n"
     "First rows:\n{sample}"),
])