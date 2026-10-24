import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# ---- V2 ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))                       # repairs after the first attempt
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))  # stop slow queries
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))                      # upload size limit

# ---- V3 additions ----
# true  = the LLM proposes the chart, rules are the safety net
# false = rules only (no extra LLM call per question, handy for testing or saving cost)
USE_LLM_CHART_PLANNER = os.getenv("USE_LLM_CHART_PLANNER", "true").strip().lower() in ("1", "true", "yes")
MAX_CHART_CATEGORIES = int(os.getenv("MAX_CHART_CATEGORIES", "20"))    # bar charts show at most this many bars

# ---- V4 additions ----
MAX_PIE_SLICES = int(os.getenv("MAX_PIE_SLICES", "8"))              # a pie shows at most this many slices (rest -> "Other")
HISTOGRAM_BINS = int(os.getenv("HISTOGRAM_BINS", "20"))             # maximum number of histogram bars
HISTOGRAM_MAX_ROWS = int(os.getenv("HISTOGRAM_MAX_ROWS", "100000"))  # histograms need the RAW values, so allow many rows
SCATTER_MAX_ROWS = int(os.getenv("SCATTER_MAX_ROWS", "5000"))       # scatter plots need many points, not just 200