import os

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---- V2 ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))                       # repairs after the first attempt
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30"))  # stop slow queries
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))                      # upload size limit

# ---- V3 additions ----
# true  = the LLM proposes the chart, rules are the safety net
# false = rules only (no extra LLM call per question, handy for testing or saving cost)
USE_LLM_CHART_PLANNER = os.getenv("USE_LLM_CHART_PLANNER", "true").strip().lower() in ("1", "true", "yes")
MAX_CHART_CATEGORIES = int(os.getenv("MAX_CHART_CATEGORIES", "20"))    # bar charts show at most this many bars