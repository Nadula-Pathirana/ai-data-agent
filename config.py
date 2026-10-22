import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "") #The second argument is the fallback value. 
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
MAX_RESULT_ROWS = int(os.getenv("MAX_RESULT_ROWS", "200"))
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ---- V2 additions ----
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))                       
QUERY_TIMEOUT_SECONDS = int(os.getenv("QUERY_TIMEOUT_SECONDS", "30")) 
MAX_FILE_MB = int(os.getenv("MAX_FILE_MB", "50"))                      