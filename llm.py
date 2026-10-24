# pyrefly: ignore[missing-import]
# from langchain_openrouter import ChatOpenRouter
# import config


# def get_llm() -> ChatOpenRouter:
#     if not config.OPENROUTER_API_KEY:
#         raise RuntimeError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
#     return ChatOpenRouter(
#         model=config.MODEL_NAME,
#         api_key=config.OPENROUTER_API_KEY,
#         temperature=0,  # SQL generation should be as deterministic as possible
#     )

# pyrefly: ignore[missing-import]
from langchain_groq import ChatGroq
import config


def get_llm() -> ChatGroq:
  if not config.GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")

  return ChatGroq(
      model=config.MODEL_NAME,  # e.g., "llama-3.3-70b-versatile"
      api_key=config.GROQ_API_KEY,
      temperature=0,
  )


