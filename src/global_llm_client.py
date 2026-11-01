import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _required_api_key(name: str) -> SecretStr:
    api_key = os.getenv(name)
    if not api_key:
        raise ValueError(f"{name} must be set in .env to use this provider.")
    return SecretStr(api_key)


def create_llm() -> BaseChatModel:
    provider = os.getenv("USE_LLM", "groq").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=_required_api_key("GROQ_API_KEY"),
            temperature=0.7,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=_required_api_key("OPENAI_API_KEY"),
            temperature=0.7,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=_required_api_key("GEMINI_API_KEY"),
            temperature=0.7,
        )

    raise ValueError("USE_LLM must be one of: groq, openai, gemini.")


llm = create_llm()