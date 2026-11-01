import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _required_api_key(name: str) -> SecretStr:
    api_key = os.getenv(name)
    if not api_key:
        raise ValueError(f"{name} must be set in .env to use this provider.")
    return SecretStr(api_key)


def _client_options(
    timeout: float | None,
    max_retries: int | None,
    max_output_tokens: int | None,
    *,
    timeout_parameter: str = "timeout",
    token_limit_parameter: str = "max_tokens",
) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if timeout is not None:
        options[timeout_parameter] = timeout
    if max_retries is not None:
        options["max_retries"] = max_retries
    if max_output_tokens is not None:
        options[token_limit_parameter] = max_output_tokens
    return options


def create_llm(
    *,
    temperature: float = 0.5,
    max_output_tokens: int | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> BaseChatModel:
    provider = os.getenv("USE_LLM", "groq").strip().lower()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
            api_key=_required_api_key("GROQ_API_KEY"),
            temperature=temperature,
            **_client_options(timeout, max_retries, max_output_tokens),
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            api_key=_required_api_key("OPENAI_API_KEY"),
            temperature=temperature,
            **_client_options(
                timeout,
                max_retries,
                max_output_tokens,
                token_limit_parameter="max_completion_tokens",
            ),
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            google_api_key=_required_api_key("GEMINI_API_KEY"),
            temperature=temperature,
            **_client_options(
                timeout,
                max_retries,
                max_output_tokens,
                timeout_parameter="request_timeout",
            ),
        )

    raise ValueError("USE_LLM must be one of: groq, openai, gemini.")


llm = create_llm()