from typing import Optional


DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
DEFAULT_API_KEY = "tp-c1xf439dt3j37rc51q3sr011s563jbdohy4ooan98kz3z9a1"
DEFAULT_MODEL_ID = "mimo-v2.5-pro"


class LLMClient:
    """Thin wrapper for calling Gemini through an OpenAI-compatible API."""

    def __init__(self, model: str, api_key: str, base_url: str):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Missing openai package. Install it with: pip install openai") from exc

        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return response.choices[0].message.content or ""


def build_llm_client() -> LLMClient:
    """Build the shared LLM client from explicit in-code configuration."""
    api_key = DEFAULT_API_KEY
    base_url = DEFAULT_BASE_URL
    model = DEFAULT_MODEL_ID

    if not api_key or api_key == "YOUR_API_KEY_HERE":
        raise ValueError("Please set DEFAULT_API_KEY, DEFAULT_BASE_URL, and DEFAULT_MODEL_ID in src/llm_client.py.")

    return LLMClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


def call_llm(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1200,
    client: Optional[LLMClient] = None,
) -> str:
    """Shared helper for the rest of the project."""
    llm_client = client or build_llm_client()
    return llm_client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )
