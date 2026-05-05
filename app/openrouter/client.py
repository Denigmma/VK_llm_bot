from typing import Any

import httpx

from app.config import Settings, get_settings
from app.openrouter.errors import OpenRouterError


class OpenRouterClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def chat_completion(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        reasoning_enabled: bool,
        reasoning_effort: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if reasoning_enabled:
            payload["reasoning"] = {
                "effort": reasoning_effort,
                "exclude": True,
            }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    self.settings.openrouter_chat_completions_url,
                    headers=headers,
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise OpenRouterError(f"Не удалось выполнить HTTP-запрос: {exc}") from exc

        if response.status_code >= 400:
            raise OpenRouterError(_extract_error_text(response))

        try:
            data = response.json()
        except ValueError as exc:
            raise OpenRouterError("OpenRouter вернул невалидный JSON") from exc

        return _extract_assistant_content(data)


def _extract_error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        body = response.text.strip()
        return f"HTTP {response.status_code}: {body or 'пустое тело ответа'}"

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code")
            if message:
                return f"HTTP {response.status_code}: {message}"
        if isinstance(error, str):
            return f"HTTP {response.status_code}: {error}"
        message = data.get("message")
        if isinstance(message, str):
            return f"HTTP {response.status_code}: {message}"

    return f"HTTP {response.status_code}: неожиданный формат ошибки OpenRouter"


def _extract_assistant_content(data: Any) -> str:
    if not isinstance(data, dict):
        raise OpenRouterError("OpenRouter вернул ответ не в формате JSON-объекта")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise OpenRouterError("В ответе OpenRouter нет choices[0]")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenRouterError("choices[0] в ответе OpenRouter имеет неожиданный формат")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise OpenRouterError("В ответе OpenRouter нет choices[0].message")

    content = message.get("content")
    if not isinstance(content, str):
        raise OpenRouterError("В ответе OpenRouter нет choices[0].message.content")

    if not content.strip():
        raise OpenRouterError("OpenRouter вернул пустой ответ ассистента")

    return content.strip()
