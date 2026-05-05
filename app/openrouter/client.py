from typing import Any
import json

import httpx

from app.config import Settings, get_settings
from app.openrouter.errors import OpenRouterError
from app.utils.logger import get_logger


logger = get_logger(__name__)


class OpenRouterClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def chat_completion(
        self,
        *,
        api_key: str,
        model: str,
        messages: list[dict[str, Any]],
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

        logger.info(
            "OpenRouter request: model=%s messages=%s image_parts=%s reasoning=%s effort=%s",
            model,
            len(messages),
            _count_image_parts(messages),
            reasoning_enabled,
            reasoning_effort,
        )

        try:
            with httpx.Client(timeout=90.0) as client:
                response = client.post(
                    self.settings.openrouter_chat_completions_url,
                    headers=headers,
                    json=payload,
                )
        except httpx.RequestError as exc:
            raise OpenRouterError(f"Не удалось выполнить HTTP-запрос: {exc}") from exc

        logger.info(
            "OpenRouter response: status=%s request_id=%s model=%s",
            response.status_code,
            response.headers.get("x-request-id", "n/a"),
            model,
        )

        if response.status_code >= 400:
            raise OpenRouterError(_extract_error_text(response))

        try:
            data = response.json()
        except ValueError as exc:
            logger.warning("OpenRouter returned invalid JSON: body=%s", _body_snippet(response.text))
            raise OpenRouterError("OpenRouter вернул невалидный JSON") from exc

        if _looks_like_error_payload(data):
            logger.warning(
                "OpenRouter returned logical error with HTTP 200: model=%s body=%s",
                model,
                _json_snippet(data),
            )
            raise OpenRouterError(_extract_error_text_from_data(data))

        return _extract_assistant_content(data, model=model)


def _count_image_parts(messages: list[dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        total += sum(1 for part in content if isinstance(part, dict) and part.get("type") == "image_url")
    return total


def _extract_error_text(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        body = response.text.strip()
        return f"HTTP {response.status_code}: {body or 'пустое тело ответа'}"

    return _extract_error_text_from_data(data, status_code=response.status_code)


def _extract_error_text_from_data(data: Any, status_code: int | None = None) -> str:
    prefix = f"HTTP {status_code}: " if status_code is not None else ""

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("metadata") or error.get("code")
            if isinstance(message, dict):
                message = json.dumps(message, ensure_ascii=False)
            if message:
                return f"{prefix}{message}"
        if isinstance(error, str):
            return f"{prefix}{error}"
        message = data.get("message")
        if isinstance(message, str):
            return f"{prefix}{message}"
        detail = data.get("detail")
        if isinstance(detail, str):
            return f"{prefix}{detail}"

    suffix = _json_snippet(data)
    return f"{prefix}неожиданный формат ответа OpenRouter: {suffix}"


def _looks_like_error_payload(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    if data.get("error"):
        return True
    if data.get("object") == "error":
        return True
    return False


def _extract_assistant_content(data: Any, *, model: str) -> str:
    if not isinstance(data, dict):
        raise OpenRouterError("OpenRouter вернул ответ не в формате JSON-объекта")

    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        logger.warning(
            "OpenRouter response without choices: model=%s keys=%s body=%s",
            model,
            sorted(data.keys()),
            _json_snippet(data),
        )
        raise OpenRouterError("OpenRouter вернул неожиданный ответ без choices. Проверьте поддержку изображений у выбранной модели.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise OpenRouterError("choices[0] в ответе OpenRouter имеет неожиданный формат")

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = first_choice.get("text")

    content = _normalize_content(content)
    if not isinstance(content, str):
        logger.warning(
            "OpenRouter response has unsupported content format: model=%s choice=%s",
            model,
            _json_snippet(first_choice),
        )
        raise OpenRouterError("В ответе OpenRouter нет поддерживаемого текстового content")

    if not content.strip():
        raise OpenRouterError("OpenRouter вернул пустой ответ ассистента")

    return content.strip()


def _normalize_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None

    text_parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = part.get("type")
        if part_type in {"text", "output_text"} and isinstance(part.get("text"), str):
            text_parts.append(part["text"].strip())
            continue
        if part_type == "message" and isinstance(part.get("content"), str):
            text_parts.append(part["content"].strip())

    normalized = "\n".join(part for part in text_parts if part)
    return normalized or None


def _json_snippet(data: Any) -> str:
    try:
        raw = json.dumps(data, ensure_ascii=False, default=str)
    except TypeError:
        raw = str(data)
    return _body_snippet(raw)


def _body_snippet(body: str, limit: int = 1200) -> str:
    compact = " ".join(body.split())
    if len(compact) <= limit:
        return compact
    return compact[:limit] + "..."
