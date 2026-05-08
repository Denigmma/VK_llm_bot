import json
from typing import Any

from app.bot.commands import ensure_active_chat, handle_command, is_chat_setup_pending, is_command, process_setup_input
from app.bot.models_catalog import get_vision_fallback_model, model_supports_images
from app.config import get_settings
from app.openrouter.client import OpenRouterClient
from app.openrouter.errors import OpenRouterError
from app.storage import repositories
from app.storage.db import session_scope
from app.storage.models import Chat, Message
from app.utils.logger import get_logger
from app.utils.text import normalize_message_text


logger = get_logger(__name__)

INTERNAL_ERROR_TEXT = "Произошла внутренняя ошибка сервера. Попробуйте позже."


def handle_user_message(
    vk_user_id: int,
    text: str,
    *,
    image_urls: list[str] | None = None,
    attachments: list[dict] | None = None,
) -> str:
    text = normalize_message_text(text)
    image_urls = image_urls or []
    attachments = attachments or []
    has_supported_attachments = bool(attachments)

    if not text and not has_supported_attachments:
        return "Пожалуйста, отправьте текстовое сообщение."

    try:
        with session_scope() as db:
            user = repositories.get_or_create_user(db, vk_user_id)
            if user.is_banned:
                return "Доступ к боту ограничен."

            if not has_supported_attachments and is_command(text):
                return handle_command(db, vk_user_id, text)

            active_chat = repositories.get_active_chat(db, vk_user_id)
            if active_chat is not None and is_chat_setup_pending(active_chat):
                if has_supported_attachments:
                    return "Сначала завершите настройку нового чата текстом."
                return process_setup_input(db, vk_user_id, text)

            chat = ensure_active_chat(db, vk_user_id)
            return generate_chat_reply(
                db,
                chat,
                text,
                image_urls=image_urls or [],
                attachments=attachments,
                vk_user_id=vk_user_id,
            )
    except Exception:
        logger.exception("Unexpected bot error")
        return INTERNAL_ERROR_TEXT


def generate_chat_reply(
    db,
    chat: Chat,
    text: str,
    *,
    image_urls: list[str] | None = None,
    attachments: list[dict] | None = None,
    vk_user_id: int | None = None,
) -> str:
    image_urls = image_urls or []
    attachments = attachments or []
    has_images = bool(image_urls)
    has_pdf = _count_pdf_attachments(attachments) > 0

    repositories.save_message(
        db,
        chat.id,
        "user",
        text,
        image_url=image_urls[0] if image_urls else None,
        attachments=attachments,
    )
    recent_messages = repositories.get_last_context_messages(db, chat.id, chat.max_context_messages)
    openrouter_messages = _build_openrouter_messages(chat, recent_messages)
    settings = get_settings()

    request_model = chat.model
    response_prefix = ""
    if has_images:
        request_model, response_prefix = _resolve_model_for_images(chat.model, chat.api_profile)
        logger.info(
            "Processing multimodal request: user=%s chat_id=%s base_model=%s request_model=%s images=%s pdfs=%s text_len=%s pdf_engine=%s",
            vk_user_id,
            chat.id,
            chat.model,
            request_model,
            len(image_urls),
            _count_pdf_attachments(attachments),
            len(text),
            chat.pdf_parser_engine,
        )
    else:
        logger.info(
            "Processing request: user=%s chat_id=%s model=%s images=%s pdfs=%s text_len=%s pdf_engine=%s",
            vk_user_id,
            chat.id,
            request_model,
            len(image_urls),
            _count_pdf_attachments(attachments),
            len(text),
            chat.pdf_parser_engine,
        )

    try:
        api_key = settings.get_openrouter_api_key(chat.api_profile)
        result, pdf_retry_note = _request_openrouter_with_pdf_fallback(
            api_key=api_key,
            chat=chat,
            model=request_model,
            messages=openrouter_messages,
            has_pdf=has_pdf,
            vk_user_id=vk_user_id,
        )
    except ValueError as exc:
        logger.warning("OpenRouter profile config is incomplete: %s", exc)
        return str(exc)
    except OpenRouterError as exc:
        logger.warning(
            "OpenRouter request failed: user=%s chat_id=%s model=%s images=%s pdfs=%s error=%s",
            vk_user_id,
            chat.id,
            request_model,
            len(image_urls),
            _count_pdf_attachments(attachments),
            exc,
        )
        return _format_openrouter_error(
            exc,
            reasoning_enabled=chat.reasoning_enabled,
            has_images=has_images,
            has_pdf=has_pdf,
            pdf_parser_engine=chat.pdf_parser_engine,
        )

    assistant_text = result.content
    if response_prefix:
        assistant_text = response_prefix + assistant_text
    if pdf_retry_note:
        assistant_text = pdf_retry_note + assistant_text

    repositories.save_message(
        db,
        chat.id,
        "assistant",
        assistant_text,
        annotations=result.annotations,
    )
    return assistant_text


def extract_message_payload(message: Message) -> tuple[str, list[str], list[dict[str, Any]]]:
    attachments = _load_raw_attachments(message)
    image_urls: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        image_url = attachment.get("image_url")
        if isinstance(image_url, str) and image_url:
            image_urls.append(image_url)

    if not image_urls and message.image_url:
        image_urls = [message.image_url]
        attachments = [{"type": "photo", "image_url": message.image_url}]

    return message.content, image_urls, attachments


def _build_openrouter_messages(chat: Chat, messages: list[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if chat.system_prompt.strip():
        result.append({"role": "system", "content": chat.system_prompt.strip()})

    for index, message in enumerate(messages):
        next_message = messages[index + 1] if index + 1 < len(messages) else None
        include_pdf_parts = _should_include_pdf_parts(message, next_message)
        payload: dict[str, Any] = {
            "role": message.role,
            "content": _build_message_content(message, include_pdf_parts=include_pdf_parts),
        }
        annotations = _load_raw_annotations(message)
        if message.role == "assistant" and annotations:
            payload["annotations"] = annotations
        result.append(payload)
    return result


def _build_message_content(message: Message, *, include_pdf_parts: bool) -> str | list[dict[str, Any]]:
    attachment_parts = _load_attachment_parts(message, include_pdf_parts=include_pdf_parts)
    if not attachment_parts:
        if message.content.strip():
            return message.content
        if _has_pdf_attachment(message):
            return _build_pdf_placeholder(message)
        return message.content

    parts: list[dict[str, Any]] = []
    if message.content.strip():
        parts.append(
            {
                "type": "text",
                "text": message.content,
            }
        )
    parts.extend(attachment_parts)
    return parts


def _load_attachment_parts(message: Message, *, include_pdf_parts: bool) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for attachment in _load_raw_attachments(message):
        if not isinstance(attachment, dict):
            continue

        attachment_type = attachment.get("type")
        if attachment_type == "photo":
            image_url = attachment.get("image_url")
            if isinstance(image_url, str) and image_url:
                parts.append(_build_image_part(image_url))
            continue

        if attachment_type != "pdf" or not include_pdf_parts:
            continue

        filename = attachment.get("filename")
        file_data = attachment.get("file_data")
        if isinstance(filename, str) and filename and isinstance(file_data, str) and file_data:
            parts.append(_build_pdf_part(filename, file_data))

    if not parts and message.image_url:
        parts.append(_build_image_part(message.image_url))
    return parts


def _load_raw_attachments(message: Message) -> list[dict[str, Any]]:
    if not message.attachments_json:
        return []

    try:
        attachments = json.loads(message.attachments_json)
    except json.JSONDecodeError:
        logger.warning("Failed to decode attachments_json for message %s", message.id)
        return []

    if isinstance(attachments, list):
        return [item for item in attachments if isinstance(item, dict)]
    return []


def _load_raw_annotations(message: Message) -> list[dict[str, Any]]:
    if not message.annotations_json:
        return []

    try:
        annotations = json.loads(message.annotations_json)
    except json.JSONDecodeError:
        logger.warning("Failed to decode annotations_json for message %s", message.id)
        return []

    if isinstance(annotations, list):
        return [item for item in annotations if isinstance(item, dict)]
    return []


def _build_image_part(image_url: str) -> dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
        },
    }


def _build_pdf_part(filename: str, file_data: str) -> dict[str, Any]:
    return {
        "type": "file",
        "file": {
            "filename": filename,
            "file_data": file_data,
        },
    }


def _resolve_model_for_images(model: str, profile: str) -> tuple[str, str]:
    supports_images = model_supports_images(model)
    if supports_images is not False:
        return model, ""

    fallback_model = get_vision_fallback_model(profile)
    if fallback_model == model:
        return model, ""

    logger.warning(
        "Selected model likely does not support images, using vision fallback: original_model=%s fallback_model=%s profile=%s",
        model,
        fallback_model,
        profile,
    )
    prefix = f"ℹ️ Для обработки изображения временно использована vision-модель: {fallback_model}.\n\n"
    return fallback_model, prefix


def _request_openrouter_with_pdf_fallback(
    *,
    api_key: str,
    chat: Chat,
    model: str,
    messages: list[dict[str, Any]],
    has_pdf: bool,
    vk_user_id: int | None,
):
    client = OpenRouterClient()
    primary_engine = chat.pdf_parser_engine if has_pdf else None

    try:
        result = client.chat_completion(
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=chat.temperature,
            reasoning_enabled=chat.reasoning_enabled,
            reasoning_effort=chat.reasoning_effort,
            pdf_parser_engine=primary_engine,
        )
        return result, ""
    except OpenRouterError as exc:
        if not _should_retry_pdf_with_native(exc, has_pdf=has_pdf, current_engine=primary_engine):
            raise

        logger.warning(
            "Retrying PDF request with native parser: user=%s chat_id=%s model=%s failed_engine=%s error=%s",
            vk_user_id,
            chat.id,
            model,
            primary_engine,
            exc,
        )
        try:
            result = client.chat_completion(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=chat.temperature,
                reasoning_enabled=chat.reasoning_enabled,
                reasoning_effort=chat.reasoning_effort,
                pdf_parser_engine="native",
            )
        except OpenRouterError as retry_exc:
            logger.warning(
                "Native PDF retry also failed: user=%s chat_id=%s model=%s original_engine=%s retry_error=%s",
                vk_user_id,
                chat.id,
                model,
                primary_engine,
                retry_exc,
            )
            raise OpenRouterError(f"{exc} | Повтор через native тоже завершился ошибкой: {retry_exc}") from retry_exc

        note = (
            f"ℹ️ PDF не удалось распарсить через `{primary_engine}`, поэтому бот автоматически повторил запрос через `native`. "
            "Если такие документы у вас встречаются часто, можно закрепить это командой /pdfengine native.\n\n"
        )
        return result, note


def _format_openrouter_error(
    error: OpenRouterError,
    *,
    reasoning_enabled: bool,
    has_images: bool,
    has_pdf: bool,
    pdf_parser_engine: str,
) -> str:
    hint = ""
    if reasoning_enabled:
        hint = (
            "\n\nПодсказка: сейчас reasoning включен. Если модель его не поддерживает, "
            "попробуйте /reasoning off."
        )
    if has_images:
        hint += (
            "\n\nПодсказка по картинкам: не все модели поддерживают изображения. "
            "Для фото лучше использовать vision-модели, например openai/gpt-5.4, openai/gpt-5-nano, "
            "google/gemini-2.0-flash-exp:free или nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free."
        )
    if has_pdf:
        hint += (
            "\n\nПодсказка по PDF: сейчас используется движок "
            f"`{pdf_parser_engine}`. Попробуйте /pdfengine cloudflare-ai, /pdfengine native или /pdfengine mistral-ocr."
        )

    return (
        "Ошибка при запросе к OpenRouter.\n\n"
        "Возможные причины:\n"
        "— модель не существует;\n"
        "— модель временно недоступна;\n"
        "— модель не поддерживает reasoning;\n"
        "— модель не поддерживает изображения;\n"
        "— PDF parser engine не подходит для этой модели или документа;\n"
        "— закончились лимиты;\n"
        "— неправильный API-ключ."
        f"{hint}\n\n"
        "Техническая информация:\n"
        f"{error}"
    )


def _should_include_pdf_parts(message: Message, next_message: Message | None) -> bool:
    if message.role != "user" or not _has_pdf_attachment(message):
        return False
    if next_message is None:
        return True
    if next_message.role != "assistant":
        return True
    return not bool(_load_raw_annotations(next_message))


def _has_pdf_attachment(message: Message) -> bool:
    return any(attachment.get("type") == "pdf" for attachment in _load_raw_attachments(message))


def _count_pdf_attachments(attachments: list[dict[str, Any]]) -> int:
    return sum(1 for attachment in attachments if attachment.get("type") == "pdf")


def _should_retry_pdf_with_native(error: OpenRouterError, *, has_pdf: bool, current_engine: str | None) -> bool:
    if not has_pdf or current_engine in {None, "native"}:
        return False
    message = str(error).lower()
    return "failed to parse" in message or "parse" in message


def _build_pdf_placeholder(message: Message) -> str:
    for attachment in _load_raw_attachments(message):
        if attachment.get("type") != "pdf":
            continue
        filename = attachment.get("filename")
        if isinstance(filename, str) and filename:
            return f"Пользователь прикрепил PDF-документ: {filename}."
    return "Пользователь прикрепил PDF-документ."
