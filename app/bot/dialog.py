import json

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
    has_images = bool(image_urls)

    if not text and not has_images:
        return "Пожалуйста, отправьте текстовое сообщение."

    try:
        with session_scope() as db:
            user = repositories.get_or_create_user(db, vk_user_id)
            if user.is_banned:
                return "Доступ к боту ограничен."

            if not has_images and is_command(text):
                return handle_command(db, vk_user_id, text)

            active_chat = repositories.get_active_chat(db, vk_user_id)
            if active_chat is not None and is_chat_setup_pending(active_chat):
                if has_images:
                    return "Сначала завершите настройку нового чата текстом."
                return process_setup_input(db, vk_user_id, text)

            chat = ensure_active_chat(db, vk_user_id)
            repositories.save_message(
                db,
                chat.id,
                "user",
                text,
                image_url=image_urls[0] if image_urls else None,
                attachments=attachments,
            )
            recent_messages = repositories.get_last_messages(db, chat.id, chat.max_context_messages)
            openrouter_messages = _build_openrouter_messages(chat, recent_messages)
            settings = get_settings()

            request_model = chat.model
            response_prefix = ""
            if has_images:
                request_model, response_prefix = _resolve_model_for_images(chat.model, chat.api_profile)
                logger.info(
                    "Processing multimodal request: user=%s chat_id=%s base_model=%s request_model=%s images=%s text_len=%s",
                    vk_user_id,
                    chat.id,
                    chat.model,
                    request_model,
                    len(image_urls),
                    len(text),
                )
            else:
                logger.info(
                    "Processing text request: user=%s chat_id=%s model=%s text_len=%s",
                    vk_user_id,
                    chat.id,
                    request_model,
                    len(text),
                )

            try:
                api_key = settings.get_openrouter_api_key(chat.api_profile)
                assistant_text = OpenRouterClient().chat_completion(
                    api_key=api_key,
                    model=request_model,
                    messages=openrouter_messages,
                    temperature=chat.temperature,
                    reasoning_enabled=chat.reasoning_enabled,
                    reasoning_effort=chat.reasoning_effort,
                )
            except ValueError as exc:
                logger.warning("OpenRouter profile config is incomplete: %s", exc)
                return str(exc)
            except OpenRouterError as exc:
                logger.warning(
                    "OpenRouter request failed: user=%s chat_id=%s model=%s images=%s error=%s",
                    vk_user_id,
                    chat.id,
                    request_model,
                    len(image_urls),
                    exc,
                )
                return _format_openrouter_error(exc, reasoning_enabled=chat.reasoning_enabled, has_images=has_images)

            if response_prefix:
                assistant_text = response_prefix + assistant_text

            repositories.save_message(db, chat.id, "assistant", assistant_text)
            return assistant_text
    except Exception:
        logger.exception("Unexpected bot error")
        return INTERNAL_ERROR_TEXT


def _build_openrouter_messages(chat: Chat, messages: list[Message]) -> list[dict]:
    result: list[dict] = []
    if chat.system_prompt.strip():
        result.append({"role": "system", "content": chat.system_prompt.strip()})

    for message in messages:
        result.append({"role": message.role, "content": _build_message_content(message)})
    return result


def _build_message_content(message: Message) -> str | list[dict]:
    attachment_parts = _load_attachment_parts(message)
    if not attachment_parts:
        return message.content

    parts: list[dict] = []
    if message.content.strip():
        parts.append(
            {
                "type": "text",
                "text": message.content,
            }
        )
    parts.extend(attachment_parts)
    return parts


def _load_attachment_parts(message: Message) -> list[dict]:
    if not message.attachments_json:
        if message.image_url:
            return [_build_image_part(message.image_url)]
        return []

    try:
        attachments = json.loads(message.attachments_json)
    except json.JSONDecodeError:
        logger.warning("Failed to decode attachments_json for message %s", message.id)
        return [_build_image_part(message.image_url)] if message.image_url else []

    parts: list[dict] = []
    if isinstance(attachments, list):
        for attachment in attachments:
            if not isinstance(attachment, dict):
                continue
            if attachment.get("type") != "photo":
                continue
            image_url = attachment.get("image_url")
            if isinstance(image_url, str) and image_url:
                parts.append(_build_image_part(image_url))

    if not parts and message.image_url:
        parts.append(_build_image_part(message.image_url))
    return parts


def _build_image_part(image_url: str) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": image_url,
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


def _format_openrouter_error(error: OpenRouterError, *, reasoning_enabled: bool, has_images: bool) -> str:
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

    return (
        "Ошибка при запросе к OpenRouter.\n\n"
        "Возможные причины:\n"
        "— модель не существует;\n"
        "— модель временно недоступна;\n"
        "— модель не поддерживает reasoning;\n"
        "— модель не поддерживает изображения;\n"
        "— закончились лимиты;\n"
        "— неправильный API-ключ."
        f"{hint}\n\n"
        "Техническая информация:\n"
        f"{error}"
    )
