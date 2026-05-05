from app.bot.commands import ensure_active_chat, handle_command, is_chat_setup_pending, is_command, process_setup_input
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


def handle_user_message(vk_user_id: int, text: str) -> str:
    text = normalize_message_text(text)
    if not text:
        return "Пожалуйста, отправьте текстовое сообщение."

    try:
        with session_scope() as db:
            user = repositories.get_or_create_user(db, vk_user_id)
            if user.is_banned:
                return "Доступ к боту ограничен."

            if is_command(text):
                return handle_command(db, vk_user_id, text)

            active_chat = repositories.get_active_chat(db, vk_user_id)
            if active_chat is not None and is_chat_setup_pending(active_chat):
                return process_setup_input(db, vk_user_id, text)

            chat = ensure_active_chat(db, vk_user_id)
            repositories.save_message(db, chat.id, "user", text)
            recent_messages = repositories.get_last_messages(db, chat.id, chat.max_context_messages)
            openrouter_messages = _build_openrouter_messages(chat, recent_messages)
            settings = get_settings()

            try:
                api_key = settings.get_openrouter_api_key(chat.api_profile)
                assistant_text = OpenRouterClient().chat_completion(
                    api_key=api_key,
                    model=chat.model,
                    messages=openrouter_messages,
                    temperature=chat.temperature,
                    reasoning_enabled=chat.reasoning_enabled,
                    reasoning_effort=chat.reasoning_effort,
                )
            except ValueError as exc:
                logger.warning("OpenRouter profile config is incomplete: %s", exc)
                return str(exc)
            except OpenRouterError as exc:
                logger.warning("OpenRouter request failed: %s", exc)
                return _format_openrouter_error(exc, reasoning_enabled=chat.reasoning_enabled)

            repositories.save_message(db, chat.id, "assistant", assistant_text)
            return assistant_text
    except Exception:
        logger.exception("Unexpected bot error")
        return INTERNAL_ERROR_TEXT


def _build_openrouter_messages(chat: Chat, messages: list[Message]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if chat.system_prompt.strip():
        result.append({"role": "system", "content": chat.system_prompt.strip()})

    for message in messages:
        result.append({"role": message.role, "content": message.content})
    return result


def _format_openrouter_error(error: OpenRouterError, *, reasoning_enabled: bool) -> str:
    hint = ""
    if reasoning_enabled:
        hint = (
            "\n\nПодсказка: сейчас reasoning включен. Если модель его не поддерживает, "
            "попробуйте /reasoning off."
        )

    return (
        "Ошибка при запросе к OpenRouter.\n\n"
        "Возможные причины:\n"
        "— модель не существует;\n"
        "— модель временно недоступна;\n"
        "— модель не поддерживает reasoning;\n"
        "— закончились лимиты;\n"
        "— неправильный API-ключ."
        f"{hint}\n\n"
        "Техническая информация:\n"
        f"{error}"
    )
