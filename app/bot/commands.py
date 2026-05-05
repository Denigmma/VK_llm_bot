from sqlalchemy.orm import Session

from app.bot.models_catalog import format_models_catalog, get_model_by_number
from app.bot.prompts import DEFAULT_SYSTEM_PROMPT
from app.config import OpenRouterProfile, get_settings
from app.storage import repositories
from app.storage.models import Chat


SETUP_STAGE_READY = "ready"
SETUP_STAGE_TITLE = "awaiting_title"
SETUP_STAGE_MODEL = "awaiting_model"
SETUP_STAGE_REASONING = "awaiting_reasoning"
SETUP_STAGE_REASONING_EFFORT = "awaiting_reasoning_effort"


HELP_TEXT = """✨ Команды бота

💬 Чаты:
/new — создать новый чат и пройти быструю настройку.
/chats — показать последние 10 чатов.
/chat 1 — переключиться на чат по номеру из списка /chats.
/deletechat 1 — удалить чат по номеру из списка /chats.
/reset — очистить сообщения текущего чата.

⚙️ Настройки:
/settings — показать настройки текущего чата.
/models — показать список предустановленных моделей.
/model — показать текущую модель.
/model 1 — выбрать модель из списка /models.
/model model_id — сохранить модель вручную и автоматически переключить free/pay ключ по модели.
/key — показать текущий профиль OpenRouter.
/key free — переключиться на free ключ и free модель.
/key pay — переключиться на pay ключ и pay модель.
/system — показать системный промпт.
/system текст — установить новый системный промпт.

🧠 Другое:
/reasoning — показать настройки reasoning.
/reasoning off — отключить reasoning.
/reasoning on — включить reasoning с текущим effort или medium.
/reasoning low|medium|high — включить reasoning с выбранным effort.
/skip — пропустить текущий шаг настройки нового чата.
/help — показать эту справку."""


def is_command(text: str) -> bool:
    return text.strip().startswith("/")


def handle_command(db: Session, vk_user_id: int, text: str) -> str:
    command, argument = _parse_command(text)

    if command == "/help":
        return HELP_TEXT
    if command == "/new":
        return _handle_new_chat(db, vk_user_id)
    if command == "/reset":
        chat = ensure_active_chat(db, vk_user_id)
        repositories.clear_chat_messages(db, chat.id)
        return f"🧹 История чата «{chat.title}» очищена."
    if command == "/chats":
        return _handle_chats(db, vk_user_id)
    if command == "/chat":
        return _handle_chat_switch(db, vk_user_id, argument)
    if command in {"/deletechat", "/delchat"}:
        return _handle_delete_chat(db, vk_user_id, argument)
    if command == "/settings":
        return _handle_settings(db, vk_user_id)
    if command == "/models":
        return format_models_catalog()
    if command == "/model":
        return _handle_model(db, vk_user_id, argument)
    if command == "/skip":
        return process_setup_input(db, vk_user_id, "/skip")
    if command == "/key":
        return _handle_key(db, vk_user_id, argument)
    if command == "/reasoning":
        return _handle_reasoning(db, vk_user_id, argument)
    if command == "/system":
        return _handle_system(db, vk_user_id, argument)

    return "Неизвестная команда. Напишите /help, чтобы увидеть список команд."


def ensure_active_chat(db: Session, vk_user_id: int) -> Chat:
    chat = repositories.get_active_chat(db, vk_user_id)
    if chat is not None:
        return chat
    return create_chat_with_defaults(db, vk_user_id)


def create_chat_with_defaults(db: Session, vk_user_id: int) -> Chat:
    settings = get_settings()
    chat_number = repositories.count_user_chats(db, vk_user_id) + 1
    default_model = settings.get_openrouter_model("free")
    return repositories.create_new_chat(
        db,
        vk_user_id,
        title=f"Чат #{chat_number}",
        api_profile="free",
        setup_stage=SETUP_STAGE_READY,
        model=default_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning_enabled=settings.default_reasoning_enabled,
        reasoning_effort=settings.default_reasoning_effort,
        temperature=settings.default_temperature,
        max_context_messages=settings.default_max_context_messages,
        is_active=True,
    )


def create_chat_for_setup(db: Session, vk_user_id: int) -> Chat:
    settings = get_settings()
    chat_number = repositories.count_user_chats(db, vk_user_id) + 1
    default_model = settings.get_openrouter_model("free")
    return repositories.create_new_chat(
        db,
        vk_user_id,
        title=f"Новый чат #{chat_number}",
        api_profile="free",
        setup_stage=SETUP_STAGE_TITLE,
        model=default_model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        reasoning_enabled=settings.default_reasoning_enabled,
        reasoning_effort=settings.default_reasoning_effort,
        temperature=settings.default_temperature,
        max_context_messages=settings.default_max_context_messages,
        is_active=True,
    )


def _parse_command(text: str) -> tuple[str, str]:
    stripped = text.strip()
    command, _, argument = stripped.partition(" ")
    command = command.split("@", maxsplit=1)[0].lower()
    return command, argument.strip()


def _handle_chats(db: Session, vk_user_id: int) -> str:
    chats = repositories.list_user_chats(db, vk_user_id, limit=10)
    if not chats:
        return "💬 У вас пока нет чатов. Напишите /new или отправьте обычное сообщение."

    lines = ["💬 Последние чаты:"]
    for index, chat in enumerate(chats, start=1):
        badges: list[str] = []
        if chat.is_active:
            badges.append("active")
        if is_chat_setup_pending(chat):
            badges.append("setup")
        suffix = f" [{' | '.join(badges)}]" if badges else ""
        lines.append(f"{index}. {chat.title}{suffix}")
    return "\n".join(lines)


def _handle_chat_switch(db: Session, vk_user_id: int, argument: str) -> str:
    if not argument:
        return "Укажите номер чата из списка /chats. Например: /chat 1"

    try:
        chat_number = int(argument)
    except ValueError:
        return "Номер чата должен быть числом. Например: /chat 1"

    chats = repositories.list_user_chats(db, vk_user_id, limit=10)
    if not 1 <= chat_number <= len(chats):
        return "Чат с таким номером не найден в последних 10 чатах. Проверьте список через /chats."

    chat = repositories.make_chat_active(db, vk_user_id, chats[chat_number - 1].id)
    if chat is None:
        return "Не удалось переключиться на выбранный чат."
    if is_chat_setup_pending(chat):
        return f"💬 Активный чат: {chat.title}\n\n{get_setup_prompt(chat)}"
    return f"💬 Активный чат: {chat.title}"


def _handle_delete_chat(db: Session, vk_user_id: int, argument: str) -> str:
    if not argument:
        return "Укажите номер чата из списка /chats. Например: /deletechat 2"

    try:
        chat_number = int(argument)
    except ValueError:
        return "Номер чата должен быть числом. Например: /deletechat 2"

    chats = repositories.list_user_chats(db, vk_user_id, limit=10)
    if not 1 <= chat_number <= len(chats):
        return "Чат с таким номером не найден в последних 10 чатах. Сначала посмотрите /chats."

    chat_to_delete = chats[chat_number - 1]
    deleted_title = chat_to_delete.title
    next_chat = repositories.delete_chat(db, chat_to_delete)

    if next_chat is None:
        return f"🗑️ Чат «{deleted_title}» удален. Других чатов пока не осталось."
    return f"🗑️ Чат «{deleted_title}» удален.\n💬 Активный чат: {next_chat.title}"


def _handle_settings(db: Session, vk_user_id: int) -> str:
    chat = ensure_active_chat(db, vk_user_id)
    return (
        "⚙️ Настройки текущего чата:\n"
        f"Название: {chat.title}\n"
        f"Профиль OpenRouter: {chat.api_profile}\n"
        f"Модель: {chat.model}\n"
        f"Reasoning: {str(chat.reasoning_enabled).lower()}\n"
        f"Уровень reasoning: {chat.reasoning_effort}\n"
        f"Temperature: {chat.temperature}\n"
        f"Контекстных сообщений: {chat.max_context_messages}\n"
        f"System prompt: {chat.system_prompt}"
    )


def _handle_model(db: Session, vk_user_id: int, argument: str) -> str:
    chat = ensure_active_chat(db, vk_user_id)
    if not argument:
        return f"Текущая модель: {chat.model}"

    model = _resolve_model_argument(argument)
    if model is None:
        return "Модель с таким номером не найдена. Посмотрите список через /models."

    profile = _infer_profile_from_model(model)
    repositories.update_chat_api_profile(db, chat, api_profile=profile, model=model)
    return (
        f"⚙️ Модель текущего чата обновлена: {model}\n"
        f"🔑 Профиль OpenRouter автоматически переключен на {profile}."
    )


def _resolve_model_argument(argument: str) -> str | None:
    if argument.isdigit():
        return get_model_by_number(int(argument))
    return argument.strip()


def _handle_reasoning(db: Session, vk_user_id: int, argument: str) -> str:
    chat = ensure_active_chat(db, vk_user_id)
    normalized = argument.lower().strip()

    if not normalized:
        return (
            "🧠 Текущие настройки reasoning:\n"
            f"Reasoning: {str(chat.reasoning_enabled).lower()}\n"
            f"Уровень: {chat.reasoning_effort}"
        )

    if normalized == "off":
        repositories.update_chat_reasoning(db, chat, enabled=False, effort=chat.reasoning_effort)
        return "🧠 Reasoning отключен."

    if normalized == "on":
        effort = chat.reasoning_effort if chat.reasoning_effort in {"low", "medium", "high"} else "medium"
        repositories.update_chat_reasoning(db, chat, enabled=True, effort=effort)
        return f"🧠 Reasoning включен. Уровень: {effort}"

    if normalized in {"low", "medium", "high"}:
        repositories.update_chat_reasoning(db, chat, enabled=True, effort=normalized)
        return f"🧠 Reasoning включен. Уровень: {normalized}"

    return "Используйте /reasoning off, /reasoning on или /reasoning low|medium|high."


def _handle_key(db: Session, vk_user_id: int, argument: str) -> str:
    chat = ensure_active_chat(db, vk_user_id)
    settings = get_settings()
    normalized = argument.lower().strip()

    if not normalized:
        return (
            "🔑 Текущий профиль OpenRouter:\n"
            f"Профиль: {chat.api_profile}\n"
            f"Модель: {chat.model}"
        )

    if normalized not in {"free", "pay"}:
        return "Используйте /key free или /key pay."

    profile = normalized
    if not _is_profile_configured(settings, profile):
        env_name = "FREE_API_KEY_OPENROUTER" if profile == "free" else "API_KEY_OPENROUTER"
        return f"Профиль {profile} не настроен. Заполните переменную {env_name} в .env."

    profile_model = settings.get_openrouter_model(profile)
    repositories.update_chat_api_profile(db, chat, api_profile=profile, model=profile_model)
    return f"🔑 Профиль OpenRouter переключен на {profile}.\n⚙️ Текущая модель: {profile_model}"


def _handle_system(db: Session, vk_user_id: int, argument: str) -> str:
    chat = ensure_active_chat(db, vk_user_id)
    if not argument:
        return f"⚙️ Текущий системный промпт:\n{chat.system_prompt}"

    repositories.update_chat_system_prompt(db, chat, argument)
    return "⚙️ Системный промпт обновлен."


def _is_profile_configured(settings, profile: OpenRouterProfile) -> bool:
    try:
        settings.get_openrouter_api_key(profile)
    except ValueError:
        return False
    return True


def _infer_profile_from_model(model: str) -> OpenRouterProfile:
    normalized = model.strip().lower()
    if ":free" in normalized:
        return "free"
    return "pay"


def process_setup_input(db: Session, vk_user_id: int, text: str) -> str:
    chat = repositories.get_active_chat(db, vk_user_id)
    if chat is None or not is_chat_setup_pending(chat):
        return "Сейчас нет активной настройки нового чата."

    value = text.strip()
    if chat.setup_stage == SETUP_STAGE_TITLE:
        if value != "/skip":
            repositories.update_chat_title(db, chat, value, setup_stage=SETUP_STAGE_MODEL)
        else:
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_MODEL)
        return (
            f"✅ Название чата: {chat.title}\n\n"
            f"{get_setup_prompt(chat)}"
        )

    if chat.setup_stage == SETUP_STAGE_MODEL:
        if value == "/skip":
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_REASONING)
            return get_setup_prompt(chat)

        model = _resolve_model_argument(value)
        if model is None:
            return (
                "⚠️ Не удалось распознать модель.\n"
                "Отправьте номер из /models, полный model_id или /skip."
            )

        profile = _infer_profile_from_model(model)
        repositories.update_chat_api_profile(
            db,
            chat,
            api_profile=profile,
            model=model,
            setup_stage=SETUP_STAGE_REASONING,
        )
        return (
            f"✅ Модель выбрана: {chat.model}\n"
            f"🔑 Профиль автоматически: {chat.api_profile}\n\n"
            f"{get_setup_prompt(chat)}"
        )

    if chat.setup_stage == SETUP_STAGE_REASONING:
        normalized = value.lower()
        if value == "/skip":
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
            return _format_setup_summary(chat)

        if normalized in {"off", "нет", "no"}:
            repositories.update_chat_reasoning(db, chat, enabled=False, effort=chat.reasoning_effort)
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
            return _format_setup_summary(chat)

        if normalized in {"on", "да", "yes"}:
            repositories.update_chat_reasoning(db, chat, enabled=True, effort=chat.reasoning_effort)
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_REASONING_EFFORT)
            return get_setup_prompt(chat)

        if normalized in {"low", "medium", "high"}:
            repositories.update_chat_reasoning(db, chat, enabled=True, effort=normalized)
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
            return _format_setup_summary(chat)

        return "⚠️ Ответьте `on`, `off`, `low`, `medium`, `high` или `/skip`."

    if chat.setup_stage == SETUP_STAGE_REASONING_EFFORT:
        normalized = value.lower()
        if value == "/skip":
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
            return _format_setup_summary(chat)

        if normalized in {"low", "medium", "high"}:
            repositories.update_chat_reasoning(db, chat, enabled=True, effort=normalized)
            repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
            return _format_setup_summary(chat)

        return "⚠️ Укажите `low`, `medium`, `high` или `/skip`."

    repositories.update_chat_setup_stage(db, chat, SETUP_STAGE_READY)
    return _format_setup_summary(chat)


def is_chat_setup_pending(chat: Chat) -> bool:
    return chat.setup_stage != SETUP_STAGE_READY


def get_setup_prompt(chat: Chat) -> str:
    if chat.setup_stage == SETUP_STAGE_TITLE:
        return (
            "🆕 Новый чат создан.\n"
            "1. Отправьте название чата.\n"
            f"Сейчас: {chat.title}\n"
            "Можно отправить /skip, чтобы оставить текущее название."
        )

    if chat.setup_stage == SETUP_STAGE_MODEL:
        return (
            "⚙️ 2. Выберите модель для чата.\n"
            "Отправьте номер из /models или полный model_id.\n"
            f"Сейчас: {chat.model}\n"
            "Если в модели есть :free, бот сам выберет free-ключ.\n"
            "Можно отправить /skip, чтобы оставить текущую модель."
        )

    if chat.setup_stage == SETUP_STAGE_REASONING:
        return (
            "🧠 3. Настроим reasoning.\n"
            "Отправьте on/off или сразу low/medium/high.\n"
            f"Сейчас: {str(chat.reasoning_enabled).lower()} / {chat.reasoning_effort}\n"
            "Можно отправить /skip, чтобы оставить текущее значение."
        )

    if chat.setup_stage == SETUP_STAGE_REASONING_EFFORT:
        return (
            "🧠 4. Выберите уровень reasoning.\n"
            "Доступно: low, medium, high.\n"
            f"Сейчас: {chat.reasoning_effort}\n"
            "Можно отправить /skip, чтобы оставить текущее значение."
        )

    return _format_setup_summary(chat)


def _handle_new_chat(db: Session, vk_user_id: int) -> str:
    chat = create_chat_for_setup(db, vk_user_id)
    return get_setup_prompt(chat)


def _format_setup_summary(chat: Chat) -> str:
    return (
        "✅ Чат настроен и готов к работе.\n"
        f"💬 Название: {chat.title}\n"
        f"⚙️ Модель: {chat.model}\n"
        f"🔑 Профиль: {chat.api_profile}\n"
        f"🧠 Reasoning: {str(chat.reasoning_enabled).lower()} / {chat.reasoning_effort}\n\n"
        "Теперь можно просто писать сообщение."
    )
