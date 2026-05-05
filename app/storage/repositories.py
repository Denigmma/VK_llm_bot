import json

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from app.storage.models import Chat, Message, User


def get_or_create_user(db: Session, vk_user_id: int) -> User:
    user = db.scalar(select(User).where(User.vk_user_id == vk_user_id))
    if user is not None:
        return user

    user = User(vk_user_id=vk_user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_active_chat(db: Session, vk_user_id: int) -> Chat | None:
    return db.scalar(
        select(Chat)
        .where(Chat.vk_user_id == vk_user_id, Chat.is_active.is_(True))
        .order_by(Chat.created_at.desc(), Chat.id.desc())
        .limit(1)
    )


def count_user_chats(db: Session, vk_user_id: int) -> int:
    return db.scalar(select(func.count(Chat.id)).where(Chat.vk_user_id == vk_user_id)) or 0


def create_new_chat(
    db: Session,
    vk_user_id: int,
    *,
    title: str,
    api_profile: str,
    setup_stage: str,
    model: str,
    system_prompt: str,
    reasoning_enabled: bool,
    reasoning_effort: str,
    temperature: float,
    max_context_messages: int,
    is_active: bool = True,
) -> Chat:
    if is_active:
        db.execute(
            update(Chat)
            .where(Chat.vk_user_id == vk_user_id, Chat.is_active.is_(True))
            .values(is_active=False)
        )

    chat = Chat(
        vk_user_id=vk_user_id,
        title=title,
        api_profile=api_profile,
        setup_stage=setup_stage,
        model=model,
        system_prompt=system_prompt,
        reasoning_enabled=reasoning_enabled,
        reasoning_effort=reasoning_effort,
        temperature=temperature,
        max_context_messages=max_context_messages,
        is_active=is_active,
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


def make_chat_active(db: Session, vk_user_id: int, chat_id: int) -> Chat | None:
    chat = db.scalar(select(Chat).where(Chat.id == chat_id, Chat.vk_user_id == vk_user_id))
    if chat is None:
        return None

    db.execute(
        update(Chat)
        .where(Chat.vk_user_id == vk_user_id, Chat.is_active.is_(True))
        .values(is_active=False)
    )
    chat.is_active = True
    db.commit()
    db.refresh(chat)
    return chat


def list_user_chats(db: Session, vk_user_id: int, limit: int = 10) -> list[Chat]:
    return list(
        db.scalars(
            select(Chat)
            .where(Chat.vk_user_id == vk_user_id)
            .order_by(Chat.created_at.desc(), Chat.id.desc())
            .limit(limit)
        )
    )


def get_latest_chat(db: Session, vk_user_id: int) -> Chat | None:
    return db.scalar(
        select(Chat)
        .where(Chat.vk_user_id == vk_user_id)
        .order_by(Chat.created_at.desc(), Chat.id.desc())
        .limit(1)
    )


def clear_chat_messages(db: Session, chat_id: int) -> None:
    db.execute(delete(Message).where(Message.chat_id == chat_id))
    db.commit()


def save_message(
    db: Session,
    chat_id: int,
    role: str,
    content: str,
    *,
    image_url: str | None = None,
    attachments: list[dict] | None = None,
) -> Message:
    attachments_json = json.dumps(attachments, ensure_ascii=False) if attachments else None
    message = Message(
        chat_id=chat_id,
        role=role,
        content=content,
        image_url=image_url,
        attachments_json=attachments_json,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_last_messages(db: Session, chat_id: int, limit: int) -> list[Message]:
    if limit <= 0:
        return []

    messages = list(
        db.scalars(
            select(Message)
            .where(Message.chat_id == chat_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
    )
    return list(reversed(messages))


def update_chat_model(db: Session, chat: Chat, model: str) -> Chat:
    chat.model = model
    db.commit()
    db.refresh(chat)
    return chat


def update_chat_api_profile(
    db: Session,
    chat: Chat,
    *,
    api_profile: str,
    model: str | None = None,
    setup_stage: str | None = None,
) -> Chat:
    chat.api_profile = api_profile
    if model is not None:
        chat.model = model
    if setup_stage is not None:
        chat.setup_stage = setup_stage
    db.commit()
    db.refresh(chat)
    return chat


def update_chat_reasoning(db: Session, chat: Chat, *, enabled: bool, effort: str) -> Chat:
    chat.reasoning_enabled = enabled
    chat.reasoning_effort = effort
    db.commit()
    db.refresh(chat)
    return chat


def update_chat_system_prompt(db: Session, chat: Chat, system_prompt: str) -> Chat:
    chat.system_prompt = system_prompt
    db.commit()
    db.refresh(chat)
    return chat


def update_chat_title(db: Session, chat: Chat, title: str, *, setup_stage: str | None = None) -> Chat:
    chat.title = title
    if setup_stage is not None:
        chat.setup_stage = setup_stage
    db.commit()
    db.refresh(chat)
    return chat


def update_chat_setup_stage(db: Session, chat: Chat, setup_stage: str) -> Chat:
    chat.setup_stage = setup_stage
    db.commit()
    db.refresh(chat)
    return chat


def delete_chat(db: Session, chat: Chat) -> Chat | None:
    vk_user_id = chat.vk_user_id
    was_active = chat.is_active
    db.delete(chat)
    db.commit()

    if not was_active:
        return get_active_chat(db, vk_user_id)

    next_chat = get_latest_chat(db, vk_user_id)
    if next_chat is None:
        return None

    next_chat.is_active = True
    db.commit()
    db.refresh(next_chat)
    return next_chat
