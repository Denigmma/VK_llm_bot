from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.storage.models import Base


settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
    class_=Session,
)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_schema_updates()


@contextmanager
def session_scope() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _ensure_schema_updates() -> None:
    inspector = inspect(engine)
    if "chats" not in inspector.get_table_names():
        return

    chat_columns = {column["name"] for column in inspector.get_columns("chats")}
    if "api_profile" not in chat_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE chats "
                    "ADD COLUMN api_profile VARCHAR(16) NOT NULL DEFAULT 'free'"
                )
            )
    if "setup_stage" not in chat_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE chats "
                    "ADD COLUMN setup_stage VARCHAR(32) NOT NULL DEFAULT 'ready'"
                )
            )
    if "pdf_parser_engine" not in chat_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE chats "
                    "ADD COLUMN pdf_parser_engine VARCHAR(32) NOT NULL DEFAULT 'cloudflare-ai'"
                )
            )

    if "messages" not in inspector.get_table_names():
        return

    message_columns = {column["name"] for column in inspector.get_columns("messages")}
    if "image_url" not in message_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE messages "
                    "ADD COLUMN image_url TEXT NULL"
                )
            )
    if "attachments_json" not in message_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE messages "
                    "ADD COLUMN attachments_json TEXT NULL"
                )
            )
    if "annotations_json" not in message_columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "ALTER TABLE messages "
                    "ADD COLUMN annotations_json TEXT NULL"
                )
            )
