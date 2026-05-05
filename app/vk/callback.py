from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from app.bot.dialog import handle_user_message
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.text import normalize_message_text
from app.vk.schemas import VkCallbackEvent, VkMessageNewObject
from app.vk.sender import send_message


router = APIRouter(prefix="/vk", tags=["vk"])
logger = get_logger(__name__)


@router.post("/callback", response_class=PlainTextResponse)
async def vk_callback(payload: dict[str, Any], background_tasks: BackgroundTasks) -> PlainTextResponse:
    settings = get_settings()
    event_type = payload.get("type")

    if event_type == "confirmation":
        return PlainTextResponse(settings.vk_confirmation_code)

    if payload.get("secret") != settings.vk_secret_key:
        logger.warning("Ignored VK event with invalid secret")
        return PlainTextResponse("ok")

    if event_type == "message_new":
        message = _extract_message(payload)
        if message is None:
            return PlainTextResponse("ok")

        text = normalize_message_text(message.text)
        if not text:
            return PlainTextResponse("ok")

        if message.peer_id is None or message.from_id is None:
            logger.warning("Ignored VK message_new without peer_id or from_id")
            return PlainTextResponse("ok")

        background_tasks.add_task(_process_message, message.peer_id, message.from_id, text)

    return PlainTextResponse("ok")


def _extract_message(payload: dict[str, Any]):
    try:
        event = VkCallbackEvent.model_validate(payload)
        message_object = VkMessageNewObject.model_validate(event.object or {})
    except ValidationError:
        logger.exception("Failed to parse VK callback payload")
        return None
    return message_object.message


def _process_message(peer_id: int, from_id: int, text: str) -> None:
    try:
        answer = handle_user_message(vk_user_id=from_id, text=text)
        send_message(peer_id=peer_id, message=answer)
    except Exception:
        logger.exception("Failed to process VK message")
