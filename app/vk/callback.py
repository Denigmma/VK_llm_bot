from typing import Any

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError

from app.bot.dialog import handle_user_message
from app.config import get_settings
from app.utils.logger import get_logger
from app.utils.text import normalize_message_text
from app.vk.schemas import VkCallbackEvent, VkMessage, VkMessageNewObject, VkPhotoSize
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
        attachments_payload = _extract_attachments_payload(message)
        image_urls = [
            item["image_url"]
            for item in attachments_payload
            if item.get("type") == "photo" and item.get("image_url")
        ]

        logger.info(
            "VK message_new received: from_id=%s peer_id=%s text_len=%s photos=%s",
            message.from_id,
            message.peer_id,
            len(text),
            len(image_urls),
        )

        if not text and not image_urls:
            logger.info("VK message ignored: no text and no supported attachments")
            return PlainTextResponse("ok")

        if message.peer_id is None or message.from_id is None:
            logger.warning("Ignored VK message_new without peer_id or from_id")
            return PlainTextResponse("ok")

        background_tasks.add_task(
            _process_message,
            message.peer_id,
            message.from_id,
            text,
            image_urls,
            attachments_payload,
        )

    return PlainTextResponse("ok")


def _extract_message(payload: dict[str, Any]) -> VkMessage | None:
    try:
        event = VkCallbackEvent.model_validate(payload)
        message_object = VkMessageNewObject.model_validate(event.object or {})
    except ValidationError:
        logger.exception("Failed to parse VK callback payload")
        return None
    return message_object.message


def _process_message(
    peer_id: int,
    from_id: int,
    text: str,
    image_urls: list[str],
    attachments: list[dict[str, str]],
) -> None:
    try:
        logger.info(
            "Processing VK background task: from_id=%s peer_id=%s text_len=%s photos=%s",
            from_id,
            peer_id,
            len(text),
            len(image_urls),
        )
        answer = handle_user_message(
            vk_user_id=from_id,
            text=text,
            image_urls=image_urls,
            attachments=attachments,
        )
        send_message(peer_id=peer_id, message=answer)
    except Exception:
        logger.exception("Failed to process VK message")


def _extract_attachments_payload(message: VkMessage) -> list[dict[str, str]]:
    attachments_payload: list[dict[str, str]] = []
    for attachment in message.attachments:
        if attachment.type != "photo" or attachment.photo is None:
            continue

        largest_size = _get_largest_photo_size(attachment.photo.sizes)
        if largest_size is None or not largest_size.url:
            continue

        attachments_payload.append(
            {
                "type": "photo",
                "image_url": largest_size.url,
            }
        )
    return attachments_payload


def _get_largest_photo_size(sizes: list[VkPhotoSize]) -> VkPhotoSize | None:
    valid_sizes = [size for size in sizes if size.url]
    if not valid_sizes:
        return None

    return max(
        valid_sizes,
        key=lambda size: (
            (size.width or 0) * (size.height or 0),
            size.width or 0,
            size.height or 0,
        ),
    )
