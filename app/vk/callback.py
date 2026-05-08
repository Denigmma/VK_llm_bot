import base64
from typing import Any

import httpx
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

SUPPORTED_DOC_EXTENSIONS = {"pdf"}
PDF_DOWNLOAD_TIMEOUT = 60.0
PDF_DOWNLOAD_ERROR_TEXT = "Не удалось подготовить PDF-файл для отправки в модель. Попробуйте отправить документ еще раз немного позже."


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
        image_urls = _extract_image_urls(attachments_payload)
        pdf_count = sum(1 for item in attachments_payload if item.get("type") == "pdf")

        logger.info(
            "VK message_new received: from_id=%s peer_id=%s text_len=%s photos=%s pdfs=%s",
            message.from_id,
            message.peer_id,
            len(text),
            len(image_urls),
            pdf_count,
        )

        if not text and not attachments_payload:
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
    attachments: list[dict[str, Any]],
) -> None:
    try:
        prepared_attachments = _prepare_attachments_for_model(attachments)
        image_urls = _extract_image_urls(prepared_attachments)
        pdf_count = sum(1 for item in prepared_attachments if item.get("type") == "pdf")

        logger.info(
            "Processing VK background task: from_id=%s peer_id=%s text_len=%s photos=%s pdfs=%s",
            from_id,
            peer_id,
            len(text),
            len(image_urls),
            pdf_count,
        )
        answer = handle_user_message(
            vk_user_id=from_id,
            text=text,
            image_urls=image_urls,
            attachments=prepared_attachments,
        )
        send_message(peer_id=peer_id, message=answer)
    except PdfPreparationError as exc:
        logger.warning("Failed to prepare PDF attachment: %s", exc)
        send_message(peer_id=peer_id, message=PDF_DOWNLOAD_ERROR_TEXT)
    except Exception:
        logger.exception("Failed to process VK message")


def _extract_attachments_payload(message: VkMessage) -> list[dict[str, Any]]:
    attachments_payload: list[dict[str, Any]] = []
    for attachment in message.attachments:
        if attachment.type == "photo" and attachment.photo is not None:
            largest_size = _get_largest_photo_size(attachment.photo.sizes)
            if largest_size is None or not largest_size.url:
                continue

            attachments_payload.append(
                {
                    "type": "photo",
                    "image_url": largest_size.url,
                }
            )
            continue

        if attachment.type == "doc" and attachment.doc is not None:
            extension = (attachment.doc.ext or "").lower()
            if extension not in SUPPORTED_DOC_EXTENSIONS:
                continue
            if not attachment.doc.url:
                continue

            attachments_payload.append(
                {
                    "type": "pdf",
                    "filename": _build_pdf_filename(attachment.doc.title, extension),
                    "source_url": attachment.doc.url,
                }
            )
    return attachments_payload


def _prepare_attachments_for_model(attachments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for attachment in attachments:
        attachment_type = attachment.get("type")
        if attachment_type == "photo":
            prepared.append(attachment)
            continue

        if attachment_type != "pdf":
            continue

        source_url = attachment.get("source_url")
        filename = attachment.get("filename") or "document.pdf"
        if not isinstance(source_url, str) or not source_url:
            raise PdfPreparationError("missing source_url")

        file_data = _download_pdf_as_data_url(source_url)
        prepared.append(
            {
                "type": "pdf",
                "filename": filename,
                "source_url": source_url,
                "file_data": file_data,
            }
        )
    return prepared


def _download_pdf_as_data_url(source_url: str) -> str:
    try:
        with httpx.Client(timeout=PDF_DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            response = client.get(source_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise PdfPreparationError(str(exc)) from exc

    encoded = base64.b64encode(response.content).decode("utf-8")
    return f"data:application/pdf;base64,{encoded}"


def _extract_image_urls(attachments: list[dict[str, Any]]) -> list[str]:
    return [
        item["image_url"]
        for item in attachments
        if item.get("type") == "photo" and item.get("image_url")
    ]


def _build_pdf_filename(title: str | None, extension: str) -> str:
    normalized_extension = extension.lower().lstrip(".") or "pdf"
    base_name = (title or "document").strip() or "document"
    if base_name.lower().endswith(f".{normalized_extension}"):
        return base_name
    return f"{base_name}.{normalized_extension}"


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


class PdfPreparationError(RuntimeError):
    pass
