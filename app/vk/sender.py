import secrets

import httpx

from app.config import get_settings
from app.utils.text import split_text_for_vk


VK_MESSAGES_SEND_URL = "https://api.vk.com/method/messages.send"
VK_API_VERSION = "5.199"


class VkSendError(Exception):
    """Raised when VK API rejects an outgoing message."""


def send_message(peer_id: int, message: str) -> None:
    settings = get_settings()
    parts = split_text_for_vk(message)

    with httpx.Client(timeout=30.0) as client:
        for part in parts:
            payload = {
                "access_token": settings.vk_group_token,
                "peer_id": peer_id,
                "message": part,
                "random_id": secrets.randbelow(2_147_483_647),
                "v": VK_API_VERSION,
            }
            response = client.post(VK_MESSAGES_SEND_URL, data=payload)
            _raise_for_vk_error(response)


def _raise_for_vk_error(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise VkSendError(f"VK API HTTP error: {exc.response.status_code}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise VkSendError("VK API returned invalid JSON") from exc

    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        message = error.get("error_msg") or error.get("message") or "unknown VK API error"
        raise VkSendError(str(message))
