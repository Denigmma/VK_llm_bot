from typing import Any

from pydantic import BaseModel, ConfigDict


class VkPhotoSize(BaseModel):
    url: str | None = None
    width: int | None = None
    height: int | None = None

    model_config = ConfigDict(extra="ignore")


class VkPhoto(BaseModel):
    sizes: list[VkPhotoSize] = []

    model_config = ConfigDict(extra="ignore")


class VkAttachment(BaseModel):
    type: str | None = None
    photo: VkPhoto | None = None

    model_config = ConfigDict(extra="ignore")


class VkMessage(BaseModel):
    text: str | None = None
    peer_id: int | None = None
    from_id: int | None = None
    attachments: list[VkAttachment] = []

    model_config = ConfigDict(extra="ignore")


class VkMessageNewObject(BaseModel):
    message: VkMessage | None = None

    model_config = ConfigDict(extra="ignore")


class VkCallbackEvent(BaseModel):
    type: str | None = None
    group_id: int | None = None
    secret: str | None = None
    object: dict[str, Any] | None = None

    model_config = ConfigDict(extra="allow")
