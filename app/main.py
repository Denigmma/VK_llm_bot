from fastapi import FastAPI

from app.storage.db import init_db
from app.utils.logger import setup_logging
from app.vk.callback import router as vk_router


setup_logging()

app = FastAPI(title="VK OpenRouter Bot")
app.include_router(vk_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
