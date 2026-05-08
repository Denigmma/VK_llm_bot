from app.config import OpenRouterProfile


PRESET_MODELS: list[str] = [
    "openai/gpt-5.4",
    "openai/gpt-5-nano",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v3.2",
    "deepseek/deepseek-v4-pro",
    "google/gemini-2.5-pro",
    "openai/o3",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "tencent/hy3-preview:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-31b-it:free"
]


VISION_FALLBACK_MODELS: dict[OpenRouterProfile, str] = {
    "free": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "pay": "openai/gpt-5.4",
}


VISION_MODELS = {
    "openai/gpt-5.4",
    "openai/gpt-5-nano",
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "google/gemini-2.0-flash-exp:free",
}


TEXT_ONLY_MODELS = {
    "nvidia/nemotron-3-super-120b-a12b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "meta-llama/llama-3.3-70b-instruct:free",
}


VISION_KEYWORDS = ("omni", "vision", "gpt-5", "gpt-4o", "gemini", "pixtral", "vl")
TEXT_ONLY_KEYWORDS = ("deepseek-chat", "llama-3.3-70b-instruct", "nemotron-3-super-120b-a12b")


def format_models_catalog() -> str:
    lines = ["⚙️ Предустановленные модели:"]
    lines.extend(
        f"{index}. {model} [{', '.join(_build_model_labels(model))}]"
        for index, model in enumerate(PRESET_MODELS, start=1)
    )
    return "\n".join(lines)


def get_model_by_number(number: int) -> str | None:
    if 1 <= number <= len(PRESET_MODELS):
        return PRESET_MODELS[number - 1]
    return None


def model_supports_images(model: str) -> bool | None:
    normalized = model.strip().lower()
    if normalized in {item.lower() for item in VISION_MODELS}:
        return True
    if normalized in {item.lower() for item in TEXT_ONLY_MODELS}:
        return False
    if any(keyword in normalized for keyword in VISION_KEYWORDS):
        return True
    if any(keyword in normalized for keyword in TEXT_ONLY_KEYWORDS):
        return False
    return None


def get_vision_fallback_model(profile: OpenRouterProfile) -> str:
    return VISION_FALLBACK_MODELS[profile]


def _build_model_labels(model: str) -> list[str]:
    labels = [_infer_profile_label(model)]
    supports_images = model_supports_images(model)
    if supports_images is True:
        labels.append("vision")
    elif supports_images is False:
        labels.append("text")
    return labels


def _infer_profile_label(model: str) -> str:
    if ":free" in model.lower():
        return "free"
    return "pay"
