PRESET_MODELS: list[str] = [
    "openai/gpt-5.4",
    "openai/gpt-5-nano",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "qwen/qwen3-235b-a22b:free",
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
]


def format_models_catalog() -> str:
    lines = ["⚙️ Предустановленные модели:"]
    lines.extend(
        f"{index}. {model} [{_infer_profile_label(model)}]"
        for index, model in enumerate(PRESET_MODELS, start=1)
    )
    return "\n".join(lines)


def get_model_by_number(number: int) -> str | None:
    if 1 <= number <= len(PRESET_MODELS):
        return PRESET_MODELS[number - 1]
    return None


def _infer_profile_label(model: str) -> str:
    if ":free" in model.lower():
        return "free"
    return "pay"
