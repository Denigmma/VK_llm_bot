VK_MESSAGE_LIMIT = 3500


def normalize_message_text(text: str | None) -> str:
    if text is None:
        return ""
    return text.strip()


def split_text_for_vk(text: str, limit: int = VK_MESSAGE_LIMIT) -> list[str]:
    text = text or " "
    if len(text) <= limit:
        return [text]

    parts = [text]
    while True:
        total = len(parts)
        next_parts: list[str] = []

        for index, part in enumerate(parts, start=1):
            prefix = f"Часть {index}/{total}\n\n"
            capacity = limit - len(prefix)
            if capacity <= 0:
                raise ValueError("VK message limit is too small for chunk prefix")
            next_parts.extend(_split_raw_text(part, capacity))

        if len(next_parts) == len(parts):
            final_total = len(next_parts)
            return [
                f"Часть {index}/{final_total}\n\n{part}"
                for index, part in enumerate(next_parts, start=1)
            ]

        parts = next_parts


def _split_raw_text(text: str, limit: int) -> list[str]:
    chunks: list[str] = []
    rest = text

    while len(rest) > limit:
        cut = _find_soft_cut(rest, limit)
        chunk = rest[:cut].rstrip()
        if chunk:
            chunks.append(chunk)
        rest = rest[cut:].lstrip()

    if rest:
        chunks.append(rest)
    return chunks


def _find_soft_cut(text: str, limit: int) -> int:
    newline_pos = text.rfind("\n", 0, limit + 1)
    space_pos = text.rfind(" ", 0, limit + 1)
    cut = max(newline_pos, space_pos)

    if cut >= limit // 2:
        return cut
    return limit
