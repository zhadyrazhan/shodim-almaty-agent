"""Turn scraped markdown into typed, structured records.

The scraped pages are long and noisy (nav bars, ad blocks, duplicated links), so
we hand chunks to an LLM with a Pydantic schema and let structured output do the
parsing. Records are deduplicated by URL, since the same event appears on both
the main page and the afisha listing.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from .config import EXTRACT_MODEL, RAW_DIR, SXODIM_DATA, require_openai_key

ItemKind = Literal["event", "place", "restaurant", "entertainment"]


class Item(BaseModel):
    title: str = Field(description="Название события или места")
    kind: ItemKind = Field(description="Тип записи")
    description: str = Field(description="Краткое описание, 1-2 предложения")
    category: str = Field(description="Категория: концерт, выставка, театр, кафе, парк и т.п.")
    url: str = Field(default="", description="Ссылка на sxodim.com, если есть")
    date: str = Field(default="", description="Дата или период, если указаны")
    price: str = Field(default="", description="Цена или «бесплатно», если указана")
    address: str = Field(default="", description="Адрес или район, если указан")
    good_for: list[str] = Field(
        default_factory=list,
        description="Кому подойдёт: свидание, дети, компания друзей, семья, одному",
    )


class ItemBatch(BaseModel):
    items: list[Item]


EXTRACT_PROMPT = """Ты извлекаешь афишу Алматы из markdown, полученного со сайта sxodim.com.

Из фрагмента ниже выпиши все конкретные события, места, рестораны и развлечения.

Требования:
- Только реальные события и места. Игнорируй навигацию, кнопки, рекламу, ссылки на соцсети, повторяющиеся меню.
- `good_for` — кому это подойдёт: «свидание», «дети», «компания друзей», «семья», «одному». Заполняй по смыслу описания.
- Если поля нет в тексте — оставь пустую строку, не выдумывай.
- Если во фрагменте нет ни одного реального события или места — верни пустой список.

Фрагмент:
{chunk}
"""


def chunk_markdown(text: str, size: int = 6000) -> list[str]:
    """Split on blank lines, packing paragraphs up to `size` characters."""
    chunks, current = [], ""
    for block in re.split(r"\n\s*\n", text):
        if len(current) + len(block) > size and current:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current.strip():
        chunks.append(current)
    return chunks


def extract_chunk(chunk: str, model: str = EXTRACT_MODEL, retries: int = 3) -> list[Item]:
    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())
    for attempt in range(retries + 1):
        try:
            response = client.responses.parse(
                model=model,
                input=EXTRACT_PROMPT.format(chunk=chunk),
                text_format=ItemBatch,
                reasoning={"effort": "minimal"},
            )
            return response.output_parsed.items
        except Exception as e:
            if "insufficient_quota" in str(e) or attempt == retries:
                print(f"  [warn] chunk failed: {e}")
                return []
            wait = 5 * (attempt + 1)
            print(f"  [warn] attempt {attempt + 1} failed ({e}); retry in {wait}s")
            time.sleep(wait)
    return []


def dedupe(items: list[Item]) -> list[Item]:
    """Drop repeats — the same event is listed on several pages."""
    seen: set[str] = set()
    unique: list[Item] = []
    for item in items:
        key = (item.url or item.title).strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def save(items: list[Item], out: Path) -> list[dict]:
    payload = [item.model_dump() for item in dedupe(items)]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def extract_all(raw_dir: Path = RAW_DIR, out: Path = SXODIM_DATA) -> list[dict]:
    pages = sorted(raw_dir.glob("*.md"))
    if not pages:
        raise RuntimeError(f"no scraped markdown in {raw_dir} — run src/scraper.py first")

    all_chunks = [
        (page.name, chunk)
        for page in pages
        for chunk in chunk_markdown(page.read_text(encoding="utf-8"))
    ]
    print(f"{len(pages)} pages -> {len(all_chunks)} chunks", flush=True)

    collected: list[Item] = []
    for i, (page_name, chunk) in enumerate(all_chunks, 1):
        items = extract_chunk(chunk)
        collected.extend(items)
        # Save after every chunk: a 57-call run that dies at chunk 55 should
        # not throw away everything it already paid for.
        payload = save(collected, out)
        print(
            f"  [{i}/{len(all_chunks)}] {page_name}: +{len(items)} "
            f"(kept {len(payload)} unique)",
            flush=True,
        )

    payload = save(collected, out)
    print(f"\ndeduped {len(collected)} -> {len(payload)} records")
    print(f"wrote {out}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RAW_DIR)
    parser.add_argument("--out", type=Path, default=SXODIM_DATA)
    args = parser.parse_args()
    extract_all(args.raw, args.out)


if __name__ == "__main__":
    main()
