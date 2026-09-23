"""Build an ORPO preference dataset for answer friendliness.

ORPO needs (prompt, chosen, rejected) triples. Rather than hand-write hundreds,
we generate both sides synthetically from the same listings: the *chosen* answer
is warm and specific, the *rejected* one is curt and impersonal. Since both are
grounded in the same records, the only axis the model learns is tone — not facts.

    python training/build_preference_data.py --n 120
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import load_records  # noqa: E402
from src.config import EXTRACT_MODEL, require_openai_key  # noqa: E402

OUT = Path(__file__).parent.parent / "data" / "preference_data.json"

QUESTION_TEMPLATES = [
    "Куда сходить на выходных?",
    "Посоветуй место для свидания",
    "Куда сводить ребенка?",
    "Какие концерты будут?",
    "Где вкусно поесть?",
    "Что нового открылось в Алматы?",
    "Куда пойти с друзьями вечером?",
    "Чем заняться в дождливый день?",
    "Куда сходить одному?",
    "Где провести время с семьей?",
    "Есть что-нибудь бесплатное?",
    "Посоветуй что-то необычное",
    "Куда пойти после работы?",
    "Где послушать живую музыку?",
    "Куда сходить с родителями?",
]

CHOSEN_STYLE = """Ты — дружелюбный местный гид по Алматы, который искренне любит свой город.
Отвечай тепло и живо, будто советуешь хорошему другу: обращайся на «ты», добавь \
короткую личную ремарку, почему тебе самому нравится это место, предложи 2-3 \
конкретных варианта с пояснением, кому и почему они подойдут. В конце — короткий \
дружелюбный вопрос или пожелание. Без канцелярита."""

REJECTED_STYLE = """Ты — справочная система. Отвечай сухо и формально: перечисли \
варианты списком без пояснений, без обращения к пользователю, без эмоций и \
рекомендаций. Минимум слов, канцелярский стиль."""

PROMPT = """Вопрос пользователя: {question}

Афиша Алматы (используй только её):
{context}

{style}"""


# What the model is actually trained on. The context MUST be here, not only in
# the generation prompt above: chosen/rejected both cite specific venues, so a
# bare question as the prompt would be teaching "invent venue names you cannot
# see". Training and inference have to feed the same shape.
TRAIN_PROMPT = """{question}

Афиша Алматы:
{context}"""


def format_training_prompt(question: str, context: str) -> str:
    """Build the prompt stored in the dataset — reused at eval so both match."""
    return TRAIN_PROMPT.format(question=question, context=context)


def format_records(records: list[dict], k: int = 5) -> str:
    lines = []
    for r in records[:k]:
        bits = [r.get("title", ""), r.get("category", ""), r.get("description", "")]
        if r.get("good_for"):
            bits.append("подойдёт: " + ", ".join(r["good_for"]))
        lines.append("- " + " | ".join(b for b in bits if b))
    return "\n".join(lines)


def generate(client, question: str, context: str, style: str, model: str) -> str:
    response = client.responses.create(
        model=model,
        input=PROMPT.format(question=question, context=context, style=style),
        reasoning={"effort": "minimal"},
    )
    return response.output_text.strip()


def build(n: int, model: str, seed: int = 42) -> list[dict]:
    from openai import OpenAI

    client = OpenAI(api_key=require_openai_key())
    records = load_records()
    rng = random.Random(seed)

    pairs = []
    for i in range(n):
        question = QUESTION_TEMPLATES[i % len(QUESTION_TEMPLATES)]
        sample = rng.sample(records, min(5, len(records)))
        context = format_records(sample)

        try:
            chosen = generate(client, question, context, CHOSEN_STYLE, model)
            rejected = generate(client, question, context, REJECTED_STYLE, model)
        except Exception as e:
            print(f"  [warn] pair {i + 1} failed: {e}")
            continue

        pairs.append(
            {
                "prompt": format_training_prompt(question, context),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
        print(f"  [{i + 1}/{n}] {question[:40]}")

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=120, help="number of preference pairs")
    parser.add_argument("--model", default=EXTRACT_MODEL)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    pairs = build(args.n, args.model)
    args.out.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} ({len(pairs)} pairs)")


if __name__ == "__main__":
    main()
