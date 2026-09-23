"""Evaluate the Almaty agent against the golden set.

The listings change every day, so asserting specific event names would make this
suite fail for the wrong reason. Instead it checks behavior that must hold no
matter what is currently playing:

  grounding  - does it answer substantively from the listings?
  routing    - does the audience change the answer (no bars for a kid question)?
  refusal    - does it decline questions the listings cannot answer?

Usage:
    python -m src.agent            # make sure the agent works at all first
    python evals/run_eval.py
    python evals/run_eval.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import ask  # noqa: E402

GOLDEN_SET = Path(__file__).parent.parent / "data" / "golden_set.json"

REFUSAL_MARKERS = [
    "не могу",
    "нет информации",
    "не нашл",
    "отсутств",
    "не располагаю",
    "нет данных",
    "не указан",
    "только по алматы",
    "не входит",
    "к сожалению",
]


def normalize(text: str) -> str:
    for dash in "‐‑‒–—":
        text = text.replace(dash, "-")
    return " ".join(text.lower().replace("ё", "е").split())


def is_refusal(answer: str) -> bool:
    low = normalize(answer)
    return any(normalize(m) in low for m in REFUSAL_MARKERS)


def check(item: dict, answer: str) -> tuple[bool, str]:
    low = normalize(answer)

    if item.get("expect_refusal"):
        return (
            (True, "correctly declined")
            if is_refusal(answer)
            else (False, "answered instead of declining")
        )

    if len(answer) < item.get("min_length", 0):
        return False, f"too short ({len(answer)} chars, need {item['min_length']})"

    banned = [w for w in item.get("must_not_contain", []) if normalize(w) in low]
    if banned:
        return False, f"contains banned terms: {banned}"

    wanted = item.get("must_contain_any")
    if wanted and not any(normalize(w) in low for w in wanted):
        return False, f"missing all of: {wanted}"

    return True, "ok"


def evaluate(items: list[dict]) -> list[dict]:
    results = []
    for item in items:
        try:
            answer = ask(item["question"])
            passed, detail = check(item, answer)
        except Exception as e:
            answer, passed, detail = "", False, f"ERROR: {e}"

        results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "category": item["category"],
                "answer": answer,
                "passed": passed,
                "detail": detail,
            }
        )
        print(f"{'PASS' if passed else 'FAIL'}  {item['id']:22} {detail}")
    return results


def report(results: list[dict]) -> None:
    def rate(subset: list[dict]) -> str:
        if not subset:
            return "n/a"
        n = sum(r["passed"] for r in subset)
        return f"{n}/{len(subset)} ({n / len(subset):.0%})"

    by = lambda cat: [r for r in results if r["category"] == cat]  # noqa: E731

    print("\n" + "=" * 52)
    print(f"Brief questions:   {rate(by('brief_question'))}")
    print(f"Audience routing:  {rate(by('audience_routing'))}")
    print(f"Constraints:       {rate(by('constraint'))}")
    print(f"Off-topic refusal: {rate(by('off_topic'))}")
    print(f"Overall:           {rate(results)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", help="write full results to this JSON file")
    args = parser.parse_args()

    golden = json.loads(GOLDEN_SET.read_text(encoding="utf-8"))
    results = evaluate(golden["items"])
    report(results)

    if args.out:
        Path(args.out).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")

    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
