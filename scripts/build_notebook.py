"""Generate sxodim_agent.ipynb from source, so the notebook never drifts from src/.

The project brief requires a notebook with saved outputs. Rather than maintain
duplicated logic in .ipynb JSON, the notebook imports from src/ and this script
regenerates its skeleton. Run it, then execute the notebook to capture outputs.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "sxodim_agent.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    md("""# Агент «Куда сходить в Алматы»

**Пайплайн:** sxodim.com → Jina Reader → GPT-5-mini (структурирование) → LlamaIndex RAG → агент

Логика живёт в `src/`, ноутбук её импортирует — так код не дублируется между `.py` и `.ipynb`.
"""),
    md("""## 0. Настройка

**В Colab** достаточно запустить ячейку ниже — она клонирует репозиторий (ноутбук импортирует из `src/`, \
поэтому одного `.ipynb` недостаточно), поставит зависимости и возьмёт ключ из панели Secrets (🔑 слева). \
Добавь туда `OPENAI_API_KEY` до запуска. GPU не нужен — хватит CPU runtime.

**Локально** ячейка ничего не делает: нужен `pip install -r requirements.txt` и `OPENAI_API_KEY` в `.env`.
"""),
    code('''REPO_URL = "https://github.com/zhadyrazhan/shodim-almaty-agent.git"

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import os
    import subprocess
    from pathlib import Path

    if not Path("shodim-almaty-agent").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL], check=True)
    if Path("shodim-almaty-agent").exists():
        os.chdir("shodim-almaty-agent")

    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)

    from google.colab import userdata

    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("Colab: репозиторий и зависимости готовы, ключ загружен")
else:
    print("Локальный запуск — ключ берётся из .env")'''),
    code("""import json

from src import scraper, extract, agent
from src.config import SXODIM_DATA, RAW_DIR, LLM_BACKEND

print("LLM backend:", LLM_BACKEND)"""),
    md("""## 1. Парсинг сайта (Jina Reader)

Jina Reader отдаёт готовый markdown по любому URL, поэтому не нужно писать CSS-селекторы под вёрстку, которая может измениться."""),
    code("""written = scraper.scrape_all()
for name, path in written.items():
    print(f"{name}: {path.stat().st_size:,} байт")"""),
    md("### Фрагмент спарсенных данных"),
    code("""raw = (RAW_DIR / "afisha.md").read_text(encoding="utf-8")
print(raw[:800])"""),
    md("""## 2. Структурирование в JSON

Спарсенный markdown шумный (меню, реклама, дубли ссылок), поэтому он режется на чанки и передаётся модели со схемой Pydantic — structured output сам приводит всё к типам."""),
    code("""records = extract.extract_all()
print(f"\\nвсего записей: {len(records)}")"""),
    md("### Структурированный JSON (пример)"),
    code("""data = json.loads(SXODIM_DATA.read_text(encoding="utf-8"))
print(f"записей: {len(data)}\\n")
print(json.dumps(data[:3], ensure_ascii=False, indent=2))"""),
    code("""from collections import Counter
print("по типам:", dict(Counter(d["kind"] for d in data)))
print("по категориям:", dict(Counter(d["category"] for d in data).most_common(10)))"""),
    md("""## 3. Агент (LlamaIndex + OpenAI)

Записи индексируются в `VectorStoreIndex`, поверх — query engine с системным промптом гида."""),
    code("""TEST_QUESTIONS = [
    "Куда сходить на выходных?",
    "Посоветуй место для свидания",
    "Куда сводить ребенка?",
    "Какие концерты будут?",
    "Где вкусно поесть?",
]

answers = {}
for q in TEST_QUESTIONS:
    a = agent.ask(q)
    answers[q] = a
    print(f"Q: {q}\\nA: {a}\\n{'-' * 70}")"""),
    md("""## 4. Сохранение примеров диалогов

Обязательный дилеверабл `agent_examples.md` — 5+ примеров."""),
    code("""lines = ["# Примеры диалогов с агентом\\n"]
for q, a in answers.items():
    lines.append(f"\\n## {q}\\n\\n{a}\\n")

(agent.SXODIM_DATA.parent.parent / "agent_examples.md").write_text(
    "".join(lines), encoding="utf-8"
)
print(f"сохранено {len(answers)} диалогов в agent_examples.md")"""),
    md("""## 5. Бонус: ORPO — делаем ответы дружелюбнее

Базовый агент отвечает корректно, но суховато. ORPO (Odds Ratio Preference Optimization) \
объединяет SFT и выравнивание по предпочтениям в один проход: не нужны ни отдельная \
reward-модель, ни reference-модель в памяти, поэтому всё помещается на бесплатный T4.

**Нужен GPU.** Runtime → Change runtime type → **T4 GPU**, затем Runtime → **Restart session** \
(смена типа без перезапуска не переносит сессию на GPU). Секции 1-4 выше работают и на CPU: \
если GPU нет, ячейки ниже сами себя пропустят."""),
    code("""import subprocess
import torch

HAS_GPU = torch.cuda.is_available()
print("CUDA available:", HAS_GPU)

if HAS_GPU:
    print("GPU:", torch.cuda.get_device_name(0))
    # Ставим только здесь, а не в секции 0: это ~2 ГБ пакетов, которые нужны
    # исключительно для ORPO. На CPU-прогоне секций 1-4 они только мешают.
    print("ставим зависимости для обучения...")
    subprocess.run(
        "pip install -q unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets".split(),
        check=True,
    )
    print("готово")
else:
    print("GPU нет — секция 5 будет пропущена")"""),
    md("""### 5.1 Датасет предпочтений

ORPO нужны тройки (prompt, chosen, rejected). Обе стороны генерируются на **одних и тех же** \
записях афиши: `chosen` — тёплый ответ живым языком, `rejected` — сухая справка списком. \
Факты одинаковые, отличается только тон, значит модель учится именно стилю, а не содержанию.

Шаг идёт через OpenAI API и GPU не требует."""),
    code("""if HAS_GPU:
    !python training/build_preference_data.py --n 120"""),
    code('''from pathlib import Path

if HAS_GPU:
    pairs = json.loads(Path("data/preference_data.json").read_text(encoding="utf-8"))
    print(f"пар: {len(pairs)}")
    p = pairs[0]
    print("\\nВОПРОС:", p["prompt"])
    print("\\n--- CHOSEN (тёплый) ---")
    print(p["chosen"][:500])
    print("\\n--- REJECTED (сухой) ---")
    print(p["rejected"][:500])'''),
    md("### 5.2 Ответы ДО обучения"),
    code('''if HAS_GPU:
    from unsloth import FastLanguageModel

    BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True
    )

    def gen(m, t, question, max_new_tokens=220):
        prompt = t.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = t([prompt], return_tensors="pt").to("cuda")
        out = m.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return t.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    FastLanguageModel.for_inference(model)
    before = {q: gen(model, tok, q) for q in TEST_QUESTIONS[:3]}
    for q, a in before.items():
        print(f"Q: {q}\\nA: {a}\\n{'-' * 70}")'''),
    md("""### 5.3 Обучение

Следим не только за падением loss, но и за **`rewards/margins`**: именно рост маржи \
показывает, что модель разводит тёплый и сухой ответы, а не просто подгоняется под оба."""),
    code("""if HAS_GPU:
    !python training/train_orpo.py --pairs data/preference_data.json --epochs 3"""),
    md("### 5.4 Ответы ПОСЛЕ обучения"),
    code('''if HAS_GPU:
    from peft import PeftModel

    tuned, tuned_tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True
    )
    tuned = PeftModel.from_pretrained(tuned, "outputs/orpo-almaty")
    FastLanguageModel.for_inference(tuned)

    after = {q: gen(tuned, tuned_tok, q) for q in TEST_QUESTIONS[:3]}
    for q, a in after.items():
        print(f"Q: {q}\\nA: {a}\\n{'-' * 70}")'''),
    md("""### 5.5 Сравнение ДО / ПОСЛЕ

Главный артефакт бонусной части: видно ли, что тон стал теплее."""),
    code('''if HAS_GPU:
    import pandas as pd

    df = pd.DataFrame(
        [{"вопрос": q, "ДО": before[q][:180], "ПОСЛЕ": after[q][:180]} for q in before]
    )
    pd.set_option("display.max_colwidth", 180)
    display(df)

    lines = ["# ORPO: ответы до и после\\n"]
    for q in before:
        lines.append(f"\\n## {q}\\n\\n**До:**\\n\\n{before[q]}\\n\\n**После:**\\n\\n{after[q]}\\n")
    Path("orpo_examples.md").write_text("".join(lines), encoding="utf-8")
    print("\\nсохранено в orpo_examples.md")'''),
    md("""## 6. Скачать результаты

Файлы лежат внутри runtime и исчезнут вместе с сессией, поэтому забери их сразу. \
Сам ноутбук скачивается отдельно: File → Download → Download .ipynb — **после** того, \
как все ячейки отработали, чтобы выводы сохранились."""),
    code('''from pathlib import Path

ARTIFACTS = ["agent_examples.md", "data/sxodim_data.json", "orpo_examples.md"]

if IN_COLAB:
    from google.colab import files

    for path in ARTIFACTS:
        # orpo_examples.md only exists if section 5 ran (needs a GPU).
        if Path(path).exists():
            files.download(path)
        else:
            print(f"{path}: пропущен (не создан)")
else:
    for path in ARTIFACTS:
        p = Path(path)
        print(f"{path}: {'есть' if p.exists() else 'НЕТ'}"
              f"{f' ({p.stat().st_size:,} байт)' if p.exists() else ''}")'''),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
