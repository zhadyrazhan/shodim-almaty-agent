"""Generate orpo_training.ipynb — the bonus track, meant to run on a Colab GPU.

Same approach as build_notebook.py: the notebook drives code that lives in
training/, so there is one implementation rather than two.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "orpo_training.ipynb"


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
    md("""# ORPO: делаем ответы агента дружелюбнее

Бонусная часть проекта (+30 баллов). Базовый агент отвечает корректно, но суховато. \
ORPO (Odds Ratio Preference Optimization) объединяет SFT и выравнивание по предпочтениям \
в один проход: не нужны ни отдельная reward-модель, ни reference-модель в памяти — \
поэтому всё помещается на бесплатный T4.

**Что делаем:**
1. Генерируем пары (вопрос → тёплый ответ / сухой ответ) на одних и тех же данных афиши
2. Обучаем Qwen2.5-3B через ORPO отличать одно от другого
3. Сравниваем ответы ДО и ПОСЛЕ на одних и тех же вопросах

**Важно:** Runtime → Change runtime type → **T4 GPU**, затем Runtime → **Restart session**. \
Смена типа runtime без перезапуска не переносит сессию на GPU."""),
    md("## 0. Настройка"),
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
    subprocess.run(
        "pip install -q unsloth unsloth_zoo trl peft accelerate bitsandbytes datasets".split(),
        check=True,
    )

    from google.colab import userdata

    os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
    print("Colab: репозиторий и зависимости готовы, ключ загружен")
else:
    print("Локальный запуск — ключ берётся из .env")'''),
    md("""### Проверка GPU

Colab при исчерпанной квоте молча выдаёт CPU вместо ошибки, и обучение потом \
«просто очень долго идёт». Лучше убедиться сразу."""),
    code("""import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise SystemExit("GPU не подключён: Runtime -> Change runtime type -> T4 GPU -> Restart session")"""),
    md("""## 1. Датасет предпочтений

ORPO нужны тройки (prompt, chosen, rejected). Обе стороны генерируются на **одних и тех же** \
записях афиши: `chosen` — тёплый ответ живым языком, `rejected` — сухая справка списком. \
Факты одинаковые, отличается только тон — значит модель учится именно стилю, а не содержанию.

Шаг идёт через OpenAI API и GPU не использует."""),
    code("""!python training/build_preference_data.py --n 120"""),
    md("### Как выглядят пары"),
    code('''import json
from pathlib import Path

pairs = json.loads(Path("data/preference_data.json").read_text(encoding="utf-8"))
print(f"пар: {len(pairs)}\\n")

p = pairs[0]
print("ВОПРОС:", p["prompt"])
print("\\n--- CHOSEN (тёплый) ---")
print(p["chosen"][:600])
print("\\n--- REJECTED (сухой) ---")
print(p["rejected"][:600])'''),
    md("""## 2. Ответы ДО обучения

Снимаем базовый уровень на тех же вопросах, чтобы потом было с чем сравнивать."""),
    code('''from unsloth import FastLanguageModel

BASE_MODEL = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)

TEST_QUESTIONS = [
    "Куда сходить на выходных?",
    "Посоветуй место для свидания",
    "Куда сводить ребенка?",
]

def generate(question: str, max_new_tokens: int = 220) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

FastLanguageModel.for_inference(model)
before = {}
for q in TEST_QUESTIONS:
    before[q] = generate(q)
    print(f"Q: {q}\\nA: {before[q]}\\n{'-' * 70}")'''),
    md("""## 3. Обучение ORPO

Следим не только за падением loss, но и за **`rewards/margins`**: именно рост маржи \
показывает, что модель разводит тёплый и сухой ответы, а не просто подгоняется под оба."""),
    code("""!python training/train_orpo.py --pairs data/preference_data.json --epochs 3"""),
    md("""## 4. Ответы ПОСЛЕ обучения

Подгружаем обученный адаптер и прогоняем те же вопросы."""),
    code('''from peft import PeftModel

tuned, tuned_tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=2048,
    load_in_4bit=True,
)
tuned = PeftModel.from_pretrained(tuned, "outputs/orpo-almaty")
FastLanguageModel.for_inference(tuned)

def generate_tuned(question: str, max_new_tokens: int = 220) -> str:
    messages = [{"role": "user", "content": question}]
    prompt = tuned_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tuned_tokenizer([prompt], return_tensors="pt").to("cuda")
    out = tuned.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tuned_tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

after = {}
for q in TEST_QUESTIONS:
    after[q] = generate_tuned(q)
    print(f"Q: {q}\\nA: {after[q]}\\n{'-' * 70}")'''),
    md("""## 5. Сравнение ДО / ПОСЛЕ

Главный артефакт бонусной части: видно ли, что тон стал теплее."""),
    code('''import pandas as pd

rows = [
    {
        "вопрос": q,
        "ДО": before[q][:200],
        "ПОСЛЕ": after[q][:200],
        "длина ДО": len(before[q]),
        "длина ПОСЛЕ": len(after[q]),
    }
    for q in TEST_QUESTIONS
]

df = pd.DataFrame(rows)
pd.set_option("display.max_colwidth", 200)
display(df)

lines = ["# ORPO: ответы до и после\\n"]
for q in TEST_QUESTIONS:
    lines.append(f"\\n## {q}\\n\\n**До:**\\n\\n{before[q]}\\n\\n**После:**\\n\\n{after[q]}\\n")
Path("orpo_examples.md").write_text("".join(lines), encoding="utf-8")
print("\\nсохранено в orpo_examples.md")'''),
    md("""## 6. Скачать результаты

Файлы исчезнут вместе с сессией Colab — забери их сразу. Сам ноутбук: \
File → Download → Download .ipynb, **после** того как все ячейки отработали."""),
    code('''ARTIFACTS = ["orpo_examples.md", "data/preference_data.json"]

if IN_COLAB:
    from google.colab import files

    for path in ARTIFACTS:
        files.download(path)
else:
    for path in ARTIFACTS:
        p = Path(path)
        print(f"{path}: {'есть' if p.exists() else 'НЕТ'}")'''),
]


def main() -> None:
    notebook = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
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
