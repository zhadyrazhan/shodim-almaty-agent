# Sxodim Almaty Agent

A conversational agent that recommends things to do in Almaty, grounded in live listings scraped from [sxodim.com](https://sxodim.com/almaty).

Ask it *"where should I go this weekend?"* or *"suggest a date spot"* and it answers from the current afisha — not from model memory.

**Pipeline:** sxodim.com → Jina Reader → GPT-5-mini (structuring) → LlamaIndex RAG → agent → web UI

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Scraping | **Jina Reader** (`r.jina.ai`) | Returns clean markdown for any URL — no CSS selectors to break when the site changes its markup |
| Structuring | **GPT-5-mini** + Pydantic structured output | Typed records straight out of noisy markdown, no regex parsing |
| Retrieval | **LlamaIndex** `VectorStoreIndex` | Small dataset, in-memory index, no vector DB to operate |
| Generation | **GPT-5-mini** (default) or **Qwen via Ollama** | Swap with one env var; the Ollama path needs no API key at all |
| UI | FastAPI + vanilla JS | Dark-mode chat, no build step |

### Two interchangeable backends

```bash
# default — OpenAI
OPENAI_API_KEY=sk-...

# local, free, no API key (embeddings run locally too)
LLM_BACKEND=ollama
OLLAMA_MODEL=qwen3:14b
```

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY

python -m src.scraper         # sxodim.com -> data/raw/*.md
python -m src.extract         # -> data/sxodim_data.json
python -m src.agent           # sanity-check a few questions in the terminal

uvicorn webapp.server:app --reload   # open http://127.0.0.1:8000
```

The scrape and extract steps are separate on purpose: extraction re-reads the cached markdown in `data/raw/`, so you can iterate on the extraction prompt without re-hitting the network.

---

## Running on Colab

Two things are worth doing in Colab: running the notebook (if you don't want a local Python setup), and the ORPO training (which needs a GPU).

### The notebook

Runtime → **CPU is fine**. The notebook imports from `src/`, so clone the repo first rather than uploading the `.ipynb` alone:

```python
!git clone https://github.com/zhadyrazhan/shodim-almaty-agent.git
%cd shodim-almaty-agent
!pip install -q -r requirements.txt
```

Add your key via the Secrets panel (🔑 in the left sidebar) as `OPENAI_API_KEY`, then:

```python
import os
from google.colab import userdata
os.environ["OPENAI_API_KEY"] = userdata.get("OPENAI_API_KEY")
```

Now Runtime → Run all. Scraping and extraction take a few minutes (57 chunks through the API). Download `sxodim_agent.ipynb` **with outputs saved** plus `data/sxodim_data.json` and `agent_examples.md`.

### ORPO training (bonus)

Section 5 of `sxodim_agent.ipynb` covers the whole bonus track: preference pairs → training → before/after comparison. The cells self-skip when no GPU is present, so sections 1–4 still run on CPU.

Runtime → Change runtime type → **T4 GPU**, then Runtime → **Restart session**. Changing the type alone doesn't move you onto GPU hardware; the notebook's second cell fails loudly if you skip the restart, because Colab silently hands you CPU when you're over quota.

Then Runtime → Run all. Watch that the loss falls **and** `rewards/margins` grows — margins are what tell you the model is separating the friendly answer from the curt one, rather than just fitting both.

To run the pieces by hand instead:

```python
!python training/build_preference_data.py --n 120   # API only, no GPU
!python training/train_orpo.py --pairs data/preference_data.json --epochs 3
```

To serve the result, export the merged model to GGUF, import it into Ollama, and point the agent at it:

```bash
LLM_BACKEND=ollama OLLAMA_MODEL=qwen-almaty uvicorn webapp.server:app
```

---

## How it works

**1. Scrape** (`src/scraper.py`) — pulls four sxodim.com pages (afisha, weekend, places, main) through Jina Reader and caches the markdown under `data/raw/`.

**2. Structure** (`src/extract.py`) — the raw pages are mostly navigation, ad blocks and repeated links. Each page is chunked and passed to GPT-5-mini with a Pydantic schema, so the model returns typed `Item` records rather than free text:

```python
class Item(BaseModel):
    title: str
    kind: Literal["event", "place", "restaurant", "entertainment"]
    description: str
    category: str
    url: str
    date: str
    price: str
    address: str
    good_for: list[str]    # "свидание", "дети", "компания друзей", ...
```

Records are deduplicated by URL, since the same event appears on several pages.

**3. Retrieve + answer** (`src/agent.py`) — each record is flattened into a searchable document, indexed with LlamaIndex, and queried with a system prompt that tells the model to recommend only what's in the listings and to say so when nothing fits. The `good_for` field is what makes *"suggest a date spot"* and *"where to take a kid"* return meaningfully different results.

---

## Web UI

Dark mode, no theme switching.

- Clickable starter questions from the project brief
- Typing indicator, graceful in-chat error messages
- Responsive down to phone width

`GET /api/health` reports record count and active backend if you want to check the server without opening a browser.

---

## Project structure

```
├── sxodim_agent.ipynb       # required deliverable, imports from src/
├── agent_examples.md        # 5+ example dialogues
├── data/
│   ├── sxodim_data.json     # structured listings
│   └── raw/                 # cached markdown (gitignored)
├── src/
│   ├── config.py            # all tunables
│   ├── scraper.py           # Jina Reader
│   ├── extract.py           # markdown -> typed records
│   └── agent.py             # LlamaIndex RAG
├── evals/
│   └── run_eval.py          # golden-set eval harness
├── tests/                   # pytest suite
├── training/
│   ├── build_preference_data.py
│   └── train_orpo.py        # ORPO bonus
├── scripts/
│   ├── build_notebook.py    # regenerates the notebook from source
│   └── restore_outputs.py   # re-attaches saved outputs after regeneration
└── webapp/
    ├── server.py
    └── static/              # dark-mode chat UI
```

The notebook imports from `src/` rather than duplicating logic, and `scripts/build_notebook.py` regenerates its skeleton — so the `.py` and `.ipynb` can't drift apart.

---

## Project brief mapping

Задача #2 requirements from `Проект.pdf`:

| Requirement | Where |
|---|---|
| Сайт спарсен | `src/scraper.py` → `data/raw/*.md` |
| Данные структурированы (JSON) | `src/extract.py` → `data/sxodim_data.json` |
| Агент отвечает на вопросы | `src/agent.py`, notebook section 3 |
| Код запускается, логика понятна | Quickstart above |
| Бонус: ORPO / дружелюбность | `training/`, notebook section 5 |

---

## Branches

Work was split so each piece could be reviewed on its own, then merged into `main` via PRs:

| Branch | Contents |
|---|---|
| `feat/tests` | pytest suite |
| `feat/evals` | golden set + eval harness |
| `feat/sft-orpo` | ORPO fine-tuning for friendlier answers (bonus) |

All three are merged; `main` has everything.

---

## Tests and evals

```bash
pytest                    # 27 unit tests, no API calls
python evals/run_eval.py  # golden set — needs the agent running
```

The golden set deliberately avoids asserting specific event names, since listings change daily. It checks behavior that must hold regardless: answers come from the data, the audience changes the answer (no bars for the kid question), and off-topic questions get declined.
