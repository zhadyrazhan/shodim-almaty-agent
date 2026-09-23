"""FastAPI wrapper around the LlamaIndex agent, serving the chat UI."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent import ask, load_records
from src.config import LLM_BACKEND, SXODIM_DATA

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="Sxodim Almaty Agent")


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


@app.get("/api/health")
def health() -> dict:
    """Surfaces whether the data file is present and which backend is active."""
    try:
        count = len(load_records())
    except FileNotFoundError:
        count = 0
    return {"records": count, "backend": LLM_BACKEND, "ready": count > 0}


@app.post("/api/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос.")

    if not SXODIM_DATA.exists():
        raise HTTPException(
            status_code=503,
            detail="data/sxodim_data.json не найден. Запусти src/scraper.py, затем src/extract.py.",
        )

    try:
        return AskResponse(answer=ask(question))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent failed: {e}")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
