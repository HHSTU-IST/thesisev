"""FastAPI application for thesis evaluation."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from thesisev.history import append_history, find_history_entry, read_history
from thesisev.service import evaluate_document, structure_document


class EvaluateRequest(BaseModel):
    """Request body for thesis evaluation."""

    path: str = Field(description="Local path to a txt, md, or docx thesis file.")
    provider: str = Field(default="deepseek", description="LLM provider name.")
    model: str | None = Field(
        default=None, description="Explicit model name for the selected provider."
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature."
    )
    max_tokens: int = Field(
        default=400, ge=64, le=4000, description="Max output tokens for commentary."
    )


class StructureRequest(BaseModel):
    """Request body for structure-only parsing."""

    path: str = Field(description="Local path to a txt, md, or docx thesis file.")


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    ok: bool
    mode: Literal[
        "evaluate",
        "structure",
        "evaluate_upload",
        "evaluate_text",
        "history",
        "history_detail",
    ]
    data: dict[str, Any]


app = FastAPI(
    title="Thesisev API",
    version="0.1.0",
    description="API for structured thesis analysis and multi-model commentary.",
)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Render the built-in thesis evaluation UI."""

    return templates.TemplateResponse(request, "index.html", {})


@app.get("/health")
def health() -> dict[str, str]:
    """Return a simple health status."""

    return {"status": "ok"}


@app.get("/history", response_model=ApiResponse)
def history() -> ApiResponse:
    """Return recent compact evaluation history."""

    return ApiResponse(ok=True, mode="history", data={"items": read_history()})


@app.get("/history/{entry_id}", response_model=ApiResponse)
def history_detail(entry_id: str) -> ApiResponse:
    """Return a single stored history entry."""

    entry = find_history_entry(entry_id)
    if entry is None:
        msg = f"history entry not found: {entry_id}"
        raise HTTPException(status_code=404, detail=msg)
    return ApiResponse(ok=True, mode="history_detail", data=entry)


@app.post("/evaluate", response_model=ApiResponse)
def evaluate(request: EvaluateRequest) -> ApiResponse:
    """Evaluate a thesis document and generate commentary."""

    source = validate_source_path(request.path)
    try:
        result = evaluate_document(
            source,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    append_history(result)
    return ApiResponse(ok=True, mode="evaluate", data=result.to_dict())


@app.post("/structure", response_model=ApiResponse)
def structure(request: StructureRequest) -> ApiResponse:
    """Parse a thesis document and return structured content only."""

    source = validate_source_path(request.path)
    try:
        document = structure_document(source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ApiResponse(ok=True, mode="structure", data=document.to_dict())


@app.post("/evaluate/upload", response_model=ApiResponse)
async def evaluate_upload(
    file: UploadFile = File(...),
    provider: str = Form(default="deepseek"),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.2),
    max_tokens: int = Form(default=400),
) -> ApiResponse:
    """Evaluate an uploaded thesis file."""

    suffix = Path(file.filename or "upload.md").suffix.lower() or ".md"
    validate_suffix(suffix)
    try:
        temp_path = await save_upload_to_tempfile(file, suffix=suffix)
        result = evaluate_document(
            temp_path,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
    append_history(result)
    return ApiResponse(ok=True, mode="evaluate_upload", data=result.to_dict())


@app.post("/evaluate/text", response_model=ApiResponse)
def evaluate_text(
    text: str = Form(...),
    filename: str = Form(default="submission.md"),
    provider: str = Form(default="deepseek"),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.2),
    max_tokens: int = Form(default=400),
) -> ApiResponse:
    """Evaluate thesis content sent as raw text."""

    suffix = Path(filename).suffix.lower() or ".md"
    validate_suffix(suffix)
    try:
        temp_path = save_text_to_tempfile(text=text, suffix=suffix)
        result = evaluate_document(
            temp_path,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink()
    append_history(result)
    return ApiResponse(ok=True, mode="evaluate_text", data=result.to_dict())


def validate_source_path(path: str) -> Path:
    """Validate the incoming local file path."""

    source = Path(path).expanduser()
    if not source.exists():
        msg = f"file not found: {source}"
        raise HTTPException(status_code=404, detail=msg)
    if not source.is_file():
        msg = f"path is not a file: {source}"
        raise HTTPException(status_code=400, detail=msg)
    return source


def validate_suffix(suffix: str) -> None:
    """Validate supported file suffixes."""

    if suffix not in {".txt", ".md", ".docx"}:
        msg = f"unsupported file type: {suffix or 'unknown'}"
        raise HTTPException(status_code=400, detail=msg)


async def save_upload_to_tempfile(file: UploadFile, *, suffix: str) -> Path:
    """Persist an uploaded file into a temporary path."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        return Path(temp_file.name)


def save_text_to_tempfile(*, text: str, suffix: str) -> Path:
    """Persist raw thesis text into a temporary file."""

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        mode="w",
        encoding="utf-8",
    ) as temp_file:
        temp_file.write(text)
        return Path(temp_file.name)


def run_api() -> None:
    """Run the FastAPI app with uvicorn."""

    uvicorn.run("thesisev.api:app", host="127.0.0.1", port=8000, reload=False)
