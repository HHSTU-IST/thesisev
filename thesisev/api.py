"""FastAPI application for thesis evaluation."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from thesisev.analyzers import (
    annotate_section_statistics,
    annotate_topic_relevance,
    build_statistics,
    calculate_score,
    detect_issues,
    extract_keywords,
    extract_technology_details,
    extract_technology_stack,
)
from thesisev.commentary import generate_comment
from thesisev.llm import ModelConfig, build_model_config
from thesisev.models import EvaluationResult, ThesisDocument
from thesisev.parser import load_document


class EvaluateRequest(BaseModel):
    """Request body for thesis evaluation."""

    path: str = Field(description="Local path to a md or docx thesis file.")
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

    path: str = Field(description="Local path to a md or docx thesis file.")


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    ok: bool
    mode: Literal[
        "evaluate",
        "structure",
        "evaluate_upload",
        "history",
    ]
    data: dict[str, Any]


app = FastAPI(
    title="Thesisev API",
    version="0.1.0",
    description="API for structured thesis analysis and multi-model commentary.",
)
BASE_DIR = Path(__file__).resolve().parent
HISTORY_DIR = BASE_DIR / "data"
HISTORY_PATH = HISTORY_DIR / "history.json"
MAX_HISTORY_ITEMS = 20
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

    if suffix not in {".md", ".docx"}:
        msg = f"unsupported file type: {suffix or 'unknown'}"
        raise HTTPException(status_code=400, detail=msg)


async def save_upload_to_tempfile(file: UploadFile, *, suffix: str) -> Path:
    """Persist an uploaded file into a temporary path."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        return Path(temp_file.name)


def run_api() -> None:
    """Run the FastAPI app with uvicorn."""

    uvicorn.run("thesisev.api:app", host="127.0.0.1", port=8000, reload=False)


def evaluate_document(
    path: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
    timeout: int = 60,
    model_config: ModelConfig | None = None,
) -> EvaluationResult:
    """Evaluate a thesis document from a local path."""

    document = load_document(path)
    annotate_section_statistics(document)
    topic_analysis = annotate_topic_relevance(document)
    statistics = build_statistics(document)
    issues = detect_issues(document)
    keywords = extract_keywords(document)
    technology_details = extract_technology_details(document)
    technology_stack = extract_technology_stack(document)
    score = calculate_score(issues, len(document.sections))
    runtime_model_config = model_config or build_model_config(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    comment, comment_checks = generate_comment(
        title=document.title,
        keywords=keywords,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        issues=issues,
        root_sections=document.root_sections,
        model_config=runtime_model_config,
    )
    return EvaluationResult(
        document=document,
        statistics=statistics,
        issues=issues,
        keywords=keywords,
        technology_stack=technology_stack,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        comment=comment,
        comment_checks=comment_checks,
        metadata={
            "version": "0.1.0",
            "topic_analysis": topic_analysis,
            "model": runtime_model_config.to_metadata(),
        },
    )


def structure_document(path: str | Path) -> ThesisDocument:
    """Load a thesis document and return only its structured representation."""

    document = load_document(path)
    annotate_section_statistics(document)
    annotate_topic_relevance(document)
    return document


def append_history(result) -> None:
    """Append a compact evaluation summary to local history."""

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    items = read_history()
    items.insert(0, build_history_entry(result))
    HISTORY_PATH.write_text(
        json.dumps(items[:MAX_HISTORY_ITEMS], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_history() -> list[dict[str, Any]]:
    """Read stored history entries from disk."""

    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def build_history_entry(result) -> dict[str, Any]:
    """Build a compact serialized history entry."""

    return {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(tz=datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "title": result.document.title,
        "source_type": result.document.source_type,
        "score": result.score,
        "comment": result.comment,
        "issue_count": len(result.issues),
        "topic_relevance_ratio": result.topic_relevance_ratio,
        "technology_stack": result.technology_stack,
        "model": result.metadata.get("model", {}),
    }
