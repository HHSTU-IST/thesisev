"""FastAPI application for thesis evaluation."""

from __future__ import annotations

import json
import tempfile
import uuid
from datetime import UTC, datetime
from json import JSONDecodeError
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
from thesisev.paths import data_dir, static_dir, templates_dir


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
        "last_upload",
    ]
    data: dict[str, Any]


app = FastAPI(
    title="Thesisev API",
    version="0.1.0",
    description="API for structured thesis analysis and multi-model commentary.",
)
HISTORY_DIR = data_dir()
HISTORY_PATH = HISTORY_DIR / "history.json"
UPLOAD_DIR = HISTORY_DIR / "uploads"
LAST_UPLOAD_PATH = HISTORY_DIR / "last_upload.json"
MAX_HISTORY_ITEMS = 20
templates = Jinja2Templates(directory=str(templates_dir()))
app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")


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


@app.get("/last-upload", response_model=ApiResponse)
def last_upload() -> ApiResponse:
    """Return the last uploaded thesis, rubric, and format files."""

    return ApiResponse(
        ok=True,
        mode="last_upload",
        data={"items": build_public_last_upload_manifest()},
    )


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
    file: UploadFile | None = File(default=None),
    rubric_file: UploadFile | None = File(default=None),
    format_file: UploadFile | None = File(default=None),
    provider: str = Form(default="deepseek"),
    model: str | None = Form(default=None),
    temperature: float = Form(default=0.2),
    max_tokens: int = Form(default=400),
) -> ApiResponse:
    """Evaluate an uploaded thesis file."""

    try:
        result = evaluate_document(
            await resolve_thesis_source(file),
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        rubric_summary = await resolve_rubric_summary(rubric_file)
        if rubric_summary is not None:
            result.metadata["rubric"] = rubric_summary
        format_summary = await resolve_format_requirements_summary(format_file)
        if format_summary is not None:
            result.metadata["format_requirements"] = format_summary
        result.metadata["last_upload"] = build_public_last_upload_manifest()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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


def validate_rubric_suffix(suffix: str) -> None:
    """Validate supported rubric file suffixes."""

    if suffix != ".json":
        msg = f"unsupported rubric file type: {suffix or 'unknown'}"
        raise HTTPException(status_code=400, detail=msg)


async def save_upload_to_tempfile(file: UploadFile, *, suffix: str) -> Path:
    """Persist an uploaded file into a temporary path."""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        content = await file.read()
        temp_file.write(content)
        return Path(temp_file.name)


async def resolve_thesis_source(file: UploadFile | None) -> Path:
    """Resolve the thesis source path from the current or last upload."""

    if file is not None:
        filename = file.filename or "upload.md"
        suffix = Path(filename).suffix.lower() or ".md"
        validate_suffix(suffix)
        return await store_upload_file(file, slot="thesis", default_name="upload.md")

    return load_last_upload_path(
        slot="thesis",
        missing_message="请先上传论文文件，之后系统会自动复用上次上传内容",
    )


async def resolve_rubric_summary(file: UploadFile | None) -> dict[str, Any] | None:
    """Resolve rubric data from the current or last upload."""

    if file is not None:
        stored_path = await store_upload_file(
            file,
            slot="rubric",
            default_name="rubric.json",
            validate_json_suffix=True,
        )
        return parse_rubric_file(
            stored_path, source_name=file.filename or "rubric.json"
        )

    stored_path = load_last_upload_path(slot="rubric", missing_message="")
    if not stored_path:
        return None
    entry = read_last_upload_manifest().get("rubric", {})
    return parse_rubric_file(
        stored_path, source_name=entry.get("filename", "rubric.json")
    )


async def resolve_format_requirements_summary(
    file: UploadFile | None,
) -> dict[str, Any] | None:
    """Resolve format requirements data from the current or last upload."""

    if file is not None:
        stored_path = await store_upload_file(
            file,
            slot="format_requirements",
            default_name="format_requirements.json",
            validate_json_suffix=True,
        )
        return parse_format_requirements_file(
            stored_path,
            source_name=file.filename or "format_requirements.json",
        )

    stored_path = load_last_upload_path(slot="format_requirements", missing_message="")
    if not stored_path:
        return None
    entry = read_last_upload_manifest().get("format_requirements", {})
    return parse_format_requirements_file(
        stored_path,
        source_name=entry.get("filename", "format_requirements.json"),
    )


async def store_upload_file(
    file: UploadFile,
    *,
    slot: str,
    default_name: str,
    validate_json_suffix: bool = False,
) -> Path:
    """Persist an uploaded file for reuse across page refreshes."""

    filename = file.filename or default_name
    suffix = Path(filename).suffix.lower() or Path(default_name).suffix.lower()
    if validate_json_suffix:
        validate_rubric_suffix(suffix)
    content = await file.read()
    if not content:
        msg = f"{slot.replace('_', ' ')} file is empty"
        raise ValueError(msg)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for existing in UPLOAD_DIR.glob(f"last_{slot}.*"):
        existing.unlink()

    stored_path = UPLOAD_DIR / f"last_{slot}{suffix}"
    stored_path.write_bytes(content)
    update_last_upload_manifest(slot=slot, filename=filename, stored_path=stored_path)
    return stored_path


def load_last_upload_path(slot: str, *, missing_message: str) -> Path | None:
    """Load a persisted upload path when available."""

    entry = read_last_upload_manifest().get(slot)
    if not entry:
        if missing_message:
            raise ValueError(missing_message)
        return None

    stored_path = resolve_upload_manifest_path(slot, entry)
    if not stored_path.exists():
        if missing_message:
            raise ValueError(missing_message)
        return None
    return stored_path


async def parse_rubric_upload(file: UploadFile | None) -> dict[str, Any] | None:
    """Parse an optional rubric JSON upload into a normalized summary."""

    if file is None:
        return None

    suffix = Path(file.filename or "rubric.json").suffix.lower() or ".json"
    validate_rubric_suffix(suffix)
    payload = await parse_json_upload_payload(file, name="rubric")
    items = normalize_rubric_payload(payload)
    return {
        "items": items,
        "total_score": round(sum(item["score"] for item in items), 4),
        "source_name": file.filename or "rubric.json",
    }


def parse_rubric_file(path: Path, *, source_name: str) -> dict[str, Any]:
    """Parse a stored rubric JSON file."""

    payload = parse_json_file(path, name="rubric")
    items = normalize_rubric_payload(payload)
    return {
        "items": items,
        "total_score": round(sum(item["score"] for item in items), 4),
        "source_name": source_name,
    }


async def parse_format_requirements_upload(
    file: UploadFile | None,
) -> dict[str, Any] | None:
    """Parse an optional format-requirements JSON upload."""

    if file is None:
        return None

    suffix = Path(file.filename or "format_requirements.json").suffix.lower() or ".json"
    validate_rubric_suffix(suffix)
    payload = await parse_json_upload_payload(file, name="format requirements")
    items = normalize_format_requirements_payload(payload)
    return {
        "items": items,
        "item_count": len(items),
        "source_name": file.filename or "format_requirements.json",
    }


def parse_format_requirements_file(path: Path, *, source_name: str) -> dict[str, Any]:
    """Parse a stored format-requirements JSON file."""

    payload = parse_json_file(path, name="format requirements")
    items = normalize_format_requirements_payload(payload)
    return {
        "items": items,
        "item_count": len(items),
        "source_name": source_name,
    }


async def parse_json_upload_payload(file: UploadFile, *, name: str) -> Any:
    """Parse a UTF-8 JSON upload payload."""

    try:
        return json.loads((await file.read()).decode("utf-8"))
    except UnicodeDecodeError as exc:
        msg = f"{name} JSON must be utf-8 encoded"
        raise ValueError(msg) from exc
    except JSONDecodeError as exc:
        msg = f"{name} JSON is invalid"
        raise ValueError(msg) from exc


def parse_json_file(path: Path, *, name: str) -> Any:
    """Parse a persisted UTF-8 JSON file."""

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except UnicodeDecodeError as exc:
        msg = f"{name} JSON must be utf-8 encoded"
        raise ValueError(msg) from exc
    except JSONDecodeError as exc:
        msg = f"{name} JSON is invalid"
        raise ValueError(msg) from exc


def normalize_rubric_payload(payload: Any) -> list[dict[str, Any]]:
    """Normalize rubric JSON into criterion-score items."""

    if isinstance(payload, dict):
        return [
            {
                "criterion": parse_rubric_criterion(criterion),
                "score": parse_rubric_score(score),
            }
            for criterion, score in payload.items()
        ]

    if isinstance(payload, list):
        items: list[dict[str, Any]] = []
        for entry in payload:
            if not isinstance(entry, dict) or len(entry) != 1:
                msg = "rubric list entries must be single-key objects"
                raise ValueError(msg)
            criterion, score = next(iter(entry.items()))
            items.append(
                {
                    "criterion": parse_rubric_criterion(criterion),
                    "score": parse_rubric_score(score),
                }
            )
        return items

    msg = "rubric JSON must be an object or a list of single-key objects"
    raise ValueError(msg)


def normalize_format_requirements_payload(payload: Any) -> list[dict[str, str]]:
    """Normalize format-requirements JSON into display-friendly items."""

    if isinstance(payload, dict):
        return [
            {
                "label": parse_format_requirement_label(label),
                "value": stringify_format_requirement_value(value),
            }
            for label, value in payload.items()
        ]

    if isinstance(payload, list):
        items: list[dict[str, str]] = []
        for index, entry in enumerate(payload, start=1):
            if isinstance(entry, str):
                items.append({"label": f"要求 {index}", "value": entry.strip()})
                continue
            if isinstance(entry, dict) and len(entry) == 1:
                label, value = next(iter(entry.items()))
                items.append(
                    {
                        "label": parse_format_requirement_label(label),
                        "value": stringify_format_requirement_value(value),
                    }
                )
                continue
            items.append(
                {
                    "label": f"要求 {index}",
                    "value": stringify_format_requirement_value(entry),
                }
            )
        if items:
            return items
        msg = "format requirements JSON list must not be empty"
        raise ValueError(msg)

    msg = "format requirements JSON must be an object or a list"
    raise ValueError(msg)


def parse_rubric_score(score: Any) -> float:
    """Parse a rubric score as a numeric value."""

    if isinstance(score, bool):
        msg = "rubric score must be numeric"
        raise ValueError(msg)
    if isinstance(score, int | float):
        return float(score)
    msg = "rubric score must be numeric"
    raise ValueError(msg)


def parse_rubric_criterion(criterion: Any) -> str:
    """Parse a rubric criterion as non-empty text."""

    if not isinstance(criterion, str) or not criterion.strip():
        msg = "rubric criterion must be a non-empty string"
        raise ValueError(msg)
    return criterion.strip()


def parse_format_requirement_label(label: Any) -> str:
    """Parse a format-requirements item label."""

    if not isinstance(label, str) or not label.strip():
        msg = "format requirement label must be a non-empty string"
        raise ValueError(msg)
    return label.strip()


def stringify_format_requirement_value(value: Any) -> str:
    """Convert a format-requirements value into compact display text."""

    if isinstance(value, str):
        text = value.strip()
        if not text:
            msg = "format requirement value must not be empty"
            raise ValueError(msg)
        return text
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, list | dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


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
        "created_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "title": result.document.title,
        "source_type": result.document.source_type,
        "score": result.score,
        "comment": result.comment,
        "issue_count": len(result.issues),
        "topic_relevance_ratio": result.topic_relevance_ratio,
        "technology_stack": result.technology_stack,
        "model": result.metadata.get("model", {}),
    }


def read_last_upload_manifest() -> dict[str, dict[str, Any]]:
    """Read the last-upload manifest from disk."""

    if not LAST_UPLOAD_PATH.exists():
        return {}
    manifest = json.loads(LAST_UPLOAD_PATH.read_text(encoding="utf-8"))
    normalized_manifest = normalize_last_upload_manifest(manifest)
    if normalized_manifest != manifest:
        LAST_UPLOAD_PATH.write_text(
            json.dumps(normalized_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return normalized_manifest


def update_last_upload_manifest(*, slot: str, filename: str, stored_path: Path) -> None:
    """Update the persisted metadata for the last uploaded file in a slot."""

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    manifest = read_last_upload_manifest()
    manifest[slot] = {
        "filename": filename,
        "stored_path": str(stored_path),
        "updated_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        "size": stored_path.stat().st_size,
        "suffix": stored_path.suffix.lower(),
    }
    LAST_UPLOAD_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_public_last_upload_manifest() -> dict[str, dict[str, Any] | None]:
    """Build a safe manifest for UI display."""

    manifest = read_last_upload_manifest()
    public_manifest: dict[str, dict[str, Any] | None] = {}
    for slot in ("thesis", "rubric", "format_requirements"):
        entry = manifest.get(slot)
        if not entry:
            public_manifest[slot] = None
            continue
        stored_path = resolve_upload_manifest_path(slot, entry)
        if not stored_path.exists():
            public_manifest[slot] = None
            continue
        public_manifest[slot] = {
            "filename": entry["filename"],
            "updated_at": entry["updated_at"],
            "size": entry["size"],
            "suffix": entry["suffix"],
            "available": True,
        }
    return public_manifest


def normalize_last_upload_manifest(
    manifest: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Normalize persisted upload paths after layout changes."""

    normalized: dict[str, dict[str, Any]] = {}
    for slot, entry in manifest.items():
        normalized_entry = dict(entry)
        normalized_entry["stored_path"] = str(resolve_upload_manifest_path(slot, entry))
        normalized[slot] = normalized_entry
    return normalized


def resolve_upload_manifest_path(slot: str, entry: dict[str, Any]) -> Path:
    """Resolve the current stored path for a manifest entry."""

    stored_path = Path(entry["stored_path"])
    if stored_path.exists():
        return stored_path

    suffix = entry.get("suffix", stored_path.suffix)
    candidate = UPLOAD_DIR / f"last_{slot}{suffix}"
    return candidate
