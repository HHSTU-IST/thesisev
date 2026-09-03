"""FastAPI application for thesis evaluation."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from thesisev.analyzers import (
    TopicAnalysis,
    annotate_report_topic_relevance,
    annotate_section_statistics,
    annotate_topic_relevance,
    build_statistics,
    detect_issue_groups,
    extract_keywords,
    extract_technology_details,
    extract_technology_stack,
    split_technology_stack,
)
from thesisev.commentary import generate_comment
from thesisev.deep_review import run_deep_review
from thesisev.llm import ModelConfig, build_model_config
from thesisev.models import EvaluationResult, Issue, ThesisDocument
from thesisev.parser import load_document
from thesisev.paths import config_dir, data_dir, project_root, static_dir, templates_dir
from thesisev.rubric_utils import (
    normalize_format_requirements_payload,
    normalize_rubric_payload,
    normalize_structured_format_requirements,
    parse_score_value,
)
from thesisev.scoring import (
    DEFAULT_THESIS_TECH_RUBRIC,
    FORMAT_RUBRIC_BY_SCORE_RUBRIC,
    build_content_context,
    build_format_standards,
    calculate_score_report,
    sum_format_rule_points,
)
from thesisev.scoring_format import extract_format_rules, normalize_format_spec_payload


class UploadRequestTooLarge(Exception):
    """Raised when an upload request exceeds the configured body limit."""


class EvaluateUploadSizeLimitMiddleware:
    """Limit /evaluate/upload request bodies before multipart parsing."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/evaluate/upload":
            await self.app(scope, receive, send)
            return

        if self.is_oversized_by_header(scope):
            await self.send_too_large_response(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise UploadRequestTooLarge
            return message

        async def limited_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, limited_send)
        except UploadRequestTooLarge:
            if response_started:
                raise
            await self.send_too_large_response(scope, receive, send)

    def is_oversized_by_header(self, scope: Scope) -> bool:
        """Check Content-Length when the client provides one."""

        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if not content_length:
            return False
        try:
            return int(content_length) > self.max_body_bytes
        except ValueError:
            return False

    async def send_too_large_response(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        """Send a standard 413 response."""

        response = JSONResponse(
            status_code=413, content={"detail": build_upload_too_large_message()}
        )
        await response(scope, receive, send)


class EvaluateRequest(BaseModel):
    """Request body for thesis evaluation."""

    path: str = Field(
        description="Local path to a md or docx thesis file under the project root."
    )
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
    preset: str = Field(
        default="thesis_tech",
        description="Built-in rubric preset name, such as thesis_tech or report_iot.",
    )


class StructureRequest(BaseModel):
    """Request body for structure-only parsing."""

    path: str = Field(
        description="Local path to a md or docx thesis file under the project root."
    )


class ApiResponse(BaseModel):
    """Generic API response wrapper."""

    ok: bool
    mode: Literal["evaluate", "structure", "evaluate_upload", "history"]
    data: dict[str, Any]


app = FastAPI(
    title="Thesisev API",
    version="0.1.0",
    description="API for structured thesis analysis and multi-model commentary.",
)
HISTORY_DIR = data_dir()
HISTORY_PATH = HISTORY_DIR / "history.json"
HISTORY_TMP_PATH = HISTORY_DIR / "history.json.tmp"
UPLOAD_DIR = HISTORY_DIR / "uploads"
MAX_HISTORY_ITEMS = 20
MAX_EVALUATE_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_EVALUATE_UPLOAD_BODY_BYTES = MAX_EVALUATE_UPLOAD_BYTES + 1024 * 1024
UPLOAD_READ_CHUNK_BYTES = 1024 * 1024
_HISTORY_LOCK = threading.Lock()
templates = Jinja2Templates(directory=str(templates_dir()))
app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")
app.add_middleware(
    EvaluateUploadSizeLimitMiddleware, max_body_bytes=MAX_EVALUATE_UPLOAD_BODY_BYTES
)

EVALUATE_UPLOAD_FILE_DEFAULT = File(default=None)
EVALUATE_UPLOAD_PRESET_DEFAULT = Form(default="thesis_tech")
EVALUATE_UPLOAD_PROVIDER_DEFAULT = Form(default="deepseek")
EVALUATE_UPLOAD_MODEL_DEFAULT = Form(default=None)
EVALUATE_UPLOAD_TEMPERATURE_DEFAULT = Form(default=0.2)
EVALUATE_UPLOAD_MAX_TOKENS_DEFAULT = Form(default=400)

PRESET_CONFIGS: dict[str, dict[str, str]] = {
    "thesis_tech": {
        "rubric": "score_thesis_tech.json",
        "format": "score_thesis_tech_f.json",
    },
    "report_iot": {
        "rubric": "score_report_iot.json",
        "format": "score_report_iot_f.json",
    },
}

#: In-process background evaluation queue.  Uploads return a ``job_id``
#: immediately and the blocking parse + LLM pipeline runs on a bounded
#: thread pool so the event loop never stalls under concurrent uploads.
EVALUATE_MAX_WORKERS = 2
MAX_JOBS = 20
_EVALUATE_EXECUTOR = ThreadPoolExecutor(
    max_workers=EVALUATE_MAX_WORKERS, thread_name_prefix="thesisev-eval"
)
_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


def register_evaluation_job() -> dict[str, Any]:
    """Create a queued job record and trim finished jobs beyond the cap."""

    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "result": None,
        "error": None,
    }
    with _JOB_LOCK:
        _JOBS[job_id] = record
        finished = [
            key for key, job in _JOBS.items() if job["status"] in {"done", "error"}
        ]
        for key in finished[: max(0, len(_JOBS) - MAX_JOBS)]:
            _JOBS.pop(key, None)
    return record


def get_evaluation_job(job_id: str) -> dict[str, Any] | None:
    """Return a shallow copy of the job record, if present."""

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        return dict(record) if record is not None else None


def update_evaluation_job(job_id: str, **fields: Any) -> None:
    """Update job fields under the registry lock."""

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is not None:
            record.update(fields)


def run_evaluation_job(
    *,
    job_id: str,
    source: Path,
    provider: str,
    model: str | None,
    temperature: float,
    max_tokens: int,
    rubric_filename: str,
    rubric_summary: dict[str, Any] | None,
    format_summary: dict[str, Any] | None,
    preset: str,
) -> None:
    """Run the full evaluation pipeline on a worker thread.

    The uploaded source file is kept until the worker finishes so the
    blocking parse step owns its lifecycle; it is cleaned up afterwards.
    """

    try:
        result = evaluate_document(
            source,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            rubric_filename=rubric_filename,
            rubric=rubric_summary,
            format_requirements=format_summary,
        )
    except Exception as exc:
        update_evaluation_job(job_id, status="error", error=str(exc))
        return
    finally:
        cleanup_upload_file(source)

    payload = result.to_dict()
    if rubric_summary is not None:
        payload["metadata"]["rubric"] = rubric_summary
    if format_summary is not None:
        payload["metadata"]["format_requirements"] = format_summary
    payload["metadata"]["preset"] = preset
    append_history(result)
    update_evaluation_job(job_id, status="done", result=payload)


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
        rubric_filename, format_filename = resolve_preset_files(request.preset)
        rubric_summary = load_builtin_rubric_summary(rubric_filename)
        format_summary = load_builtin_format_requirements_summary(format_filename)
        result = evaluate_document(
            source,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            rubric_filename=rubric_filename,
            rubric=rubric_summary,
            format_requirements=format_summary,
        )
        result.metadata["rubric"] = rubric_summary
        result.metadata["format_requirements"] = format_summary
        result.metadata["preset"] = request.preset
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
    file: UploadFile | None = EVALUATE_UPLOAD_FILE_DEFAULT,
    preset: str = EVALUATE_UPLOAD_PRESET_DEFAULT,
    provider: str = EVALUATE_UPLOAD_PROVIDER_DEFAULT,
    model: str | None = EVALUATE_UPLOAD_MODEL_DEFAULT,
    temperature: float = EVALUATE_UPLOAD_TEMPERATURE_DEFAULT,
    max_tokens: int = EVALUATE_UPLOAD_MAX_TOKENS_DEFAULT,
) -> ApiResponse:
    """Submit an uploaded thesis file for background evaluation.

    The upload is persisted and the blocking pipeline is queued on a bounded
    worker pool; the caller polls ``GET /evaluate/jobs/{job_id}`` for the
    result instead of blocking the event loop for the whole review.
    """

    try:
        rubric_filename, format_filename = resolve_preset_files(preset)
        source = await resolve_thesis_source(file)
        rubric_summary = load_builtin_rubric_summary(rubric_filename)
        format_summary = load_builtin_format_requirements_summary(format_filename)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = register_evaluation_job()
    _EVALUATE_EXECUTOR.submit(
        run_evaluation_job,
        job_id=job["job_id"],
        source=source,
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        rubric_filename=rubric_filename,
        rubric_summary=rubric_summary,
        format_summary=format_summary,
        preset=preset,
    )
    return ApiResponse(
        ok=True,
        mode="evaluate_upload",
        data={"job_id": job["job_id"], "status": job["status"]},
    )


@app.get("/evaluate/jobs/{job_id}", response_model=ApiResponse)
def evaluate_job_status(job_id: str) -> ApiResponse:
    """Return the current status and, once finished, the evaluation result."""

    job = get_evaluation_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="evaluation job not found")
    data: dict[str, Any] = {
        "job_id": job["job_id"],
        "status": job["status"],
        "error": job["error"],
    }
    if job["status"] == "done" and job.get("result") is not None:
        data["result"] = job["result"]
    return ApiResponse(ok=True, mode="evaluate_upload", data=data)


def validate_source_path(path: str) -> Path:
    """Validate the incoming local file path."""

    source = Path(path).expanduser()
    if not source.exists():
        msg = f"file not found: {source}"
        raise HTTPException(status_code=404, detail=msg)
    if not source.is_file():
        msg = f"path is not a file: {source}"
        raise HTTPException(status_code=400, detail=msg)

    resolved_source = source.resolve()
    root = project_root().resolve()
    if resolved_source != root and root not in resolved_source.parents:
        msg = "path must be inside the project root"
        raise HTTPException(status_code=403, detail=msg)
    return resolved_source


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


def resolve_preset_files(preset: str) -> tuple[str, str | None]:
    """Resolve bundled rubric filenames for a preset."""

    config = PRESET_CONFIGS.get(preset)
    if not config:
        msg = f"unknown preset: {preset}"
        raise HTTPException(status_code=400, detail=msg)
    return config["rubric"], config["format"] or None


async def resolve_thesis_source(file: UploadFile | None) -> Path:
    """Resolve the thesis source path from the current upload."""

    if file is None:
        msg = "请上传论文文件"
        raise ValueError(msg)

    filename = file.filename or "upload.md"
    suffix = Path(filename).suffix.lower() or ".md"
    validate_suffix(suffix)
    return await store_upload_file(file, slot="thesis", default_name="upload.md")


def load_builtin_rubric_summary(filename: str) -> dict[str, Any]:
    """Load a bundled rubric summary.

    When the rubric preset has a paired structured format file, its format
    item is appended so the displayed ``total_score`` matches the scoring
    engine's ``raw_total`` (which already includes the format criterion).
    """

    payload = parse_json_file(config_dir() / filename, name="rubric")
    items = normalize_rubric_payload(payload)
    format_filename = FORMAT_RUBRIC_BY_SCORE_RUBRIC.get(filename)
    if format_filename and not any(
        item.get("criterion") == "格式规范" for item in items
    ):
        format_spec = normalize_format_spec_payload(
            parse_json_file(config_dir() / format_filename, name="format rubric")
        )
        rules = extract_format_rules(format_spec)
        items = [
            *items,
            {
                "criterion": "格式规范",
                "key": "format",
                "standard": build_format_standards(rules),
                "score": sum_format_rule_points(rules),
                "evaluation": "local",
            },
        ]
    return {
        "items": items,
        "total_score": round(
            sum(parse_score_value(item["score"]) for item in items), 4
        ),
        "source_name": filename,
    }


def load_builtin_format_requirements_summary(
    filename: str | None,
) -> dict[str, Any] | None:
    """Load a bundled format-requirements summary."""

    if not filename:
        return None
    payload = parse_json_file(config_dir() / filename, name="format requirements")
    if isinstance(payload, dict) and "sections" in payload:
        return normalize_structured_format_requirements(payload, source_name=filename)
    if isinstance(payload, list):
        return normalize_structured_format_requirements(
            {"sections": payload}, source_name=filename
        )
    items = normalize_format_requirements_payload(payload)
    return {"items": items, "item_count": len(items), "source_name": filename}


async def store_upload_file(
    file: UploadFile,
    *,
    slot: str,
    default_name: str,
    validate_json_suffix: bool = False,
) -> Path:
    """Persist the current uploaded file for evaluation."""

    filename = file.filename or default_name
    suffix = Path(filename).suffix.lower() or Path(default_name).suffix.lower()
    if validate_json_suffix:
        validate_rubric_suffix(suffix)

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f".upload_{slot}_{uuid.uuid4().hex}{suffix}.tmp"
    stored_path = UPLOAD_DIR / f"{slot}_{uuid.uuid4().hex}{suffix}"
    written = 0
    try:
        with temp_path.open("wb") as output:
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                next_written = written + len(chunk)
                if next_written > MAX_EVALUATE_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail=build_upload_too_large_message()
                    )
                written = next_written
                output.write(chunk)
        if written == 0:
            msg = f"{slot.replace('_', ' ')} file is empty"
            raise ValueError(msg)

        temp_path.replace(stored_path)
    finally:
        temp_path.unlink(missing_ok=True)

    return stored_path


def cleanup_upload_file(path: Path) -> None:
    """Remove a per-request uploaded file after evaluation."""

    try:
        resolved_path = path.resolve()
        upload_root = UPLOAD_DIR.resolve()
    except OSError:
        return
    if resolved_path == upload_root or upload_root not in resolved_path.parents:
        return
    path.unlink(missing_ok=True)


def build_upload_too_large_message() -> str:
    """Build a consistent upload size-limit error message."""

    max_mib = MAX_EVALUATE_UPLOAD_BYTES // (1024 * 1024)
    return f"uploaded thesis file is too large; maximum size is {max_mib} MiB"


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


def run_api() -> None:
    """Run the FastAPI app with uvicorn."""

    uvicorn.run("thesisev.api:app", host="127.0.0.1", port=8000, reload=False)


def build_topic_analysis(
    document: ThesisDocument, *, rubric_filename: str, rubric: dict[str, Any] | None
) -> TopicAnalysis:
    """Build topic analysis using report rubric standards when available."""

    if rubric_filename.startswith("score_report_"):
        report_rubric = (
            rubric
            if rubric is not None
            else load_builtin_rubric_summary(rubric_filename)
        )
        rubric_items = report_rubric.get("items")
        if not isinstance(rubric_items, list) or not rubric_items:
            msg = "report rubric items must be a non-empty list"
            raise ValueError(msg)
        topic_analysis = annotate_report_topic_relevance(document, rubric_items)
        topic_analysis["source"] = rubric_filename
        return topic_analysis
    return annotate_topic_relevance(document)


def evaluate_document(
    path: str | Path,
    *,
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
    timeout: int = 60,
    model_config: ModelConfig | None = None,
    rubric_filename: str = DEFAULT_THESIS_TECH_RUBRIC,
    rubric: dict[str, Any] | None = None,
    format_requirements: dict[str, Any] | None = None,
) -> EvaluationResult:
    """Evaluate a thesis document from a local path."""

    document = load_document(path)
    topic_analysis = build_topic_analysis(
        document, rubric_filename=rubric_filename, rubric=rubric
    )
    statistics = build_statistics(document, topic_analysis=dict(topic_analysis))
    issue_groups = detect_issue_groups(document)
    format_issues = issue_groups.format_issues
    writing_issues = issue_groups.writing_issues
    issues = issue_groups.all_issues
    keywords = extract_keywords(document)
    technology_details = extract_technology_details(document)
    technology_stack = extract_technology_stack(document)
    technology_groups = split_technology_stack(technology_details)
    content_context = build_content_context(
        document=document,
        topic_analysis=dict(topic_analysis),
        keywords=keywords,
        technology_details=technology_details,
    )
    runtime_model_config = model_config or build_model_config(
        provider=provider,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    score_report = calculate_score_report(
        document=document,
        topic_analysis=dict(topic_analysis),
        format_issues=format_issues,
        writing_issues=writing_issues,
        content_context=content_context,
        keywords=keywords,
        technology_details=technology_details,
        format_requirements=format_requirements,
        rubric=rubric,
        rubric_filename=rubric_filename,
        model_config=runtime_model_config,
    )
    score = score_report.score
    comment, comment_checks, comment_source = generate_comment(
        title=document.title,
        keywords=keywords,
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        root_sections=document.root_sections,
        model_config=runtime_model_config,
    )
    if runtime_model_config.is_available():
        deep_issues = run_deep_review(
            document=document, model_config=runtime_model_config, existing_issues=issues
        )
        if deep_issues:
            issues = [*issues, *deep_issues]
    else:
        deep_issues: list[Issue] = []
    return EvaluationResult(
        document=document,
        statistics=statistics,
        issues=issues,
        format_issues=format_issues,
        writing_issues=writing_issues,
        content_context=content_context,
        keywords=keywords,
        technology_stack=technology_stack,
        software_technology_stack=technology_groups["software_technology_stack"],
        hardware_technology_stack=technology_groups["hardware_technology_stack"],
        technology_details=technology_details,
        topic_keywords=topic_analysis["topic_keywords"],
        topic_relevance_ratio=topic_analysis["document_ratio"],
        score=score,
        comment=comment,
        comment_checks=comment_checks,
        metadata={
            "version": "0.1.0",
            "topic_analysis": topic_analysis,
            "score_source": score_report.score_source,
            "score_detail": score_report.to_dict(),
            "comment_source": comment_source,
            "rubric": rubric,
            "format_requirements": format_requirements,
            "evaluation_roles": {
                "format_detection": "local",
                "format_evaluation": "local",
                "writing_detection": "local",
                "writing_evaluation": "local",
                "logic_detection": "local",
                "logic_deep_review": (
                    "llm" if runtime_model_config.is_available() else "skipped"
                ),
                "content_evaluation": comment_source,
                "llm_input": "content_context_only",
            },
            "model": runtime_model_config.to_metadata(),
            "deep_review": {
                "source": "llm" if runtime_model_config.is_available() else "skipped",
                "added_count": len(deep_issues),
            },
        },
    )


def structure_document(path: str | Path) -> ThesisDocument:
    """Load a thesis document and return only its structured representation."""

    document = load_document(path)
    annotate_section_statistics(document)
    annotate_topic_relevance(document)
    return document


def append_history(result: EvaluationResult) -> None:
    """Append a compact evaluation summary to local history.

    Reads, inserts, and writes happen under a process-wide lock, and the file
    is replaced atomically via ``os.replace``, so concurrent evaluations and
    multi-worker deployments cannot drop or corrupt entries.
    """

    with _HISTORY_LOCK:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        items = read_history_unlocked()
        items.insert(0, build_history_entry(result))
        HISTORY_TMP_PATH.write_text(
            json.dumps(items[:MAX_HISTORY_ITEMS], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        HISTORY_TMP_PATH.replace(HISTORY_PATH)


def read_history() -> list[dict[str, Any]]:
    """Read stored history entries from disk."""

    with _HISTORY_LOCK:
        return read_history_unlocked()


def read_history_unlocked() -> list[dict[str, Any]]:
    """Read history entries without acquiring the history lock."""

    if not HISTORY_PATH.exists():
        return []
    return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))


def build_history_entry(result: EvaluationResult) -> dict[str, Any]:
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
