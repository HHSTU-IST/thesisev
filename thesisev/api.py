"""FastAPI application for thesis evaluation."""

from __future__ import annotations

import json
import logging
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
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from thesisev.llm import ENGINE_LOCAL, build_model_config, resolve_runtime
from thesisev.models import EvaluationResult
from thesisev.paths import data_dir, project_root, static_dir, templates_dir
from thesisev.resources import load_json_resource
from thesisev.rubric_utils import (
    FORMAT_RUBRIC_BY_SCORE_RUBRIC,
    FORMAT_RUBRIC_KEY,
    FORMAT_RUBRIC_NAME,
    normalize_format_requirements_payload,
    normalize_rubric_payload,
    normalize_structured_format_requirements,
    parse_score_value,
)
from thesisev.scoring import build_format_standards, sum_format_rule_points
from thesisev.scoring_format import extract_format_rules, normalize_format_spec_payload
from thesisev.service import (
    evaluate_document,
    resolve_preset_files,
    structure_document,
)
from thesisev.settings import get_int

logger = logging.getLogger(__name__)


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
    engine: str = Field(
        default=ENGINE_LOCAL,
        description=(
            "Inference engine: 'local' keeps the document on this host, "
            "'remote' sends it to the DeepSeek API."
        ),
    )
    provider: str | None = Field(
        default=None, description="LLM provider name, overriding the engine default."
    )
    model: str | None = Field(
        default=None, description="Explicit model name for the selected provider."
    )
    temperature: float = Field(
        default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature."
    )
    max_tokens: int | None = Field(
        default=None,
        ge=64,
        le=4000,
        description="Max output tokens; omitted selects the provider default.",
    )
    preset: str = Field(
        default="thesis_tech",
        description="Built-in rubric preset name; currently only thesis_tech.",
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
MAX_HISTORY_ITEMS = get_int("api", "max_history_items")
MAX_EVALUATE_UPLOAD_BYTES = get_int("api", "max_upload_bytes")
#: The multipart envelope adds framing around the file itself, so the raw
#: request-body ceiling is the file cap plus a fixed allowance.  This overhead
#: is protocol framing rather than a tuning knob, so it stays in code.
UPLOAD_BODY_OVERHEAD_BYTES = 1024 * 1024
MAX_EVALUATE_UPLOAD_BODY_BYTES = MAX_EVALUATE_UPLOAD_BYTES + UPLOAD_BODY_OVERHEAD_BYTES
UPLOAD_READ_CHUNK_BYTES = get_int("api", "upload_read_chunk_bytes")
_HISTORY_LOCK = threading.Lock()
templates = Jinja2Templates(directory=str(templates_dir()))
app.mount("/static", StaticFiles(directory=str(static_dir())), name="static")
app.add_middleware(
    EvaluateUploadSizeLimitMiddleware, max_body_bytes=MAX_EVALUATE_UPLOAD_BODY_BYTES
)

EVALUATE_UPLOAD_FILE_DEFAULT = File(default=None)
EVALUATE_UPLOAD_PRESET_DEFAULT = Form(default="thesis_tech")
EVALUATE_UPLOAD_ENGINE_DEFAULT = Form(default=ENGINE_LOCAL)
EVALUATE_UPLOAD_PROVIDER_DEFAULT = Form(default=None)
EVALUATE_UPLOAD_MODEL_DEFAULT = Form(default=None)
EVALUATE_UPLOAD_TEMPERATURE_DEFAULT = Form(default=0.2)
EVALUATE_UPLOAD_MAX_TOKENS_DEFAULT = Form(default=None)

#: In-process background evaluation queue.  Uploads return a ``job_id``
#: immediately and the blocking parse + LLM pipeline runs on a bounded
#: thread pool so the event loop never stalls under concurrent uploads.
#:
#: KNOWN LIMITATION: ``_JOBS`` lives only in process memory and the queue is
#: designed for single-process use.  A restart drops every in-flight job (it
#: never reaches a terminal state, so clients must resubmit), and a crash can
#: skip :func:`cleanup_upload_file` in the worker's ``finally`` and leave a
#: stray file under ``uploads/``.  A deliberate simplicity trade-off: durable
#: jobs across restarts would need a persistence layer this project omits.
EVALUATE_MAX_WORKERS = get_int("api", "evaluate_max_workers")
MAX_JOBS = get_int("api", "max_jobs")
JOB_TIMEOUT_SECONDS = get_int("api", "job_timeout_seconds")
_EVALUATE_EXECUTOR = ThreadPoolExecutor(
    max_workers=EVALUATE_MAX_WORKERS, thread_name_prefix="thesisev-eval"
)
_JOB_LOCK = threading.Lock()
_JOBS: dict[str, dict[str, Any]] = {}


#: Statuses that mean a job will never change again; only these are eligible
#: for capacity eviction, so an accepted job stays pollable until it finishes.
TERMINAL_JOB_STATUSES = frozenset({"done", "error"})


def register_evaluation_job() -> dict[str, Any] | None:
    """Create a queued job record, or reject the submission when saturated.

    ``_JOBS`` is bounded by :data:`MAX_JOBS`, enforced without ever losing an
    in-flight evaluation:

    * Back-pressure: when ``MAX_JOBS`` records are already non-terminal
      (queued/running) the registry is saturated and this returns ``None`` so
      the caller can answer ``429`` instead of accepting a request whose
      result could never be retained.  Every accepted job maps to exactly one
      executor task, so this also caps the unbounded ``ThreadPoolExecutor``
      pending queue at :data:`MAX_JOBS`.
    * Capacity eviction: finished (``done``/``error``) records are dropped,
      oldest first, to make room for a newly accepted job.  Non-terminal jobs
      are never evicted, so a job that returned a ``job_id`` stays pollable
      until it reaches a terminal state.

    Returns a snapshot of the new record (never the shared live dict) so the
    caller cannot observe a worker mutating it outside the lock.
    """

    reap_stale_jobs()
    job_id = uuid.uuid4().hex
    record = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "result": None,
        "error": None,
    }
    with _JOB_LOCK:
        live = sum(
            1 for job in _JOBS.values() if job["status"] not in TERMINAL_JOB_STATUSES
        )
        if live >= MAX_JOBS:
            return None
        _JOBS[job_id] = record
        overflow = len(_JOBS) - MAX_JOBS
        if overflow > 0:
            # ``_JOBS`` is insertion-ordered by creation, so iterating gives
            # terminal records oldest-first.
            finished = [
                key
                for key, job in _JOBS.items()
                if key != job_id and job["status"] in TERMINAL_JOB_STATUSES
            ]
            for key in finished[:overflow]:
                _JOBS.pop(key, None)
        return dict(record)


def get_evaluation_job(job_id: str) -> dict[str, Any] | None:
    """Return a shallow copy of the job record, if present.

    Polling also drives the deadline sweep so a stuck job is surfaced as
    ``error`` to the waiting client rather than spinning until it is reaped
    by an unrelated request.
    """

    reap_stale_jobs()
    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        return dict(record) if record is not None else None


def update_evaluation_job(job_id: str, **fields: Any) -> None:
    """Update job fields under the registry lock."""

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is not None:
            record.update(fields)


def finalize_evaluation_job(job_id: str, **fields: Any) -> bool:
    """Move a job to a terminal state unless the reaper already got there.

    Returns whether the write happened.  A worker whose evaluation overran
    the deadline must not resurrect a job the reaper already failed, so a
    terminal record is left untouched.
    """

    with _JOB_LOCK:
        record = _JOBS.get(job_id)
        if record is None or record["status"] in TERMINAL_JOB_STATUSES:
            return False
        record.update(fields)
        return True


def reap_stale_jobs(now: datetime | None = None) -> list[str]:
    """Fail non-terminal jobs that have overrun :data:`JOB_TIMEOUT_SECONDS`.

    Invoked opportunistically from the submit and poll paths (no background
    thread), so a hung or wedged job eventually reaches a terminal state:
    waiting clients stop spinning and the registry slot becomes eligible for
    capacity eviction again.  Running jobs are timed against ``started_at``,
    still-queued jobs against ``created_at``.  A worker thread that is still
    executing cannot be killed, but its late result is discarded by
    :func:`finalize_evaluation_job`.
    """

    reference = now or datetime.now(UTC)
    reaped: list[str] = []
    with _JOB_LOCK:
        for key, record in _JOBS.items():
            if record["status"] in TERMINAL_JOB_STATUSES:
                continue
            stamp = record.get("started_at") or record.get("created_at")
            if not isinstance(stamp, str):
                continue
            try:
                started = datetime.fromisoformat(stamp)
            except ValueError:
                continue
            if (reference - started).total_seconds() > JOB_TIMEOUT_SECONDS:
                record["status"] = "error"
                record["error"] = f"evaluation timed out after {JOB_TIMEOUT_SECONDS}s"
                reaped.append(key)
    return reaped


def run_evaluation_job(
    *,
    job_id: str,
    source: Path,
    engine: str,
    provider: str | None,
    model: str | None,
    temperature: float,
    max_tokens: int | None,
    rubric_filename: str,
    rubric_summary: dict[str, Any] | None,
    format_summary: dict[str, Any] | None,
    preset: str,
) -> None:
    """Run the full evaluation pipeline on a worker thread.

    The uploaded source file is kept until the worker finishes so the
    blocking parse step owns its lifecycle; it is cleaned up afterwards.
    The runtime probe runs here rather than on the event loop, so a local
    server that is slow to answer never delays the HTTP response.
    """

    update_evaluation_job(
        job_id,
        status="running",
        started_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    try:
        model_config = build_model_config(
            engine=engine,
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_config, runtime_warning = resolve_runtime(model_config)
        if bool(runtime_warning):
            logger.warning("evaluation job %s: %s", job_id, runtime_warning)
        result = evaluate_document(
            source,
            model_config=model_config,
            rubric_filename=rubric_filename,
            rubric=rubric_summary,
            format_requirements=format_summary,
        )
    except Exception as exc:  # noqa: BLE001 - record any failure on the job
        finalize_evaluation_job(job_id, status="error", error=str(exc))
        return
    finally:
        cleanup_upload_file(source)

    payload = result.to_dict()
    if rubric_summary is not None:
        payload["metadata"]["rubric"] = rubric_summary
    if format_summary is not None:
        payload["metadata"]["format_requirements"] = format_summary
    payload["metadata"]["preset"] = preset
    if finalize_evaluation_job(job_id, status="done", result=payload):
        append_history(result)


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
        model_config = build_model_config(
            engine=request.engine,
            provider=request.provider,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        model_config, runtime_warning = resolve_runtime(model_config)
        if bool(runtime_warning):
            logger.warning("evaluate request: %s", runtime_warning)
        result = evaluate_document(
            source,
            model_config=model_config,
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
        logger.exception("evaluate request failed")
        raise HTTPException(
            status_code=500, detail="evaluation failed; see server logs"
        ) from exc
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
        logger.exception("structure request failed")
        raise HTTPException(
            status_code=500, detail="structuring failed; see server logs"
        ) from exc
    return ApiResponse(ok=True, mode="structure", data=document.to_dict())


@app.post("/evaluate/upload", response_model=ApiResponse)
async def evaluate_upload(
    file: UploadFile | None = EVALUATE_UPLOAD_FILE_DEFAULT,
    preset: str = EVALUATE_UPLOAD_PRESET_DEFAULT,
    engine: str = EVALUATE_UPLOAD_ENGINE_DEFAULT,
    provider: str | None = EVALUATE_UPLOAD_PROVIDER_DEFAULT,
    model: str | None = EVALUATE_UPLOAD_MODEL_DEFAULT,
    temperature: float = EVALUATE_UPLOAD_TEMPERATURE_DEFAULT,
    max_tokens: int | None = EVALUATE_UPLOAD_MAX_TOKENS_DEFAULT,
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
    if job is None:
        # resolve_thesis_source already persisted the upload, so this
        # rejection path owns deleting it: a refused burst must not leak files.
        cleanup_upload_file(source)
        raise HTTPException(
            status_code=429, detail="too many evaluations in progress; retry shortly"
        )
    _EVALUATE_EXECUTOR.submit(
        run_evaluation_job,
        job_id=job["job_id"],
        source=source,
        engine=engine,
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

    payload = load_json_resource(filename)
    items = normalize_rubric_payload(payload)
    format_filename = FORMAT_RUBRIC_BY_SCORE_RUBRIC.get(filename)
    if format_filename is not None and not any(
        item.get("criterion") == FORMAT_RUBRIC_NAME for item in items
    ):
        format_spec = normalize_format_spec_payload(load_json_resource(format_filename))
        rules = extract_format_rules(format_spec)
        items = [
            *items,
            {
                "criterion": FORMAT_RUBRIC_NAME,
                "key": FORMAT_RUBRIC_KEY,
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

    if filename is None or filename == "":
        return None
    payload = load_json_resource(filename)
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

    await run_in_threadpool(UPLOAD_DIR.mkdir, parents=True, exist_ok=True)
    temp_path = UPLOAD_DIR / f".upload_{slot}_{uuid.uuid4().hex}{suffix}.tmp"
    stored_path = UPLOAD_DIR / f"{slot}_{uuid.uuid4().hex}{suffix}"
    written = 0
    try:
        output = await run_in_threadpool(lambda: temp_path.open("wb"))
        try:
            while chunk := await file.read(UPLOAD_READ_CHUNK_BYTES):
                next_written = written + len(chunk)
                if next_written > MAX_EVALUATE_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413, detail=build_upload_too_large_message()
                    )
                written = next_written
                await run_in_threadpool(output.write, chunk)
        finally:
            await run_in_threadpool(output.close)
        if written == 0:
            msg = f"{slot.replace('_', ' ')} file is empty"
            raise ValueError(msg)

        await run_in_threadpool(temp_path.replace, stored_path)
    finally:
        await run_in_threadpool(temp_path.unlink, missing_ok=True)

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


def run_api() -> None:
    """Run the FastAPI app with uvicorn."""

    uvicorn.run("thesisev.api:app", host="127.0.0.1", port=8000, reload=False)


def append_history(result: EvaluationResult) -> None:
    """Append a compact evaluation summary to local history.

    Within a single process, the read/insert/write runs under
    ``_HISTORY_LOCK`` and the file is swapped atomically via ``Path.replace``,
    so concurrent evaluations cannot tear or corrupt the file.  The lock is
    process-local: on a multi-worker / multi-process deployment two appends
    can still interleave and lose an update (last-writer-wins), and the swap
    does not ``fsync``, so an OS crash right after the write may drop the
    newest entry.  History durability therefore assumes the same single-process
    posture as the job queue above.
    """

    with _HISTORY_LOCK:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        items = read_history_unlocked()
        items.insert(0, build_history_entry(result))
        HISTORY_TMP_PATH.write_text(
            json.dumps(
                items[:MAX_HISTORY_ITEMS],
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        HISTORY_TMP_PATH.replace(HISTORY_PATH)


def read_history() -> list[dict[str, Any]]:
    """Read stored history entries from disk."""

    with _HISTORY_LOCK:
        return read_history_unlocked()


def coerce_history_entries(raw: Any) -> list[dict[str, Any]]:
    """Keep only well-formed dict entries from a decoded history payload.

    A non-list payload yields an empty list; list entries that are not dicts
    are dropped. This is the shape guard for the persisted-history boundary.
    """

    if not isinstance(raw, list):
        return []
    return [entry for entry in raw if isinstance(entry, dict)]


def read_history_unlocked() -> list[dict[str, Any]]:
    """Read and normalize history entries without acquiring the history lock.

    ``history.json`` is an external persistence boundary: it may be hand-edited,
    left truncated by an interrupted process, or written by an older schema.
    A malformed file must degrade to an empty/partial history rather than crash
    the ``/history`` endpoint or break :func:`append_history`'s read-modify-write
    (which does ``items.insert(0, ...)`` and therefore needs a real list).
    """

    if not HISTORY_PATH.exists():
        return []
    # A genuine OSError (permissions, transient lock) is *not* swallowed: if it
    # were, append_history would overwrite the file and drop recoverable data.
    text = HISTORY_PATH.read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except JSONDecodeError as exc:
        logger.warning(
            "history file is not valid JSON (%s); treating history as empty", exc
        )
        return []
    items = coerce_history_entries(raw)
    if not isinstance(raw, list):
        logger.warning("history file is not a JSON list; treating history as empty")
    elif len(items) != len(raw):
        logger.warning("dropped %d malformed history entries", len(raw) - len(items))
    return items


#: Metadata keys that must never reach disk.  ``api_key`` was persisted for
#: sixteen reviews before the credential handling was corrected, so the write
#: path filters defensively rather than trusting every producer of ``model``
#: metadata to keep omitting it.
FORBIDDEN_METADATA_KEYS = frozenset({"api_key"})


def build_history_entry(result: EvaluationResult) -> dict[str, Any]:
    """Build a compact serialized history entry."""

    model_metadata = {
        key: value
        for key, value in (result.metadata.get("model") or {}).items()
        if key not in FORBIDDEN_METADATA_KEYS
    }
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
        "model": model_metadata,
    }
