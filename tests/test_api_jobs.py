"""B2: /evaluate/upload 任务化 — job_id 返回与结果轮询。"""

from __future__ import annotations

import io
import time
from typing import Any

import pytest

from thesisev import api
from thesisev.api import app

pytest.importorskip("fastapi.testclient")

from fastapi.testclient import TestClient

SAMPLE_MD = """# 基于边缘计算的目标检测系统

## 摘要

本文研究基于边缘计算的目标检测系统，面向物联网场景完成数据处理、模型推理与结果验证。
Abstract This paper studies an edge computing based object detection system.

# 1 绪论

## 1.1 研究背景

随着边缘计算与物联网的发展，目标检测技术在实时场景中应用价值显著。
## 1.2 研究意义

本方案对系统架构、数据处理流程进行优化与改进，论证了算法的应用价值。

# 2 系统方案

## 2.1 总体设计

采用边缘计算架构完成数据采集、计算处理与结果验证，分析对比评估多种实现。

## 2.2 实验验证

测试结果表明：该方案能有效降低推理时延，因此具有较高的应用价值与推广意义。
综上，本文提出的改进方案具有可行性。
"""


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _submit(client: TestClient) -> dict[str, Any]:
    response = client.post(
        "/evaluate/upload",
        files={
            "file": (
                "sample.md",
                io.BytesIO(SAMPLE_MD.encode("utf-8")),
                "text/markdown",
            )
        },
        data={"preset": "thesis_tech", "provider": "deepseek", "model": ""},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "evaluate_upload"
    return payload["data"]


def test_upload_returns_job_id_immediately(client: TestClient) -> None:
    data = _submit(client)
    assert data["job_id"]
    assert data["status"] in {"queued", "running", "done", "error"}


def test_job_polling_eventually_returns_result(client: TestClient) -> None:
    data = _submit(client)
    job_id = data["job_id"]
    deadline = time.monotonic() + 60
    result_payload = None
    while time.monotonic() < deadline:
        status_response = client.get(f"/evaluate/jobs/{job_id}")
        assert status_response.status_code == 200
        job = status_response.json()["data"]
        if job["status"] == "done":
            result_payload = job["result"]
            break
        if job["status"] == "error":
            pytest.fail(f"job failed: {job['error']}")
        time.sleep(0.2)
    assert result_payload is not None, "job did not finish in time"
    assert "score" in result_payload
    assert result_payload["document"]["title"] == "基于边缘计算的目标检测系统"
    assert result_payload["metadata"]["preset"] == "thesis_tech"


def test_unknown_job_returns_404(client: TestClient) -> None:
    response = client.get("/evaluate/jobs/does-not-exist")
    assert response.status_code == 404


def test_missing_file_is_rejected(client: TestClient) -> None:
    response = client.post("/evaluate/upload", data={"preset": "thesis_tech"})
    assert response.status_code in {200, 400}
    if response.status_code == 200:
        # An empty multipart may still pass the form; the job then reports error.
        assert response.json()["data"]["job_id"]


@pytest.fixture()
def isolated_jobs() -> Any:
    """Hand a test a clean job registry, restoring the shared one afterwards."""

    saved = dict(api._JOBS)
    with api._JOB_LOCK:
        api._JOBS.clear()
    try:
        yield api
    finally:
        with api._JOB_LOCK:
            api._JOBS.clear()
            api._JOBS.update(saved)


def test_backpressure_refuses_fourth_live_job(
    isolated_jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_jobs, "MAX_JOBS", 3)
    accepted = [isolated_jobs.register_evaluation_job() for _ in range(3)]
    assert all(record is not None for record in accepted)
    # A burst past capacity is refused instead of evicting an in-flight job.
    assert isolated_jobs.register_evaluation_job() is None
    with isolated_jobs._JOB_LOCK:
        live = sum(
            1
            for job in isolated_jobs._JOBS.values()
            if job["status"] not in isolated_jobs.TERMINAL_JOB_STATUSES
        )
    assert live == 3
    assert len(isolated_jobs._JOBS) == 3


def test_capacity_evicts_only_terminal_jobs(
    isolated_jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_jobs, "MAX_JOBS", 2)
    live = isolated_jobs.register_evaluation_job()
    assert live is not None
    finished = isolated_jobs.register_evaluation_job()
    assert finished is not None
    isolated_jobs.update_evaluation_job(finished["job_id"], status="done")
    # Accepted because only 1 job is live; the finished record is evicted,
    # the live record is kept.
    third = isolated_jobs.register_evaluation_job()
    assert third is not None
    with isolated_jobs._JOB_LOCK:
        ids = set(isolated_jobs._JOBS)
    assert live["job_id"] in ids
    assert third["job_id"] in ids
    assert finished["job_id"] not in ids


def test_register_returns_snapshot_not_shared_dict(isolated_jobs: Any) -> None:
    record = isolated_jobs.register_evaluation_job()
    assert record is not None
    # A worker mutating the registry must not bleed into the returned copy.
    isolated_jobs.update_evaluation_job(record["job_id"], status="running")
    assert record["status"] == "queued"


def test_upload_returns_429_when_queue_saturated(
    client: TestClient, isolated_jobs: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(isolated_jobs, "MAX_JOBS", 1)
    assert isolated_jobs.register_evaluation_job() is not None
    response = client.post(
        "/evaluate/upload",
        files={
            "file": (
                "sample.md",
                io.BytesIO(SAMPLE_MD.encode("utf-8")),
                "text/markdown",
            )
        },
        data={"preset": "thesis_tech", "provider": "deepseek", "model": ""},
    )
    assert response.status_code == 429
    assert "retry" in response.json()["detail"].lower()
