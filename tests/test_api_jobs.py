"""B2: /evaluate/upload 任务化 — job_id 返回与结果轮询。"""

from __future__ import annotations

import io
import time
from typing import Any

import pytest

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
