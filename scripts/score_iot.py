from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

parent_dir = Path.home() / "Documents/HHSTU-L-CV"

for dir in Path.iterdir(parent_dir):
    if dir.is_dir() and dir.name.endswith("结课"):
        base_dir = parent_dir / dir

output_path = base_dir / "report_iot_scores.csv"
files = sorted(
    path
    for path in base_dir.iterdir()
    if path.is_file() and path.suffix.lower() in {".docx", ".md"}
)

rows: list[dict[str, str]] = []
criterion_names: list[str] = []
seen_criteria: set[str] = set()

for index, path in enumerate(files, start=1):
    cmd = ["uv", "run", "thesisev", str(path), "--preset", "report_iot", "--json"]
    completed = subprocess.run(
        cmd,
        cwd=str(Path.home() / "Documents/GitHub/thesisev"),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    row: dict[str, str] = {
        "index": str(index),
        "file_name": path.name,
        "title": "",
        "score": "",
        "topic_relevance_ratio": "",
        "issue_count": "",
        "software_technology_stack": "",
        "hardware_technology_stack": "",
        "comment": "",
        "error": "",
    }
    if completed.returncode != 0:
        row["error"] = (
            (completed.stderr or completed.stdout).strip().replace("\n", " | ")
        )
        rows.append(row)
        continue

    payload = json.loads(completed.stdout)
    criteria = payload.get("metadata", {}).get("score_detail", {}).get("criteria", [])
    for criterion in criteria:
        name = str(criterion.get("name", "")).strip()
        if name and name not in seen_criteria:
            seen_criteria.add(name)
            criterion_names.append(name)
    row.update(
        {
            "title": str(payload.get("document", {}).get("title", "")),
            "score": str(payload.get("score", "")),
            "topic_relevance_ratio": str(payload.get("topic_relevance_ratio", "")),
            "issue_count": str(len(payload.get("issues", []))),
            "software_technology_stack": "、".join(
                payload.get("software_technology_stack", [])
            ),
            "hardware_technology_stack": "、".join(
                payload.get("hardware_technology_stack", [])
            ),
            "comment": str(payload.get("comment", "")).replace("\n", " "),
        }
    )
    for criterion in criteria:
        name = str(criterion.get("name", "")).strip()
        if name:
            row[f"criterion_{name}"] = str(criterion.get("score", ""))
    rows.append(row)

fieldnames = [
    "index",
    "file_name",
    "title",
    "score",
    "topic_relevance_ratio",
    "issue_count",
    "software_technology_stack",
    "hardware_technology_stack",
]
fieldnames.extend(f"criterion_{name}" for name in criterion_names)
fieldnames.extend(["comment", "error"])

with output_path.open("w", encoding="utf-8-sig", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(output_path)
