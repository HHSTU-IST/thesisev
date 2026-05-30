# Thesisev

轻量论文评审工具。

## 安装

```bash
uv sync
```

## CLI

只接受 `md` 或 `docx` 文件输入：

```bash
uv run thesisev examples/sample_thesis.md
uv run thesisev examples/sample_thesis.md --output structure
```

## API / UI

```bash
./scripts/start_api.sh
open http://127.0.0.1:8000
```

UI 可选上传评分标准 `json` 文件，支持这两种格式：

```json
{"摘要": 10, "结构": 20, "结论": 10}
```

```json
[{"摘要": 10}, {"结构": 20}, {"结论": 10}]
```

UI 也可选上传格式要求 `json` 文件，支持对象或数组，例如：

```json
{"页边距": "上2.5cm，下2.5cm", "标题层级": "最多三级", "参考文献": "GB/T 7714"}
```

```json
["正文使用小四宋体", "摘要不超过 300 字", {"行距": "1.5 倍"}]
```

## 接口

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/history
curl http://127.0.0.1:8000/last-upload
```

## 说明

- 默认模型：`deepseek/deepseek-chat`
- 未配置 API Key 时自动回退到规则评语
- 最近评审会写入本地 `data/history.json`
- 现在只支持上传 `md` 或 `docx` 文件评审
- 上传过一次后，页面刷新或再次评审时会自动复用上次上传的论文、评分标准、格式要求
