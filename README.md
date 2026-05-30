# Thesisev

轻量论文评审工具。

## 安装

```bash
uv sync
```

## CLI

```bash
uv run thesisev examples/sample_thesis.md
uv run thesisev examples/sample_thesis.md --output structure
```

## API / UI

```bash
./scripts/start_api.sh
open http://127.0.0.1:8000
```

## 接口

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/history
```

## 说明

- 默认模型：`deepseek/deepseek-chat`
- 未配置 API Key 时自动回退到规则评语
- 最近评审会写入本地 `thesisev/data/history.json`
- 现在只支持上传文件评审
