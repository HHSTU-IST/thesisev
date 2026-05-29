# Thesisev

一个面向论文初审的轻量评审工具，支持：

- 解析 `md`、`txt`、`docx`
- 统计章节结构与主题相关度
- 检测部分标点和口语化问题
- 使用 LangChain 生成评语
- 提供 CLI、FastAPI 和内置 Web UI

## 安装

```bash
uv sync
```

## CLI

基础评估：

```bash
uv run thesisev examples/sample_thesis.md
```

输出结构化结果：

```bash
uv run thesisev examples/sample_thesis.md --output structure --json
```

使用 DeepSeek 生成评语：

```bash
export DEEPSEEK_API_KEY=your_api_key
uv run thesisev examples/sample_thesis.md --provider deepseek
```

## API 与 UI

启动服务：

```bash
./scripts/start_api.sh
```

打开 UI：

```bash
open http://127.0.0.1:8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

提交文本评估：

```bash
curl -X POST http://127.0.0.1:8000/evaluate/text \
  -F "filename=sample_thesis.md" \
  -F "text=基于 LangChain 的论文评价助手设计与实现\n第一章 绪论\n本文研究一个用于论文评价的辅助系统。" \
  -F "provider=deepseek"
```

上传文件评估：

```bash
curl -X POST http://127.0.0.1:8000/evaluate/upload \
  -F "file=@examples/sample_thesis.md" \
  -F "provider=deepseek"
```

最近历史：

```bash
curl http://127.0.0.1:8000/history
```

## 说明

- 默认模型为 `deepseek/deepseek-chat`
- 未配置对应 API Key 时，会回退到内置规则评语
- 最近评审会保存到本地 `thesisev/data/history.json`
