# Thesisev

轻量论文评审工具。

## 设计理念

- 格式检测与格式评价由本地程序完成
- 评分标准由大模型打分
- 内容评价由大模型生成
- 评分标准与格式要求均来自程序内置 `json` 文件

## 主要功能

- 论文结构解析：支持上传 `md` 或 `docx` 论文，解析标题、章节、段落与句子结构。
- 本地格式检测与评分：自动统计章节占比、主题相关度、关键词和技术栈，并由本地程序识别格式问题；格式评价仅做本地复核，不依赖上传规则文件。
- 大模型内容评价：基于 LangChain 接入多种大模型，默认使用 `deepseek/deepseek-chat` 生成论文内容评价，LLM 负责内容评价与评分。
- 预设规则读取：评分标准与格式要求均从程序内置 `json` 文件读取，UI 提供评分预设下拉菜单，选择后自动加载对应的评分标准与格式要求。
- 历史记录：自动保存最近评审记录；上传文件仅用于本次评审。
- 多入口使用：同时提供 CLI、FastAPI API 和内置 Web UI，便于命令行调用、接口集成和页面操作。

## 快速开始

安装依赖：

```bash
uv sync
```

命令行评审：

```bash
uv run thesisev examples/sample_thesis.md
```

启动 Web UI：

```bash
./scripts/start_api.sh
open http://127.0.0.1:8000
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

评分预设由程序内置，当前 UI 可选择的预设例如 `thesis_tech` 和 `report_iot`。对应的评分标准与格式要求会自动加载，无需上传。`report_iot` 会同时读取 `score_report_iot.json` 和 `score_report_iot_f.json`。

内置格式规则采用结构化 `json`，每条规则在 `check` 中声明 `type`、`expected` 和必要参数，便于本地解析与人工复核。

## 评分逻辑

- 格式检测与格式评价由本地程序完成，包括问题清单、格式要求读取和规则化评分明细。
- 内容评价由 LLM 生成，当前实现位于 `thesisev/commentary.py`；未配置 API Key 时使用本地内容评价模板回退。
- 六项评分标准由 LLM 生成，当前实现位于 `thesisev/scoring.py` 的 `calculate_score_report()`；未配置 API Key 时回退到本地规则。
- 返回结果中：
  - `score` 表示最终分数
  - `metadata.score_detail` 表示 6 项规则化评分明细
  - `metadata.score_source` 为 `llm` 或 `local`
  - `metadata.comment_source` 为 `llm` 或 `fallback`
  - `metadata.evaluation_roles` 表示格式检测、格式评价和内容评价分别由谁完成

## 评价标准

### 毕业设计

- 理工科：`config/score_thesis_tech.json`

### 调研报告

- 物联网：
  - 内容：`config/score_report_iot.json`
  - 格式：`config/score_report_iot_f.json`

## 毕业设计评分实现方案

- `analyzers.py` 继续负责提取结构、关键词、问题、主题相关度等信号
- `scoring.py` 负责把这些信号整理给 LLM 生成 6 项标准分，失败时回退到本地规则。

输出结构建议：

```json
{
  "score": 74,
  "raw_score": 48.0,
  "raw_total": 60.0,
  "score_source": "llm",
  "criteria": [
    {
      "key": "topic_workload",
      "name": "选题及工作量",
      "score": 15.5,
      "max_score": 20,
      "evidence": ["章节数 5", "篇幅 8200", "主题相关占比 72.4%"],
      "deductions": ["章节分布略不均衡"],
      "suggestions": ["补充需求分析或实验验证内容"]
    }
  ]
}
```

6 项标准的规则化实现：

- [x] `score_topic_workload(document, topic_analysis, technology_details)`：根据内容规模、章节完整度、主题聚焦度和技术覆盖给分。
- [x] `score_research_argument(document, keywords)`：检测参考文献、引用数量、文献与主题关键词的相关性，以及“因此/表明/综上”等论证表达。
- [x] `score_translation(document)`：检测是否同时存在中英文摘要，判断英文摘要长度、句子完整性和英文摘要篇幅。
- [x] `score_experiment_analysis(document, technology_details)`：判断是否包含方案、数据处理、分析论证、可行性或效益分析等要素。
- [x] `score_writing_quality(document, issues, format_requirements)`：基于问题数量和严重程度扣分，并读取内置格式要求作为复核依据。
- [x] `score_innovation(document, technology_details)`：检测“创新/改进/优化/提出/应用价值”等表述，并结合结论章节和技术组合给启发式评分。

总分计算建议：

```python
raw_score = sum(item.score for item in criteria)
raw_total = sum(item.max_score for item in criteria)  # 当前为 60
score = round(raw_score / raw_total * 100)
```

## 输出示例

CLI 文本输出示例：

```text
Title: 基于 LangChain 的论文评价助手设计与实现
Type: md
Score: 74

Statistics:
- 篇幅: 213
- 章节数: 3

Content Evaluation:
论文围绕 LangChain、论文评价助手设计等内容展开，章节安排较为均衡，
主题关联内容占比基本合理，技术方案已有一定体现。
```

API JSON 输出示例：

```json
{
  "ok": true,
  "mode": "evaluate_upload",
  "data": {
    "score": 74,
    "comment": "论文围绕 LangChain、论文评价助手设计等内容展开……",
    "statistics": [
      {"label": "篇幅", "value": "213"},
      {"label": "章节数", "value": "3"}
    ],
    "metadata": {
      "score_source": "llm",
      "score_detail": {
        "raw_score": 48.0,
        "raw_total": 60.0,
        "rubric_source": "score_thesis_tech.json",
        "score_source": "llm"
      },
      "comment_source": "llm",
      "evaluation_roles": {
        "format_detection": "local_program",
        "format_evaluation": "local_program",
        "content_evaluation": "llm"
      },
      "model": {
        "provider": "deepseek",
        "model": "deepseek-chat"
      },
      "rubric": {
        "total_score": 40.0
      },
      "format_requirements": {
        "item_count": 3
      }
    }
  }
}
```

## 接口

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/history
```

## 说明

- 默认模型：`deepseek/deepseek-chat`
- 未配置 API Key 时，内容评价自动回退到本地模板
- 格式检测与格式评价由本地程序完成，LLM 只负责内容评价
- 返回结果中可通过 `metadata.score_source`、`metadata.comment_source` 和 `metadata.evaluation_roles` 判断职责来源
- 最近评审会写入本地 `data/history.json`
- 现在只支持上传 `md` 或 `docx` 文件评审
- 每次通过 Web UI 或 `/evaluate/upload` 评审都必须显式上传论文文件；系统不会复用、展示或缓存全局最近上传内容

## 目录说明

- `config/`：静态配置文件，包括规则库、关键词库和 `provider_env.toml`
- `data/`：运行时数据，包括历史记录和上传过程中的临时文件
- `static/`：前端静态资源，包括样式和交互脚本
- `templates/`：FastAPI 内置 UI 的 HTML 模板
- `thesisev/`：核心 Python 包，包括解析、分析、本地评分、内容评价生成、CLI 和 API
- `scripts/`：项目启动脚本

## 架构图

### 工程结构

```mermaid
flowchart LR
    U["用户层<br/>浏览器 UI / CLI"]
    A["应用层<br/>FastAPI 路由与任务编排"]
    C["核心能力层<br/>论文解析 / 本地格式检测 / 本地评分 / 内容评价"]
    M["模型层<br/>LangChain 多模型接入<br/>默认 DeepSeek"]
    CFG["配置层<br/>规则库 / 模型配置"]
    D["数据层<br/>历史记录 / 临时上传文件"]

    U --> A
    A --> C
    C --> M
    C --> CFG
    C --> D
    A --> D
```

### 运行流程

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as 前端界面
    participant API as FastAPI 服务
    participant Core as 评审引擎
    participant LLM as 大模型
    participant Config as 本地配置
    participant Data as 本地数据

    User->>UI: 上传论文并选择内置评分预设
    UI->>API: 发起评审请求
    API->>Data: 保存当前上传文件与历史状态
    API->>Core: 执行解析、统计、本地格式检测与评分
    Core->>LLM: 生成论文内容评价
    Core->>Config: 读取规则与模型配置
    API->>Data: 保存历史记录
    API-->>UI: 返回结构化评审结果
    UI-->>User: 展示内容评价、格式检测、评分标准、格式要求
```
