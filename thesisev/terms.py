"""Analyzer vocabularies inlined as code.

These catalogs are analysis heuristics rather than scoring rules, so they live
next to the algorithms that consume them instead of in ``config/``. ``config/``
stays reserved for the rubric presets, their paired format specifications and
the runtime dictionaries (colloquial words, stopwords, punctuation).

Every catalog keeps the container type its consumer needs: ``frozenset`` when
the value is only tested for membership, ``tuple`` when iteration order is
significant (prefix/suffix stripping).
"""

from __future__ import annotations

from typing import TypedDict


class TechnologyKeyword(TypedDict):
    """One technology-stack entry with its alias spellings."""

    name: str
    category: str
    aliases: list[str]


TECHNOLOGY_KEYWORDS: tuple[TechnologyKeyword, ...] = (
    {"name": "Python", "category": "编程语言", "aliases": ["Python", "python"]},
    {"name": "Java", "category": "编程语言", "aliases": ["Java", "java"]},
    {"name": "C++", "category": "编程语言", "aliases": ["C++", "c++"]},
    {
        "name": "LangChain",
        "category": "模型框架",
        "aliases": ["LangChain", "langchain"],
    },
    {"name": "FAISS", "category": "向量检索", "aliases": ["FAISS", "faiss"]},
    {"name": "NumPy", "category": "数据处理", "aliases": ["NumPy", "numpy"]},
    {"name": "Pandas", "category": "数据处理", "aliases": ["Pandas", "pandas"]},
    {"name": "PyTorch", "category": "机器学习框架", "aliases": ["PyTorch", "pytorch"]},
    {
        "name": "TensorFlow",
        "category": "机器学习框架",
        "aliases": ["TensorFlow", "tensorflow"],
    },
    {
        "name": "Scikit-learn",
        "category": "机器学习框架",
        "aliases": ["Scikit-learn", "scikit-learn", "sklearn"],
    },
    {"name": "OpenAI", "category": "模型平台", "aliases": ["OpenAI", "openai"]},
    {"name": "MySQL", "category": "数据库", "aliases": ["MySQL", "mysql"]},
    {
        "name": "PostgreSQL",
        "category": "数据库",
        "aliases": ["PostgreSQL", "postgresql", "postgres"],
    },
    {"name": "Redis", "category": "数据库", "aliases": ["Redis", "redis"]},
    {"name": "Docker", "category": "工程基础设施", "aliases": ["Docker", "docker"]},
    {"name": "FastAPI", "category": "Web 框架", "aliases": ["FastAPI", "fastapi"]},
    {"name": "Flask", "category": "Web 框架", "aliases": ["Flask", "flask"]},
    {"name": "OpenCV", "category": "图像处理", "aliases": ["OpenCV", "opencv"]},
    {"name": "TensorRT", "category": "推理框架", "aliases": ["TensorRT", "tensorrt"]},
    {"name": "MQTT", "category": "通信协议", "aliases": ["MQTT", "mqtt"]},
    {
        "name": "Jetson Nano",
        "category": "hardware",
        "aliases": ["Jetson Nano", "jetson nano"],
    },
    {
        "name": "树莓派",
        "category": "hardware",
        "aliases": ["树莓派", "Raspberry Pi", "raspberry pi"],
    },
    {"name": "RK3588", "category": "hardware", "aliases": ["RK3588", "rk3588"]},
    {"name": "V853", "category": "hardware", "aliases": ["V853", "v853"]},
    {
        "name": "摄像头",
        "category": "device",
        "aliases": ["摄像头", "Camera Module", "camera module"],
    },
)

TOPIC_NOISE_GENERIC_TERMS: frozenset[str] = frozenset(
    {
        "研究背景",
        "结构安排",
        "内容分布",
        "模块划分",
        "快速了解",
        "系统需要",
        "本章介绍",
        "本章说明",
        "往往需要",
        "介绍研究背景",
        "说明解析流程",
        "docx",
        "draw",
        "draw.io",
        "io",
        "mermaid",
    }
)

TOPIC_NOISE_GENERIC_FRAGMENTS: tuple[str, ...] = (
    "研究背景",
    "结构安排",
    "内容分布",
    "模块划分",
    "快速了解",
    "系统需要",
    "本章介绍",
    "本章说明",
    "往往需要",
)

GENERIC_ANALYSIS_TERMS: frozenset[str] = frozenset(
    {
        "内容",
        "模块",
        "结构",
        "设计",
        "实现",
        "系统",
        "论文",
        "评价",
        "助手",
        "分析",
        "研究",
    }
)

GENERIC_TOPIC_TERMS: frozenset[str] = frozenset(
    {
        "章节",
        "评语",
        "后续",
        "基础",
        "结果",
        "报告",
        "问题",
        "任务",
        "目标",
        "整体",
        "质量",
        "老师",
        "场景",
        "初步",
        "本章",
    }
)

GENERIC_PHRASE_TERMS: frozenset[str] = frozenset(
    {
        "第一",
        "第二",
        "第三",
        "基于",
        "提出",
        "形成",
        "快速",
        "需要",
        "介绍",
        "说明",
        "了解",
        "划分",
        "句子",
        "段落",
    }
)

PHRASE_PREFIXES: tuple[str, ...] = (
    "本文提出",
    "本章介绍",
    "本章说明",
    "往往需要",
    "系统需要",
    "需要快速",
    "需要识别",
    "用于支持",
    "用于",
)

PHRASE_NOISE_FRAGMENTS: tuple[str, ...] = (
    "往往需要",
    "快速了解",
    "本章介绍",
    "本章说明",
    "系统需要",
)

DOMAIN_KEY_PHRASES: tuple[str, ...] = (
    "论文评价助手",
    "结构化分析",
    "数据结构",
    "解析流程",
    "模块划分",
    "研究背景",
    "问题定义",
    "内容分布",
    "结构安排",
    "章节识别",
    "段落识别",
    "句子识别",
)

ACTION_OBJECT_TERMS: frozenset[str] = frozenset(
    {"章节", "段落", "句子", "结构", "内容", "数据"}
)

ACTION_VERBS: tuple[str, ...] = (
    "识别",
    "检查",
    "生成",
    "分析",
    "处理",
    "统计",
    "拆分",
    "计算",
    "检索",
    "评价",
)

ACTION_PREFIXES: tuple[str, ...] = (
    "这个系统使用",
    "系统需要完成",
    "系统需要",
    "系统使用",
    "一个用于",
    "用于",
    "需要完成",
    "需要",
    "完成",
    "支持",
    "根据",
    "针对",
    "进行",
    "实现",
)

ACTION_NOISE_FRAGMENTS: tuple[str, ...] = (
    "辅助系统",
    "整体质量",
    "课程论文",
    "老师需要",
    "很多场景",
    "结果不仅",
    "模块负责",
)

TITLE_SUFFIXES: tuple[str, ...] = (
    "设计与实现",
    "设计实现",
    "设计",
    "实现",
    "研究",
    "方法",
    "方案",
)

SECTION_HEADING_TERMS: frozenset[str] = frozenset(
    {
        "摘要",
        "绪论",
        "第一章",
        "第二章",
        "第三章",
        "第四章",
        "第五章",
        "第六章",
        "第七章",
        "第八章",
    }
)

GENERIC_TITLE_TERMS: frozenset[str] = frozenset(
    {
        "论文",
        "设计",
        "实现",
        "系统",
        "研究",
        "分析",
        "评价",
        "助手",
        "基于",
        "面向",
        "方法",
    }
)

COMMENT_KEYWORD_NOISE_TERMS: frozenset[str] = frozenset(
    {"docx", "draw", "draw.io", "io", "mermaid"}
)
