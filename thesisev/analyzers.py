"""Analysis helpers for statistics and issue detection."""

from __future__ import annotations

import re
from collections import Counter

from thesisev.models import Issue, Statistic, ThesisDocument
from thesisev.resources import load_json_resource

TECH_KEYWORDS = set(load_json_resource("tech_keywords.json"))
STOPWORDS = set(load_json_resource("stopwords.json"))
COLLOQUIAL_PATTERNS = load_json_resource("colloquial_patterns.json")


def build_statistics(document: ThesisDocument) -> list[Statistic]:
    """Build human-readable statistics for the thesis."""

    statistics = [
        Statistic(label="总字数", value=str(document.total_word_count)),
        Statistic(label="章节数", value=str(len(document.sections))),
        Statistic(label="段落数", value=str(len(document.paragraphs))),
        Statistic(label="句子数", value=str(len(document.sentences))),
    ]
    total = max(sum(section.word_count for section in document.sections), 1)
    for section in document.sections:
        section.ratio = round(section.word_count / total, 4)
        statistics.append(
            Statistic(
                label=f"章节占比 - {section.title}", value=f"{section.ratio * 100:.1f}%"
            )
        )
    return statistics


def extract_keywords(document: ThesisDocument) -> list[str]:
    """Extract simple thesis keywords from the title and top frequent terms."""

    title_tokens = split_title_keywords(document.title)
    body_tokens = tokenize(document.cleaned_text)
    candidates = [
        token
        for token in title_tokens + body_tokens
        if token not in STOPWORDS and len(token) <= 12
    ]
    counts = Counter(candidates)
    ordered = [token for token, _ in counts.most_common(10)]
    return ordered[:5]


def extract_technology_stack(document: ThesisDocument) -> list[str]:
    """Extract technology names with keyword matching."""

    found = []
    lowered_text = document.cleaned_text.lower()
    for keyword in sorted(TECH_KEYWORDS):
        if keyword.lower() in lowered_text:
            found.append(keyword)
    return found


def detect_issues(document: ThesisDocument) -> list[Issue]:
    """Detect punctuation and colloquial writing issues."""

    issues = detect_punctuation_issues(document)
    issues.extend(detect_colloquial_issues(document))
    return issues


def detect_punctuation_issues(document: ThesisDocument) -> list[Issue]:
    """Detect mixed Chinese and English punctuation in the same sentence."""

    issues: list[Issue] = []
    for section in document.sections:
        for sentence in section.sentences:
            has_chinese = bool(re.search(r"[\u4e00-\u9fff]", sentence))
            english_punct = any(
                token in sentence for token in (",", ".", ";", ":", "(", ")", "[", "]")
            )
            chinese_punct = any(
                token in sentence
                for token in ("，", "。", "；", "：", "（", "）", "【", "】")
            )
            if has_chinese and english_punct:
                issues.append(
                    Issue(
                        category="标点误用",
                        severity="medium",
                        message="中文语境中出现英文标点，可能影响行文一致性。",
                        suggestion="建议统一替换为中文标点，并检查括号和分号等符号是否符合中文论文规范。",
                        section_title=section.title,
                        excerpt=sentence,
                    )
                )
                continue
            if not has_chinese and chinese_punct:
                issues.append(
                    Issue(
                        category="标点误用",
                        severity="low",
                        message="英文或代码语境中出现中文标点，建议统一。",
                        suggestion="建议根据上下文改为英文标点，保持术语、公式或代码片段的一致性。",
                        section_title=section.title,
                        excerpt=sentence,
                    )
                )
    return deduplicate_issues(issues)


def detect_colloquial_issues(document: ThesisDocument) -> list[Issue]:
    """Detect colloquial expressions in thesis sections."""

    issues: list[Issue] = []
    for section in document.sections:
        for sentence in section.sentences:
            for phrase, suggestion in COLLOQUIAL_PATTERNS.items():
                if phrase not in sentence:
                    continue
                issues.append(
                    Issue(
                        category="口语化表达",
                        severity="medium",
                        message=f"检测到口语化表达“{phrase}”，可能不适合正式论文语境。",
                        suggestion=suggestion,
                        section_title=section.title,
                        excerpt=sentence,
                    )
                )
    return deduplicate_issues(issues)


def calculate_score(issues: list[Issue], section_count: int) -> int:
    """Create a rough score from detected issues and document completeness."""

    score = 90
    for issue in issues:
        if issue.severity == "medium":
            score -= 4
        elif issue.severity == "low":
            score -= 2
        else:
            score -= 6
    if section_count <= 1:
        score -= 8
    return max(60, min(98, score))


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin text with light filtering."""

    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    latin_terms = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}", text)
    tokens = chinese_terms + latin_terms
    return [token for token in tokens if len(token.strip()) >= 2]


def split_title_keywords(title: str) -> list[str]:
    """Split the title into compact keyword-like units."""

    parts = re.split(r"[的与及和：:（）()\-\s]+", title)
    return [
        part for part in parts if part and part not in STOPWORDS and len(part) <= 10
    ]


def deduplicate_issues(issues: list[Issue]) -> list[Issue]:
    """Deduplicate issues by category and excerpt."""

    unique: dict[tuple[str, str], Issue] = {}
    for issue in issues:
        key = (issue.category, issue.excerpt)
        unique.setdefault(key, issue)
    return list(unique.values())
