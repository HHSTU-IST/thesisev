"""Analysis helpers for statistics and issue detection."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from thesisev.models import (
    Issue,
    Section,
    Statistic,
    TechnologyStackItem,
    ThesisDocument,
)
from thesisev.resources import load_json_resource

TECH_KEYWORDS = load_json_resource("tech_keywords.json")
STOPWORDS = set(load_json_resource("stopwords.json"))
COLLOQUIAL_PATTERNS = load_json_resource("colloquial_patterns.json")


def build_statistics(document: ThesisDocument) -> list[Statistic]:
    """Build human-readable statistics for the thesis."""

    annotate_section_statistics(document)
    statistics = [
        Statistic(label="总字数", value=str(document.total_word_count)),
        Statistic(label="章节数", value=str(len(document.sections))),
        Statistic(label="段落数", value=str(len(document.paragraphs))),
        Statistic(label="句子数", value=str(len(document.sentences))),
    ]
    for section in document.root_sections:
        statistics.append(
            Statistic(
                label=f"章节占比 - {section.title}",
                value=(
                    f"{section.ratio * 100:.1f}% "
                    f"({section.subtree_word_count}/{document.total_word_count})"
                ),
            )
        )
        statistics.extend(build_child_statistics(section))
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


def extract_technology_details(document: ThesisDocument) -> list[TechnologyStackItem]:
    """Extract categorized technology names with keyword matching."""

    found: list[TechnologyStackItem] = []
    for entry in TECH_KEYWORDS:
        matched_terms = [
            alias
            for alias in entry["aliases"]
            if contains_term(document.cleaned_text, alias)
        ]
        if not matched_terms:
            continue
        found.append(
            TechnologyStackItem(
                name=entry["name"],
                category=entry["category"],
                matched_terms=sorted(set(matched_terms), key=str.lower),
            )
        )
    return sorted(found, key=lambda item: (item.category, item.name.lower()))


def extract_technology_stack(document: ThesisDocument) -> list[str]:
    """Extract technology names with keyword matching."""

    return [item.name for item in extract_technology_details(document)]


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


def annotate_section_statistics(document: ThesisDocument) -> None:
    """Populate ratio statistics on the document's section tree."""

    total_words = max(document.total_word_count, 1)
    for section in document.root_sections:
        compute_section_subtree(section)
    for section in document.root_sections:
        section.parent_ratio = 1.0
        apply_section_ratios(
            section, total_words=total_words, parent_total=section.subtree_word_count
        )


def compute_section_subtree(section: Section) -> int:
    """Compute the subtree word count for a section."""

    child_total = sum(compute_section_subtree(child) for child in section.children)
    section.subtree_word_count = section.word_count + child_total
    return section.subtree_word_count


def apply_section_ratios(section: Section, total_words: int, parent_total: int) -> None:
    """Apply document-level and parent-level ratios to a section tree."""

    section.ratio = round(section.subtree_word_count / max(total_words, 1), 4)
    if section.children:
        for child in section.children:
            child.parent_ratio = round(
                child.subtree_word_count / max(section.subtree_word_count, 1), 4
            )
            apply_section_ratios(
                child, total_words=total_words, parent_total=section.subtree_word_count
            )


def build_child_statistics(section: Section) -> list[Statistic]:
    """Build nested statistics for child sections within a parent section."""

    statistics: list[Statistic] = []
    for child in section.children:
        statistics.append(
            Statistic(
                label=f"章内占比 - {section.title} / {child.title}",
                value=(
                    f"{child.parent_ratio * 100:.1f}% "
                    f"({child.subtree_word_count}/{section.subtree_word_count})"
                ),
            )
        )
        statistics.extend(build_child_statistics(child))
    return statistics


def group_technology_stack(
    technology_details: list[TechnologyStackItem],
) -> dict[str, list[str]]:
    """Group extracted technologies by category."""

    grouped: dict[str, list[str]] = defaultdict(list)
    for item in technology_details:
        grouped[item.category].append(item.name)
    return dict(sorted(grouped.items()))


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


def contains_term(text: str, term: str) -> bool:
    """Check whether a term appears in the text with simple boundary handling."""

    if re.search(r"[A-Za-z0-9]", term):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE
        )
        return pattern.search(text) is not None
    return term in text
