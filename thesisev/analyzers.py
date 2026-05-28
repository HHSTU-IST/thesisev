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
CHINESE_CONTEXT_PUNCTUATION = load_json_resource("chinese_context_punctuation.json")
ENGLISH_CONTEXT_PUNCTUATION = load_json_resource("english_context_punctuation.json")
REPEATED_PUNCTUATION_RULE = load_json_resource("repeated_punctuation_rule.json")
REPEATED_PUNCTUATION_PATTERN = re.compile(REPEATED_PUNCTUATION_RULE["pattern"])


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
    """Detect punctuation issues with sentence-level locations."""

    issues: list[Issue] = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                sentence_text = sentence.text
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", sentence_text))
                has_latin = bool(re.search(r"[A-Za-z]", sentence_text))
                if has_chinese:
                    issues.extend(
                        detect_chinese_context_punctuation(
                            section=section,
                            paragraph_index=paragraph.index,
                            sentence_index=sentence.index,
                            sentence_text=sentence_text,
                        )
                    )
                    if sentence_text.endswith("."):
                        issues.append(
                            build_issue(
                                category="标点误用",
                                rule_id="cn_ascii_period",
                                severity="medium",
                                message="中文语境中的句末使用了英文句号，建议统一为中文句号。",
                                suggestion="建议将句末英文句号`.`替换为中文句号`。`。",
                                section=section,
                                paragraph_index=paragraph.index,
                                sentence_index=sentence.index,
                                matched_text=".",
                                excerpt=sentence_text,
                            )
                        )
                if not has_chinese and has_latin:
                    issues.extend(
                        detect_english_context_punctuation(
                            section=section,
                            paragraph_index=paragraph.index,
                            sentence_index=sentence.index,
                            sentence_text=sentence_text,
                        )
                    )
                repeated_match = REPEATED_PUNCTUATION_PATTERN.search(sentence_text)
                if repeated_match is not None:
                    issues.append(
                        build_issue(
                            category="标点误用",
                            rule_id=REPEATED_PUNCTUATION_RULE["rule_id"],
                            severity=REPEATED_PUNCTUATION_RULE["severity"],
                            message=REPEATED_PUNCTUATION_RULE["message"],
                            suggestion=REPEATED_PUNCTUATION_RULE["suggestion"],
                            section=section,
                            paragraph_index=paragraph.index,
                            sentence_index=sentence.index,
                            matched_text=repeated_match.group(0),
                            excerpt=sentence_text,
                        )
                    )
    return deduplicate_issues(issues)


def detect_colloquial_issues(document: ThesisDocument) -> list[Issue]:
    """Detect colloquial expressions in thesis sections."""

    issues: list[Issue] = []
    for section in document.sections:
        for paragraph in section.paragraphs:
            for sentence in paragraph.sentences:
                for phrase, rule in COLLOQUIAL_PATTERNS.items():
                    if phrase not in sentence.text:
                        continue
                    if not should_match_colloquial(phrase, sentence.text, rule):
                        continue
                    issues.append(
                        build_issue(
                            category="口语化表达",
                            rule_id=rule["rule_id"],
                            severity=rule["severity"],
                            message=build_colloquial_message(phrase, rule),
                            suggestion=build_colloquial_suggestion(rule),
                            section=section,
                            paragraph_index=paragraph.index,
                            sentence_index=sentence.index,
                            matched_text=phrase,
                            excerpt=sentence.text,
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

    unique: dict[tuple[str, str, str, int, int, str], Issue] = {}
    for issue in issues:
        key = (
            issue.category,
            issue.rule_id,
            issue.section_identifier,
            issue.paragraph_index,
            issue.sentence_index,
            issue.matched_text,
        )
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


def should_match_colloquial(
    phrase: str, sentence_text: str, rule: dict[str, object]
) -> bool:
    """Filter overly broad colloquial rules to reduce obvious false positives."""

    if len(phrase) == 1 and not bool(rule.get("allow_single_char", False)):
        return False
    return phrase in sentence_text


def build_colloquial_message(phrase: str, rule: dict[str, object]) -> str:
    """Build a colloquial issue message from rule config."""

    message = rule.get("message")
    if isinstance(message, str) and message:
        return message
    return f"检测到口语化表达“{phrase}”，可能不适合正式论文语境。"


def build_colloquial_suggestion(rule: dict[str, object]) -> str:
    """Build a colloquial issue suggestion from rule config."""

    suggestion = str(rule["suggestion"])
    replacements = rule.get("replacement_examples", [])
    if isinstance(replacements, list) and replacements:
        example_text = "、".join(str(item) for item in replacements)
        return f"{suggestion} 可参考：{example_text}。"
    return suggestion


def detect_chinese_context_punctuation(
    section: Section,
    paragraph_index: int,
    sentence_index: int,
    sentence_text: str,
) -> list[Issue]:
    """Detect ASCII punctuation used in Chinese sentences."""

    issues: list[Issue] = []
    for mark, rule in CHINESE_CONTEXT_PUNCTUATION.items():
        if mark not in sentence_text:
            continue
        issues.append(
            build_issue(
                category="标点误用",
                rule_id=rule["rule_id"],
                severity=rule["severity"],
                message=rule["message"],
                suggestion=rule["suggestion"],
                section=section,
                paragraph_index=paragraph_index,
                sentence_index=sentence_index,
                matched_text=mark,
                excerpt=sentence_text,
            )
        )
    return issues


def detect_english_context_punctuation(
    section: Section,
    paragraph_index: int,
    sentence_index: int,
    sentence_text: str,
) -> list[Issue]:
    """Detect Chinese punctuation used in English-like sentences."""

    issues: list[Issue] = []
    for mark, rule in ENGLISH_CONTEXT_PUNCTUATION.items():
        if mark not in sentence_text:
            continue
        issues.append(
            build_issue(
                category="标点误用",
                rule_id=rule["rule_id"],
                severity=rule["severity"],
                message=rule["message"],
                suggestion=rule["suggestion"],
                section=section,
                paragraph_index=paragraph_index,
                sentence_index=sentence_index,
                matched_text=mark,
                excerpt=sentence_text,
            )
        )
    return issues


def build_issue(
    *,
    category: str,
    rule_id: str,
    severity: str,
    message: str,
    suggestion: str,
    section: Section,
    paragraph_index: int,
    sentence_index: int,
    matched_text: str,
    excerpt: str,
) -> Issue:
    """Build an issue object with stable location metadata."""

    return Issue(
        category=category,
        rule_id=rule_id,
        severity=severity,
        message=message,
        suggestion=suggestion,
        section_identifier=section.identifier,
        section_title=section.title,
        paragraph_index=paragraph_index,
        sentence_index=sentence_index,
        matched_text=matched_text,
        excerpt=excerpt,
    )
