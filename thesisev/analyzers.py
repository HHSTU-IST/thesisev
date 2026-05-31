"""Analysis helpers for statistics and issue detection."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import tiktoken

from thesisev.models import (
    Issue,
    Paragraph,
    Section,
    Statistic,
    TechnologyStackItem,
    ThesisDocument,
)
from thesisev.resources import load_json_resource

KEYWORDS_TECH = load_json_resource("keywords_tech.json")
STOPWORDS = set(load_json_resource("stopwords.json"))
TOPIC_NOISE_TERMS = load_json_resource("topic_noise_terms.json")
COLLOQUIAL = load_json_resource("colloquial.json")
PUNCTUATION_CHINESE = load_json_resource("punctuation_chinese.json")
PUNCTUATION_ENGLISH = load_json_resource("punctuation_english.json")
PUNCTUATION_REPEATED = load_json_resource("punctuation_repeated.json")
REPEATED_PUNCTUATION_PATTERN = re.compile(PUNCTUATION_REPEATED["pattern"])
TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")
TOPIC_NOISE_GENERIC_TERMS = set(TOPIC_NOISE_TERMS["generic_terms"])
TOPIC_NOISE_GENERIC_FRAGMENTS = tuple(TOPIC_NOISE_TERMS["generic_fragments"])


@dataclass(slots=True)
class LocalIssueGroups:
    """Local issue groups kept separate from LLM content evidence."""

    format_issues: list[Issue]
    writing_issues: list[Issue]

    @property
    def all_issues(self) -> list[Issue]:
        """Return the combined list for compatibility with the existing UI."""

        return [*self.format_issues, *self.writing_issues]


def build_statistics(document: ThesisDocument) -> list[Statistic]:
    """Build human-readable statistics for the thesis."""

    annotate_section_statistics(document)
    topic_analysis = annotate_topic_relevance(document)
    statistics = [
        Statistic(label="篇幅", value=str(document.total_word_count)),
        Statistic(label="章节数", value=str(len(document.sections))),
        Statistic(label="段落数", value=str(len(document.paragraphs))),
        Statistic(label="句子数", value=str(len(document.sentences))),
        Statistic(
            label="主题相关内容占比",
            value=(
                f"{topic_analysis['document_ratio'] * 100:.1f}% "
                f"({topic_analysis['relevant_word_count']}/{document.total_word_count})"
            ),
        ),
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
        statistics.append(
            Statistic(
                label=f"主题相关度 - {section.title}",
                value=(
                    f"{section.topic_relevance_score * 100:.1f}% "
                    f"({section.topic_relevant_word_count}/{max(section.word_count, 1)})"
                ),
            )
        )
        statistics.extend(build_child_statistics(section))
    return statistics


def extract_keywords(document: ThesisDocument) -> list[str]:
    """Extract simple thesis keywords from the title and top frequent terms."""

    title_tokens = normalize_topic_terms(split_title_keywords(document.title))
    topic_tokens = extract_topic_keywords(document)
    latin_terms = normalize_topic_terms(extract_latin_terms(document.cleaned_text))
    candidates = deduplicate_preserving_order(title_tokens + topic_tokens + latin_terms)
    return candidates[:5]


def extract_technology_details(document: ThesisDocument) -> list[TechnologyStackItem]:
    """Extract categorized technology names with keyword matching."""

    found: list[TechnologyStackItem] = []
    for entry in KEYWORDS_TECH:
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


def extract_topic_keywords(document: ThesisDocument) -> list[str]:
    """Extract topic keywords from the title, abstract, and body."""

    title_tokens = normalize_topic_terms(split_title_keywords(document.title))
    abstract_tokens = normalize_topic_terms(
        extract_tiktoken_keywords(document.abstract, top_k=6)
    )
    action_tokens = normalize_topic_terms(
        extract_paragraph_action_keywords(document.paragraphs)
    )
    body_tokens = normalize_topic_terms(
        extract_tiktoken_keywords(document.cleaned_text, top_k=8)
    )
    latin_terms = normalize_topic_terms(extract_latin_terms(document.cleaned_text))

    prioritized: list[str] = []
    for token in (
        title_tokens + abstract_tokens + latin_terms + action_tokens + body_tokens
    ):
        if token in prioritized:
            continue
        if prioritized and not should_keep_supplemental_topic_term(token):
            continue
        prioritized.append(token)
    return prioritized[:6]


def detect_issues(document: ThesisDocument) -> list[Issue]:
    """Detect punctuation and colloquial writing issues."""

    return detect_issue_groups(document).all_issues


def detect_issue_groups(document: ThesisDocument) -> LocalIssueGroups:
    """Detect local issues without mixing format and writing concerns."""

    return LocalIssueGroups(
        format_issues=detect_punctuation_issues(document),
        writing_issues=detect_colloquial_issues(document),
    )


def detect_punctuation_issues(document: ThesisDocument) -> list[Issue]:
    """Detect punctuation issues with sentence-level locations."""

    issues: list[Issue] = []
    for section in document.sections:
        if section.skip_format_check:
            continue
        for paragraph in section.paragraphs:
            if paragraph.skip_format_check:
                continue
            for sentence in paragraph.sentences:
                sentence_text = sentence.text
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", sentence_text))
                has_latin = bool(re.search(r"[A-Za-z]", sentence_text))
                if has_chinese:
                    issues.extend(
                        detect_punctuation_chinese(
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
                        detect_punctuation_english(
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
                            rule_id=PUNCTUATION_REPEATED["rule_id"],
                            severity=PUNCTUATION_REPEATED["severity"],
                            message=PUNCTUATION_REPEATED["message"],
                            suggestion=PUNCTUATION_REPEATED["suggestion"],
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
        if section.skip_format_check:
            continue
        for paragraph in section.paragraphs:
            if paragraph.skip_format_check:
                continue
            for sentence in paragraph.sentences:
                for phrase, rule in COLLOQUIAL.items():
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


def extract_standard_keywords(standard: str) -> list[str]:
    """Split a report rubric standard into ordered topic keywords."""

    keywords: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9+\-]*|[\u4e00-\u9fff]+", standard):
        if re.search(r"[A-Za-z]", token):
            if token.lower() not in REPORT_STANDARD_GENERIC_ENGLISH_TERMS:
                keywords.append(token)
            continue
        for phrase in re.split(r"以及|的|与|和|及|或|在|对", token):
            normalized = normalize_standard_keyword(phrase)
            if normalized:
                keywords.append(normalized)
    return deduplicate_preserving_order(keywords)


def normalize_standard_keyword(keyword: str) -> str:
    """Remove instructional noise from one report-standard keyword."""

    compact = keyword.strip()
    changed = True
    while changed:
        changed = False
        for prefix in REPORT_STANDARD_GENERIC_PREFIXES:
            if compact.startswith(prefix) and len(compact) > len(prefix):
                compact = compact[len(prefix) :]
                changed = True
                break
    for suffix in REPORT_STANDARD_GENERIC_SUFFIXES:
        if compact.endswith(suffix) and len(compact) > len(suffix):
            compact = compact[: -len(suffix)]
            break
    if compact in REPORT_STANDARD_GENERIC_TERMS or len(compact) < 2:
        return ""
    return compact


def parse_report_standards(value: object) -> list[str]:
    """Normalize report rubric standards without importing scoring helpers."""

    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def annotate_report_topic_relevance(
    document: ThesisDocument, rubric_items: list[dict[str, object]]
) -> dict[str, object]:
    """Annotate report sections using configured rubric-standard coverage."""

    for paragraph in document.paragraphs:
        paragraph.topic_relevance_score = 0.0
        paragraph.topic_matched_keywords = []
        paragraph.topic_is_relevant = False
    for section in document.sections:
        section.topic_relevance_score = 0.0
        section.topic_relevant_word_count = 0
        section.topic_matched_keywords = []

    topic_keywords: list[str] = []
    earned_score = 0.0
    total_score = 0.0
    for rubric_item in rubric_items:
        criterion = str(rubric_item.get("criterion", "")).strip()
        standards = parse_report_standards(
            rubric_item.get("standard", rubric_item.get("standards", []))
        )
        max_score = float(rubric_item.get("score", rubric_item.get("max_score", 0)))
        total_score += max_score
        standard_keywords = [extract_standard_keywords(standard) for standard in standards]
        section_keywords = deduplicate_preserving_order(
            [keyword for keywords in standard_keywords for keyword in keywords]
        )
        topic_keywords.extend(section_keywords)
        section = next(
            (
                candidate
                for candidate in document.sections
                if criterion and criterion in candidate.title
            ),
            None,
        )
        if section is None:
            continue
        covered_standard_count = sum(
            any(matches_topic_keyword(section.content, keyword) for keyword in keywords)
            for keywords in standard_keywords
        )
        coverage_ratio = round(
            covered_standard_count / max(len(standard_keywords), 1), 4
        )
        annotate_report_section_topic_relevance(section, section_keywords, coverage_ratio)
        earned_score += max_score * coverage_ratio

    relevant_word_count = sum(
        section.topic_relevant_word_count for section in document.sections
    )
    return {
        "topic_keywords": deduplicate_preserving_order(topic_keywords),
        "relevant_word_count": relevant_word_count,
        "document_ratio": round(earned_score / max(total_score, 1.0), 4),
        "earned_score": round(earned_score, 4),
        "total_score": round(total_score, 4),
    }


def annotate_topic_relevance(document: ThesisDocument) -> dict[str, object]:
    """Annotate paragraphs and sections with topic relevance information."""

    topic_keywords = extract_topic_keywords(document)
    for paragraph in document.paragraphs:
        result = analyze_paragraph_topic_relevance(paragraph, topic_keywords)
        paragraph.topic_relevance_score = result["score"]
        paragraph.topic_matched_keywords = result["matched_keywords"]
        paragraph.topic_is_relevant = result["is_relevant"]
    relevant_word_count = 0
    for section in document.sections:
        annotate_section_topic_relevance(section, topic_keywords)
        relevant_word_count += section.topic_relevant_word_count
    document_ratio = round(relevant_word_count / max(document.total_word_count, 1), 4)
    return {
        "topic_keywords": topic_keywords,
        "relevant_word_count": relevant_word_count,
        "document_ratio": document_ratio,
    }


def annotate_section_topic_relevance(
    section: Section, topic_keywords: list[str]
) -> None:
    """Annotate a section with topic relevance derived from its paragraphs."""

    matched_keywords: set[str] = set()
    relevant_word_count = 0
    weighted_score = 0.0
    for paragraph in section.paragraphs:
        result = analyze_paragraph_topic_relevance(paragraph, topic_keywords)
        paragraph.topic_relevance_score = result["score"]
        paragraph.topic_matched_keywords = result["matched_keywords"]
        paragraph.topic_is_relevant = result["is_relevant"]
        if paragraph.topic_is_relevant:
            relevant_word_count += paragraph.word_count
        matched_keywords.update(paragraph.topic_matched_keywords)
        weighted_score += paragraph.topic_relevance_score * paragraph.word_count

    section.topic_relevance_score = round(
        weighted_score / max(section.word_count, 1), 4
    )
    section.topic_relevant_word_count = relevant_word_count
    section.topic_matched_keywords = sorted(matched_keywords)


def analyze_paragraph_topic_relevance(
    paragraph: Paragraph, topic_keywords: list[str]
) -> dict[str, object]:
    """Analyze a paragraph's topic relevance using keyword coverage."""

    if not topic_keywords:
        return {"score": 0.0, "matched_keywords": [], "is_relevant": False}
    matched_keywords = [
        keyword
        for keyword in topic_keywords
        if matches_topic_keyword(paragraph.text, keyword)
    ]
    total_weight = sum(topic_keyword_weight(keyword) for keyword in topic_keywords)
    matched_weight = sum(topic_keyword_weight(keyword) for keyword in matched_keywords)
    strong_match_count = sum(
        is_strong_topic_keyword(keyword) for keyword in matched_keywords
    )
    score = round(matched_weight / max(total_weight, 1.0), 4)
    is_relevant = (
        strong_match_count >= 2
        or matched_weight >= 1.8
        or score >= 0.38
        or (
            strong_match_count >= 1
            and paragraph.word_count <= 40
            and matched_weight >= 1.0
        )
    )
    return {
        "score": score,
        "matched_keywords": matched_keywords,
        "is_relevant": is_relevant,
    }


def annotate_report_section_topic_relevance(
    section: Section, topic_keywords: list[str], coverage_ratio: float
) -> None:
    """Annotate one matched report section and its paragraphs."""

    matched_keywords: set[str] = set()
    relevant_word_count = 0
    for paragraph in section.paragraphs:
        paragraph_matches = [
            keyword
            for keyword in topic_keywords
            if matches_topic_keyword(paragraph.text, keyword)
        ]
        paragraph.topic_relevance_score = round(
            len(paragraph_matches) / max(len(topic_keywords), 1), 4
        )
        paragraph.topic_matched_keywords = paragraph_matches
        paragraph.topic_is_relevant = bool(paragraph_matches)
        if paragraph.topic_is_relevant:
            relevant_word_count += paragraph.word_count
        matched_keywords.update(paragraph_matches)

    section.topic_relevance_score = coverage_ratio
    section.topic_relevant_word_count = relevant_word_count
    section.topic_matched_keywords = [
        keyword for keyword in topic_keywords if keyword in matched_keywords
    ]


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
        statistics.append(
            Statistic(
                label=f"主题相关度 - {section.title} / {child.title}",
                value=(
                    f"{child.topic_relevance_score * 100:.1f}% "
                    f"({child.topic_relevant_word_count}/{max(child.word_count, 1)})"
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


def split_technology_stack(
    technology_details: list[TechnologyStackItem],
) -> dict[str, list[str]]:
    """Split display technologies into software and hardware groups."""

    software_categories = {
        "software",
        "platform",
        "Web 框架",
        "向量检索",
        "图像处理",
        "工程基础设施",
        "推理框架",
        "数据库",
        "数据处理",
        "机器学习框架",
        "模型平台",
        "模型框架",
        "编程语言",
        "通信协议",
    }
    grouped = {
        "software_technology_stack": [],
        "hardware_technology_stack": [],
    }
    for item in technology_details:
        if item.category in software_categories:
            grouped["software_technology_stack"].append(item.name)
        elif item.category in {"hardware", "device"}:
            grouped["hardware_technology_stack"].append(item.name)
    return grouped


def tokenize(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin text with light filtering."""

    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    latin_terms = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}", text)
    tokens = chinese_terms + latin_terms
    return [token for token in tokens if len(token.strip()) >= 2]


def extract_compact_terms(text: str) -> list[str]:
    """Extract shorter, more stable terms from mixed Chinese and Latin text."""

    normalized = re.sub(r"[，。！？；：,.!?;:()\[\]（）【】\n\r\t]+", " ", text)
    segments = [segment.strip() for segment in normalized.split() if segment.strip()]
    tokens: list[str] = []
    for segment in segments:
        tokens.extend(
            re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}|[\u4e00-\u9fff]{2,6}", segment)
        )
    return [
        token
        for token in tokens
        if token not in STOPWORDS
        and token not in GENERIC_ANALYSIS_TERMS
        and token not in SECTION_HEADING_TERMS
        and is_topic_keyword_candidate(token)
    ]


def extract_tiktoken_keywords(text: str, top_k: int) -> list[str]:
    """Extract mixed-language keywords from tiktoken windows with light filtering."""

    if not text.strip():
        return []
    candidate_counts: Counter[str] = Counter()
    for segment in build_tiktoken_text_chunks(text):
        for keyword in extract_phrase_candidates(segment):
            candidate_counts[keyword] += 1

    ranked_candidates = sorted(
        candidate_counts.items(),
        key=lambda item: (
            keyword_candidate_score(item[0], item[1]),
            len(item[0]),
            item[0],
        ),
        reverse=True,
    )
    selected: list[str] = []
    for candidate, _ in ranked_candidates:
        if is_redundant_keyword_candidate(candidate, selected):
            continue
        selected.append(candidate)
        if len(selected) >= top_k * 2:
            break
    return normalize_topic_terms(selected)[:top_k]


def build_tiktoken_text_chunks(text: str, max_tokens: int = 48) -> list[str]:
    """Split original text into readable chunks, using tiktoken only for sizing."""

    raw_segments = [
        segment.strip()
        for segment in re.split(r"[，。！？；：\n\r]+", text)
        if segment.strip()
    ]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_token_count = 0

    for segment in raw_segments:
        segment_token_count = len(TIKTOKEN_ENCODING.encode(segment))
        if current_parts and current_token_count + segment_token_count > max_tokens:
            chunks.append("，".join(current_parts))
            current_parts = []
            current_token_count = 0
        current_parts.append(segment)
        current_token_count += segment_token_count

    if current_parts:
        chunks.append("，".join(current_parts))

    return [
        segment
        for segment in deduplicate_preserving_order(chunks + raw_segments)
        if segment and segment not in STOPWORDS
    ]


def extract_phrase_candidates(text: str) -> list[str]:
    """Extract compact phrase candidates from readable text fragments."""

    segments = [
        segment.strip()
        for segment in re.split(r"[，。！？；：,.!?;:()\[\]（）【】\n\r\t、/]+", text)
        if segment.strip()
    ]
    candidates: list[str] = []
    for segment in segments:
        for term in extract_latin_terms(segment):
            candidates.append(term)
        for phrase in extract_chinese_phrases(segment):
            candidates.append(phrase)
    return normalize_topic_terms(candidates)


def extract_chinese_phrases(text: str) -> list[str]:
    """Extract compact Chinese phrases from a sentence-like fragment."""

    cleaned = re.sub(r"[A-Za-z0-9+\-]+", " ", text)
    fragments = [
        fragment.strip()
        for fragment in re.split(r"[，。！？；：、\s]+", cleaned)
        if fragment.strip()
    ]
    phrases: list[str] = []
    for fragment in fragments:
        fragment = normalize_phrase_fragment(fragment)
        if not fragment:
            continue
        for match in re.findall(r"[\u4e00-\u9fff]{2,10}", fragment):
            normalized_match = normalize_domain_phrase(match)
            if not normalized_match:
                continue
            if is_topic_keyword_candidate_phrase(normalized_match):
                phrases.append(normalized_match)
    return deduplicate_preserving_order(phrases)


def normalize_phrase_fragment(fragment: str) -> str:
    """Trim common narrative prefixes from a Chinese fragment."""

    compact = fragment.strip()
    changed = True
    while changed:
        changed = False
        for prefix in PHRASE_PREFIXES:
            if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
                compact = compact[len(prefix) :]
                changed = True
                break
    return compact.strip()


def normalize_domain_phrase(phrase: str) -> str:
    """Normalize longer Chinese fragments into compact domain phrases."""

    compact = phrase.strip()
    for target in DOMAIN_KEY_PHRASES:
        if target in compact:
            return target
    if compact.endswith("和句子") and "段落" in compact:
        return "句子识别"
    if compact.startswith("识别") and len(compact) <= 6:
        suffix = compact[2:]
        if suffix in ACTION_OBJECT_TERMS:
            return f"{suffix}识别"
    if compact.startswith("解析") and len(compact) <= 6:
        return compact
    if compact.startswith("形成") and "数据结构" in compact:
        return "数据结构"
    compact = compact.strip("的与及和并且或者以及在对将把为是中上下")
    if not compact:
        return ""
    if len(compact) > 6 and not re.search(r"[A-Za-z0-9]", compact):
        return ""
    if not is_topic_keyword_candidate(compact):
        return ""
    return compact


def is_topic_keyword_candidate_phrase(phrase: str) -> bool:
    """Reject generic Chinese phrase candidates."""

    if phrase in GENERIC_TOPIC_TERMS or phrase in GENERIC_ANALYSIS_TERMS:
        return False
    if phrase in GENERIC_PHRASE_TERMS:
        return False
    if phrase in TOPIC_NOISE_GENERIC_TERMS:
        return False
    if phrase.startswith(PHRASE_PREFIXES):
        return False
    if any(fragment in phrase for fragment in PHRASE_NOISE_FRAGMENTS):
        return False
    return not any(fragment in phrase for fragment in TOPIC_NOISE_GENERIC_FRAGMENTS)


def keyword_candidate_score(candidate: str, frequency: int) -> float:
    """Score a keyword candidate by frequency and phrase specificity."""

    length_bonus = min(len(candidate), 6) * 0.15
    latin_bonus = 0.4 if re.search(r"[A-Za-z0-9]", candidate) else 0.0
    return frequency + length_bonus + latin_bonus


def is_redundant_keyword_candidate(candidate: str, selected: list[str]) -> bool:
    """Drop candidates that are mostly covered by a stronger selected phrase."""

    for existing in selected:
        if candidate == existing:
            return True
        if re.search(r"[A-Za-z0-9]", candidate) or re.search(r"[A-Za-z0-9]", existing):
            continue
        if candidate in existing and len(existing) >= len(candidate):
            return True
    return False


def extract_action_phrases(text: str) -> list[str]:
    """Extract explicit functionality phrases that are often central to the topic."""

    if not text.strip():
        return []
    phrases: list[str] = []
    for segment in split_action_segments(text):
        compact = clean_action_phrase(segment)
        if not compact:
            continue
        phrases.extend(extract_action_phrase_candidates(compact))
    return deduplicate_preserving_order(phrases)


def split_action_segments(text: str) -> list[str]:
    """Split text into short action-oriented segments."""

    return [
        segment.strip()
        for segment in re.split(
            r"[，。！？；：,.!?;:()\[\]（）【】\n\r\t、/]+|(?:和|与|及|并|以及)", text
        )
        if segment.strip()
    ]


def extract_action_phrase_candidates(text: str) -> list[str]:
    """Extract compact action phrases from a cleaned sentence fragment."""

    candidates: list[str] = []
    for verb in ACTION_VERBS:
        candidates.extend(
            match.group(1)
            for match in re.finditer(rf"([\u4e00-\u9fff]{{2,6}}{verb})", text)
        )
        candidates.extend(
            match.group(1)
            for match in re.finditer(rf"({verb}[\u4e00-\u9fff]{{2,4}})", text)
        )
    return [
        phrase
        for phrase in deduplicate_preserving_order(candidates)
        if phrase not in STOPWORDS
        and is_action_phrase_candidate(phrase)
        and is_specific_action_phrase(phrase)
    ]


def extract_paragraph_action_keywords(paragraphs: list[Paragraph]) -> list[str]:
    """Pick a small set of action phrases with paragraph-level coverage."""

    selected: list[str] = []
    for paragraph in paragraphs:
        paragraph_phrases = extract_action_phrases(paragraph.text)
        if not paragraph_phrases:
            continue
        selected.extend(paragraph_phrases[:2])
    return deduplicate_preserving_order(selected)


def extract_latin_terms(text: str) -> list[str]:
    """Extract Latin technical terms that keyword windows may not prioritize well."""

    latin_terms = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{1,}", text)
    return [term for term in latin_terms if is_topic_keyword_candidate(term)]


def split_title_keywords(title: str) -> list[str]:
    """Split the title into compact keyword-like units."""

    parts = re.split(r"[的与及和：:（）()\-\s]+", title)
    keywords: list[str] = []
    for part in parts:
        compact = normalize_title_phrase(part.strip())
        if not compact or compact in STOPWORDS or len(compact) > 10:
            continue
        keywords.append(compact)
    return deduplicate_preserving_order(keywords)


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


def deduplicate_preserving_order(values: list[str]) -> list[str]:
    """Deduplicate a list while preserving order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def contains_term(text: str, term: str) -> bool:
    """Check whether a term appears in the text with simple boundary handling."""

    if re.search(r"[A-Za-z0-9]", term):
        pattern = re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", re.IGNORECASE
        )
        return pattern.search(text) is not None
    return term in text


def matches_topic_keyword(text: str, keyword: str) -> bool:
    """Match a topic keyword with a fallback for longer Chinese phrases."""

    if contains_term(text, keyword):
        return True
    if re.search(r"[A-Za-z0-9]", keyword):
        return False
    if len(keyword) < 4:
        return False
    subterms = [keyword[index : index + 3] for index in range(len(keyword) - 2)]
    hit_count = sum(subterm in text for subterm in subterms)
    return hit_count >= max(2, (len(subterms) * 2 + 2) // 3)


def topic_keyword_weight(keyword: str) -> float:
    """Assign a higher weight to stronger topic keywords."""

    if re.search(r"[A-Za-z0-9]", keyword):
        return 1.2
    if len(keyword) >= 4:
        return 1.0
    return 0.6


def is_strong_topic_keyword(keyword: str) -> bool:
    """Decide whether a keyword is strong enough to anchor paragraph relevance."""

    if re.search(r"[A-Za-z0-9]", keyword):
        return True
    return len(keyword) >= 4


def is_topic_keyword_candidate(token: str) -> bool:
    """Keep topic keywords compact and readable."""

    if not token or len(token) < 2:
        return False
    if re.search(r"[A-Za-z0-9]", token):
        return len(token) <= 24
    if len(token) > 8:
        return False
    if re.fullmatch(r"\d+(\.\d+)*", token):
        return False
    if token.endswith(("一个用于", "辅助系", "研究背景与目标")):
        return False
    return 2 <= len(token) <= 6


def is_action_phrase_candidate(token: str) -> bool:
    """Allow slightly longer action phrases than generic topic terms."""

    if not token or len(token) < 2:
        return False
    if re.search(r"[A-Za-z0-9]", token):
        return False
    if len(token) > 8:
        return False
    return not re.fullmatch(r"\d+(\.\d+)*", token)


def normalize_topic_terms(tokens: list[str]) -> list[str]:
    """Normalize and filter topic term candidates."""

    normalized: list[str] = []
    for token in tokens:
        compact = token.strip("，。！？；：,.!?;:()[]（）【】")
        if not compact:
            continue
        compact = normalize_title_phrase(compact)
        if compact in GENERIC_ANALYSIS_TERMS or compact in SECTION_HEADING_TERMS:
            continue
        if compact in TOPIC_NOISE_GENERIC_TERMS:
            continue
        if (
            compact.startswith(("本文", "本章", "该模块", "本系统"))
            and len(compact) > 2
        ):
            continue
        if compact.endswith(("本章", "后续", "部分")):
            continue
        if re.fullmatch(r"\d+(\.\d+)*", compact):
            continue
        if re.search(r"[A-Za-z0-9]", compact):
            normalized.append(compact)
            continue
        if any(
            stop_fragment in compact
            for stop_fragment in (
                "用于",
                "目标",
                "内容",
                "模块",
                "后续",
                "基础",
                "结果",
                "整体",
                "质量",
                "任务",
                "报告",
            )
        ):
            continue
        if compact in GENERIC_TOPIC_TERMS:
            continue
        if any(fragment in compact for fragment in TOPIC_NOISE_GENERIC_FRAGMENTS):
            continue
        normalized.append(compact)
    return deduplicate_preserving_order(normalized)


def normalize_title_phrase(token: str) -> str:
    """Trim generic title suffixes while preserving the core subject phrase."""

    compact = token.strip()
    for prefix in ("基于", "面向"):
        if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
            compact = compact[len(prefix) :]
    for suffix in TITLE_SUFFIXES:
        if compact.endswith(suffix) and len(compact) > len(suffix) + 1:
            compact = compact[: -len(suffix)]
            break
    return compact.strip()


def should_keep_supplemental_topic_term(token: str) -> bool:
    """Keep only stronger supplemental terms once core keywords are already present."""

    if re.search(r"[A-Za-z0-9]", token):
        return True
    if len(token) < 4:
        return False
    return token not in GENERIC_TOPIC_TERMS


def clean_action_phrase(phrase: str) -> str:
    """Trim context noise from action-like phrases extracted from the body."""

    compact = phrase.strip("，。！？；：,.!?;:()[]（）【】")
    for marker in ("负责", "可以"):
        if marker in compact:
            compact = compact.split(marker, 1)[-1]
    changed = True
    while changed:
        changed = False
        for prefix in ACTION_PREFIXES:
            if compact.startswith(prefix) and len(compact) > len(prefix) + 1:
                compact = compact[len(prefix) :]
                changed = True
                break
    compact = compact.lstrip("的与和及并可会将对把在从为")
    if "的" in compact:
        tail = compact.split("的")[-1]
        if len(tail) >= 4:
            compact = tail
    compact = compact.strip()
    if compact in GENERIC_TOPIC_TERMS:
        return ""
    if any(fragment in compact for fragment in ACTION_NOISE_FRAGMENTS):
        return ""
    return compact


def is_specific_action_phrase(phrase: str) -> bool:
    """Reject overly generic action phrases that are poor topic anchors."""

    if phrase in GENERIC_TOPIC_TERMS:
        return False
    if any(fragment in phrase for fragment in ACTION_NOISE_FRAGMENTS):
        return False
    if phrase.startswith(ACTION_VERBS):
        suffix = phrase[2:]
        return (
            suffix not in GENERIC_TOPIC_TERMS and suffix not in GENERIC_ANALYSIS_TERMS
        )
    if phrase.endswith(ACTION_VERBS):
        prefix = phrase[:-2]
        return (
            prefix not in GENERIC_TOPIC_TERMS and prefix not in GENERIC_ANALYSIS_TERMS
        )
    return True


def should_match_colloquial(
    phrase: str, sentence_text: str, rule: dict[str, object]
) -> bool:
    """Filter overly broad colloquial rules to reduce obvious false positives."""

    if len(phrase) == 1 and not bool(rule.get("allow_single_char", False)):
        return False
    if phrase == "特别" and any(
        token in sentence_text for token in ("特别是", "特别是在")
    ):
        return False
    if phrase == "然后" and any(
        token in sentence_text for token in ("然后进行", "然后对", "然后将")
    ):
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


GENERIC_ANALYSIS_TERMS = {
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

GENERIC_TOPIC_TERMS = {
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

GENERIC_PHRASE_TERMS = {
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

REPORT_STANDARD_GENERIC_TERMS = {
    "与",
    "中",
    "环节",
    "主流",
    "使用",
    "关键字",
    "常见",
    "当前",
    "所选",
    "指标",
    "应用",
    "技术",
    "发展",
    "时",
    "重点使用",
    "阐述",
    "需要",
    "相关",
    "现有",
    "风险",
}

REPORT_STANDARD_GENERIC_PREFIXES = (
    "重点使用",
    "常见",
    "当前",
    "完成",
    "所选",
    "使用",
    "主流",
    "现有",
    "相关",
    "阐述",
)

REPORT_STANDARD_GENERIC_SUFFIXES = (
    "可能需要联系",
    "需要掌握",
    "过程中",
    "等指标",
    "指标",
    "相关",
    "需要",
    "时",
    "中",
)

REPORT_STANDARD_GENERIC_ENGLISH_TERMS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "or",
    "related",
    "the",
    "to",
    "use",
    "using",
    "with",
}

PHRASE_PREFIXES = (
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

PHRASE_NOISE_FRAGMENTS = ("往往需要", "快速了解", "本章介绍", "本章说明", "系统需要")

DOMAIN_KEY_PHRASES = (
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

ACTION_OBJECT_TERMS = {"章节", "段落", "句子", "结构", "内容", "数据"}

ACTION_VERBS = (
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

ACTION_PREFIXES = (
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

ACTION_NOISE_FRAGMENTS = (
    "辅助系统",
    "整体质量",
    "课程论文",
    "老师需要",
    "很多场景",
    "结果不仅",
    "模块负责",
)

TITLE_SUFFIXES = ("设计与实现", "设计实现", "设计", "实现", "研究", "方法", "方案")

SECTION_HEADING_TERMS = {
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


def detect_punctuation_chinese(
    section: Section, paragraph_index: int, sentence_index: int, sentence_text: str
) -> list[Issue]:
    """Detect ASCII punctuation used in Chinese sentences."""

    issues: list[Issue] = []
    for mark, rule in PUNCTUATION_CHINESE.items():
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


def detect_punctuation_english(
    section: Section, paragraph_index: int, sentence_index: int, sentence_text: str
) -> list[Issue]:
    """Detect Chinese punctuation used in English-like sentences."""

    issues: list[Issue] = []
    for mark, rule in PUNCTUATION_ENGLISH.items():
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
