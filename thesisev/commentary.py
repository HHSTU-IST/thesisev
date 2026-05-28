"""Comment generation helpers."""

from __future__ import annotations


def generate_comment(
    title: str,
    keywords: list[str],
    technology_stack: list[str],
    score: int,
    issue_count: int,
) -> str:
    """Generate a concise thesis evaluation comment."""

    focus_keywords = "、".join(keywords[:3]) if keywords else "研究主题"
    tech_summary = "、".join(technology_stack[:3]) if technology_stack else "相关技术"
    quality_phrase = score_to_phrase(score)

    if issue_count == 0:
        issue_summary = "全文表达较为规范，暂未发现明显格式或口语化问题。"
    elif issue_count <= 3:
        issue_summary = "文中存在少量格式或表达细节需要润色，但整体不影响主要内容呈现。"
    else:
        issue_summary = "文中格式一致性和学术表达仍有较明显的优化空间，建议集中修订。"

    return (
        f"论文围绕{focus_keywords}展开，内容组织{quality_phrase}，能够体现对{tech_summary}的关注。"
        f"{issue_summary}综合来看，该文具有一定完成度，当前评分约为 {score} 分。"
    )


def score_to_phrase(score: int) -> str:
    """Map a score to a short quality phrase."""

    if score >= 90:
        return "较为完整"
    if score >= 80:
        return "基本清晰"
    if score >= 70:
        return "较为基础"
    return "仍需加强"
