const providerDefaults = {
  deepseek: "deepseek-chat",
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest"
};

const state = {
  result: null,
  activeIssueFilter: "all",
  activeSectionId: null,
  activeIssueKey: null,
  activeScoreKey: null,
};

const form = document.getElementById("evaluate-form");
const providerInput = document.getElementById("provider");
const modelInput = document.getElementById("model");
const presetInput = document.getElementById("preset");
const fileInput = document.getElementById("file");
const submitButton = document.getElementById("submit-button");
const statusNode = document.getElementById("status");
const resultsNode = document.getElementById("results");
const exportMdButton = document.getElementById("export-md");
const refreshHistoryButton = document.getElementById("refresh-history");
const reuseHintNode = document.getElementById("reuse-hint");

providerInput.addEventListener("change", () => {
  modelInput.value = providerDefaults[providerInput.value] || "";
});

exportMdButton.addEventListener("click", () => exportResult("md"));
refreshHistoryButton.addEventListener("click", () => {
  void loadHistory();
});

void loadHistory();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetRenderedResults();
  setLoading(true, "正在分析论文，请稍候...");

  try {
    const response = await submitEvaluation();
    if (!response.ok) {
      throw new Error(response.detail || "请求失败");
    }
    state.result = response.data;
    state.activeIssueFilter = "all";
    state.activeIssueKey = null;
    state.activeScoreKey = null;
    state.activeSectionId =
      response.data.document.root_sections[0]?.identifier || null;
    renderResults();
    void loadHistory();
    setLoading(false, "分析完成");
  } catch (error) {
    setLoading(false, error.message || "处理失败");
  }
});

function resetRenderedResults() {
  state.result = null;
  state.activeIssueFilter = "all";
  state.activeSectionId = null;
  state.activeIssueKey = null;
  state.activeScoreKey = null;
  resultsNode.classList.add("hidden");
  document.getElementById("doc-title").textContent = "-";
}

async function submitEvaluation() {
  const formData = new FormData();
  formData.append("preset", presetInput.value);
  formData.append("provider", providerInput.value);
  formData.append("model", modelInput.value);
  formData.append("temperature", document.getElementById("temperature").value);
  formData.append("max_tokens", document.getElementById("max_tokens").value);

  const file = fileInput.files[0];
  if (!file) {
    throw new Error("请先选择论文文件");
  }
  formData.append("file", file);
  return postForm("/evaluate/upload", formData);
}

async function postForm(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  const payload = await response.json();
  if (!response.ok) {
    return { ok: false, detail: payload.detail || "请求失败" };
  }
  return payload;
}

async function loadHistory() {
  const response = await fetch("/history");
  const payload = await response.json();
  renderHistory(payload.data?.items || []);
}

function renderResults() {
  const data = state.result;
  if (!data) {
    return;
  }

  resultsNode.classList.remove("hidden");
  document.getElementById("comment").textContent = data.comment || "暂无内容评价";
  document.getElementById("score-badge").textContent = `${data.score} 分`;
  document.getElementById("doc-title").textContent = data.document.title;
  document.getElementById("doc-type").textContent = data.document.source_type;
  document.getElementById("doc-level-one-sections").textContent = String(
    countSectionsByLevel(data.document.sections, 1),
  );
  document.getElementById("doc-level-two-sections").textContent = String(
    countSectionsByLevel(data.document.sections, 2),
  );
  document.getElementById("doc-level-three-sections").textContent = String(
    countSectionsByLevel(data.document.sections, 3),
  );
  document.getElementById("topic-ratio").textContent = `${(
    data.topic_relevance_ratio * 100
  ).toFixed(1)}%`;

  const modelMeta = data.metadata?.model || {};
  const scoreSource = data.metadata?.score_source || "local";
  const commentSource = data.metadata?.comment_source || "fallback";
  const roles = data.metadata?.evaluation_roles || {};
  renderModelMeta(modelMeta, scoreSource, commentSource, roles);

  renderScoreDetail(data.metadata?.score_detail || null);
  renderIssueFilters(data.issues);
  renderIssues(data.issues);
  renderTechList("software-tech-list", data.software_technology_stack || []);
  renderTechList("hardware-tech-list", data.hardware_technology_stack || []);
  renderSectionTree(data.document.root_sections);
  renderFocusPanel();
}

function renderScoreDetail(scoreDetail) {
  const metaNode = document.getElementById("score-detail-meta");
  const listNode = document.getElementById("score-detail-list");
  listNode.innerHTML = "";

  if (!scoreDetail || !scoreDetail.criteria || !scoreDetail.criteria.length) {
    metaNode.textContent = "暂无评分明细";
    const li = document.createElement("li");
    li.textContent = "未返回规则化评分明细";
    listNode.appendChild(li);
    return;
  }

  metaNode.textContent =
    `分数来源: ${formatScoreSource(scoreDetail.score_source || "local")} | ` +
    `原始分: ${scoreDetail.raw_score}/${scoreDetail.raw_total} | ` +
    `百分制: ${scoreDetail.score}`;
  scoreDetail.criteria.forEach((item) => {
    const key = buildScoreKey(item);
    const li = document.createElement("li");
    li.className = `score-item${state.activeScoreKey === key ? " active" : ""}`;
    const deductions = (item.deductions || []).join("；") || "无明显扣分项";
    const title = document.createElement("strong");
    title.textContent = `${item.name}: ${item.score}/${item.max_score}`;
    li.appendChild(title);

    li.appendChild(buildSubline(`方法: ${formatEvaluationMethod(item)}`));
    appendScoreEvidence(li, item);
    li.appendChild(buildSubline(`扣分: ${deductions}`));
    li.addEventListener("click", () => {
      state.activeScoreKey = key;
      state.activeIssueKey = null;
      renderScoreDetail(state.result.metadata?.score_detail || null);
      renderIssues(state.result.issues);
      renderFocusPanel();
    });
    listNode.appendChild(li);
  });
}

function countSectionsByLevel(sections, level) {
  return sections.filter((section) => section.level === level).length;
}

function appendScoreEvidence(parent, item) {
  const evidenceItems = item.evidence || [];
  if (!evidenceItems.length) {
    parent.appendChild(buildSubline("证据: 暂无证据"));
    return;
  }
  if (item.key !== "iot_format" && item.name !== "格式规范") {
    parent.appendChild(buildSubline(`证据: ${evidenceItems.join("；")}`));
    return;
  }

  const container = document.createElement("div");
  container.className = "score-evidence";
  const label = document.createElement("div");
  label.className = "score-evidence-label";
  label.textContent = "证据";
  container.appendChild(label);

  const list = document.createElement("ul");
  evidenceItems.forEach((text) => {
    const itemNode = document.createElement("li");
    itemNode.textContent = text;
    list.appendChild(itemNode);
  });
  container.appendChild(list);
  parent.appendChild(container);
}

function renderIssueFilters(issues) {
  const node = document.getElementById("issue-filters");
  node.innerHTML = "";

  const counts = issues.reduce(
    (acc, issue) => {
      acc[issue.category] = (acc[issue.category] || 0) + 1;
      return acc;
    },
    { all: issues.length },
  );

  Object.entries(counts).forEach(([key, value]) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `filter-chip${state.activeIssueFilter === key ? " active" : ""
      }`;
    button.textContent = key === "all" ? `全部 ${value}` : `${key} ${value}`;
    button.addEventListener("click", () => {
      state.activeIssueFilter = key;
      renderIssueFilters(issues);
      renderIssues(issues);
    });
    node.appendChild(button);
  });
}

function renderIssues(issues) {
  const filteredIssues =
    state.activeIssueFilter === "all"
      ? issues
      : issues.filter((issue) => issue.category === state.activeIssueFilter);

  const node = document.getElementById("issues-list");
  node.innerHTML = "";

  if (!filteredIssues.length) {
    const li = document.createElement("li");
    li.textContent = "当前筛选条件下没有问题项";
    node.appendChild(li);
    return;
  }

  filteredIssues.forEach((issue) => {
    const key = buildIssueKey(issue);
    const li = document.createElement("li");
    li.className = `issue-item${state.activeIssueKey === key ? " active" : ""}`;
    const title = document.createElement("strong");
    title.textContent = `[${issue.category}] ${issue.section_identifier} ${issue.section_title}`;
    li.appendChild(title);
    const message = document.createElement("div");
    message.textContent = issue.message;
    li.appendChild(message);
    li.appendChild(buildIssueExcerpt(issue));
    li.appendChild(buildSubline(`建议: ${issue.suggestion}`));
    li.addEventListener("click", () => {
      state.activeIssueKey = key;
      state.activeScoreKey = null;
      state.activeSectionId = issue.section_identifier;
      renderScoreDetail(state.result.metadata?.score_detail || null);
      renderIssues(issues);
      renderSectionTree(state.result.document.root_sections);
      renderFocusPanel();
    });
    node.appendChild(li);
  });
}

function buildIssueExcerpt(issue) {
  const node = document.createElement("div");
  node.className = "issue-excerpt";

  const excerpt = document.createElement("div");
  excerpt.textContent = `所在片段: ${issue.excerpt || "未提供片段"}`;
  node.appendChild(excerpt);

  if (issue.matched_text) {
    const matched = document.createElement("div");
    matched.className = "issue-match";
    matched.textContent = `命中内容: ${issue.matched_text}`;
    node.appendChild(matched);
  }

  return node;
}

function buildIssueKey(issue) {
  return [
    issue.category || "",
    issue.section_identifier || "",
    issue.section_title || "",
    issue.message || "",
    issue.suggestion || "",
  ].join("::");
}

function buildScoreKey(item) {
  return [item.key || "", item.name || ""].join("::");
}

function renderFocusPanel() {
  const node = document.getElementById("focus-panel");
  node.replaceChildren();

  const scoreItem = findActiveScoreItem();
  if (scoreItem) {
    const block = buildFocusBlock(
      "评分扣分项",
      `${scoreItem.name}: ${scoreItem.score}/${scoreItem.max_score}`,
    );
    appendFocusList(block, "扣分项", scoreItem.deductions, "无明显扣分项");
    appendFocusList(block, "证据", scoreItem.evidence, "暂无证据");
    appendFocusList(block, "建议", scoreItem.suggestions, "暂无建议");
    node.appendChild(block);
    return;
  }

  const issue = findActiveIssue();
  if (issue) {
    const block = buildFocusBlock(
      "格式问题",
      `[${issue.category}] ${issue.section_identifier} ${issue.section_title}`,
    );
    block.appendChild(buildIssueExcerpt(issue));
    block.appendChild(buildSubline(`建议: ${issue.suggestion}`, "focus-meta"));
    node.appendChild(block);
    return;
  }

  const section = findSectionByIdentifier(
    state.result?.document?.root_sections || [],
    state.activeSectionId,
  );
  if (section) {
    const block = buildFocusBlock(
      "章节内容",
      `${section.identifier} ${section.title}`,
    );
    block.appendChild(
      buildSubline(
        `内容占比 ${(section.ratio * 100).toFixed(1)}% · ` +
        `主题相关度 ${(section.topic_relevance_score * 100).toFixed(1)}%`,
        "focus-meta",
      ),
    );
    block.appendChild(
      buildSubline(section.content || "该章节暂无正文内容。", "focus-paragraph"),
    );
    node.appendChild(block);
    return;
  }

  node.appendChild(
    buildSubline("点击评分项、章节树或问题项后，这里会显示对应详情。", "focus-empty"),
  );
}

function findActiveScoreItem() {
  const criteria = state.result?.metadata?.score_detail?.criteria || [];
  return criteria.find((item) => buildScoreKey(item) === state.activeScoreKey);
}

function findActiveIssue() {
  return (state.result?.issues || []).find(
    (issue) => buildIssueKey(issue) === state.activeIssueKey,
  );
}

function findSectionByIdentifier(sections, identifier) {
  for (const section of sections) {
    if (section.identifier === identifier) {
      return section;
    }
    const child = findSectionByIdentifier(section.children || [], identifier);
    if (child) {
      return child;
    }
  }
  return null;
}

function buildFocusBlock(kickerText, titleText) {
  const block = document.createElement("div");
  block.className = "focus-block";
  block.appendChild(buildSubline(kickerText, "focus-kicker"));
  const title = document.createElement("h3");
  title.textContent = titleText;
  block.appendChild(title);
  return block;
}

function appendFocusList(parent, label, items, fallback) {
  parent.appendChild(buildSubline(label, "focus-meta"));
  const list = document.createElement("ul");
  list.className = "list";
  (items && items.length ? items : [fallback]).forEach((item) => {
    const row = document.createElement("li");
    row.textContent = item;
    list.appendChild(row);
  });
  parent.appendChild(list);
}

function renderTechList(elementId, technologyStack) {
  renderTextList(
    elementId,
    technologyStack.length ? technologyStack : ["无"],
  );
}

const MAX_VISIBLE_SECTION_LEVEL = 2;

function renderSectionTree(rootSections) {
  const node = document.getElementById("section-tree");
  node.innerHTML = "";
  const visibleRootSections = rootSections.filter(
    (section) => section.level <= MAX_VISIBLE_SECTION_LEVEL,
  );
  if (!visibleRootSections.length) {
    node.textContent = "未识别到章节结构";
    return;
  }
  const list = document.createElement("ul");
  list.className = "tree-list";
  visibleRootSections.forEach((section) => {
    list.appendChild(buildSectionNode(section));
  });
  node.appendChild(list);
}

function buildSectionNode(section) {
  const item = document.createElement("li");
  item.className = "tree-item";

  const button = document.createElement("button");
  button.type = "button";
  button.className = `tree-button${state.activeSectionId === section.identifier ? " active" : ""
    }`;
  const header = document.createElement("div");
  header.className = "tree-header";
  const title = document.createElement("strong");
  title.textContent = `${section.identifier} ${section.title}`;
  const summary = document.createElement("span");
  summary.textContent =
    `内容占比 ${(section.ratio * 100).toFixed(1)}% / ` +
    `${section.subtree_word_count} 字`;
  header.appendChild(title);
  header.appendChild(summary);

  const meta = document.createElement("div");
  meta.className = "tree-meta";
  meta.textContent =
    `层级 L${section.level} · 段落 ${section.paragraphs.length} · ` +
    `主题相关度 ${(section.topic_relevance_score * 100).toFixed(1)}%`;

  button.appendChild(header);
  button.appendChild(meta);
  button.addEventListener("click", () => {
    state.activeSectionId = section.identifier;
    state.activeIssueKey = null;
    state.activeScoreKey = null;
    renderScoreDetail(state.result.metadata?.score_detail || null);
    renderSectionTree(state.result.document.root_sections);
    renderIssues(state.result.issues);
    renderFocusPanel();
  });
  item.appendChild(button);

  const visibleChildren = (section.children || []).filter(
    (section) => section.level <= MAX_VISIBLE_SECTION_LEVEL,
  );
  if (visibleChildren.length) {
    const childList = document.createElement("ul");
    childList.className = "tree-list";
    visibleChildren.forEach((child) => {
      childList.appendChild(buildSectionNode(child));
    });
    item.appendChild(childList);
  }
  return item;
}

function renderTextList(id, items) {
  const node = document.getElementById(id);
  node.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    node.appendChild(li);
  });
}

function renderHistory(items) {
  const node = document.getElementById("history-list");
  node.innerHTML = "";
  if (!items.length) {
    const li = document.createElement("li");
    li.textContent = "暂无历史评审记录";
    node.appendChild(li);
    return;
  }
  items.forEach((item) => {
    const li = document.createElement("li");
    const title = document.createElement("strong");
    title.textContent = item.title;
    li.appendChild(title);

    const meta = document.createElement("div");
    meta.className = "history-subline";
    meta.textContent =
      `${item.created_at} · ${item.source_type} · ${item.score} 分 · ` +
      `${Math.round((item.topic_relevance_ratio || 0) * 100)}% 主题相关`;
    li.appendChild(meta);

    li.appendChild(buildSubline(item.comment, "history-subline"));
    node.appendChild(li);
  });
}

function renderModelMeta(modelMeta, scoreSource, commentSource, roles) {
  const node = document.getElementById("model-meta");
  node.replaceChildren();

  node.appendChild(
    buildMetaChip(
      `模型 ${modelMeta.provider || "-"} / ${modelMeta.model || "-"}`,
    ),
  );
  node.appendChild(
    buildMetaChip(
      `Key ${modelMeta.available ? "可用" : "不可用"}`,
      modelMeta.available ? "is-ready" : "is-muted",
    ),
  );
  node.appendChild(
    buildMetaChip(formatScoreSource(scoreSource), "is-score"),
  );
  node.appendChild(
    buildMetaChip(
      formatCommentSource(commentSource),
      commentSource === "llm" ? "is-llm" : "is-fallback",
    ),
  );
  node.appendChild(
    buildMetaChip(
      `格式检测与评价: ${formatRoleSource(roles.format_evaluation || "local")}`,
      "is-score",
    ),
  );
}

function buildMetaChip(text, className = "") {
  const chip = document.createElement("span");
  chip.className = `meta-chip${className ? ` ${className}` : ""}`;
  chip.textContent = text;
  return chip;
}

function buildSubline(text, className = "issue-subline") {
  const node = document.createElement("div");
  node.className = className;
  node.textContent = text;
  return node;
}

function exportResult(format) {
  if (!state.result) {
    setLoading(false, "暂无可导出的结果");
    return;
  }
  downloadFile(
    "thesisev-result.md",
    "text/markdown;charset=utf-8",
    buildMarkdown(state.result),
  );
}

function buildMarkdown(data) {
  const statistics = data.statistics
    .map((item) => `- ${item.label}: ${item.value}`)
    .join("\n");
  const issues = data.issues.length
    ? data.issues
      .map(
        (issue) =>
          [
            `- [${issue.category}] ${issue.section_identifier} ${issue.section_title}: ${issue.message}`,
            `  - 所在片段：${issue.excerpt || "未提供片段"}`,
            issue.matched_text ? `  - 命中内容：${issue.matched_text}` : "",
            `  - 建议：${issue.suggestion}`,
          ]
            .filter(Boolean)
            .join("\n"),
      )
      .join("\n")
    : "- 未检测到明显问题";
  const softwareTech = buildTechnologyStackMarkdown(
    data.software_technology_stack || [],
  );
  const hardwareTech = buildTechnologyStackMarkdown(
    data.hardware_technology_stack || [],
  );
  const scoreDetail = buildScoreDetailMarkdown(data.metadata?.score_detail || null);
  const formatRequirements = buildFormatRequirementsMarkdown(
    data.metadata?.format_requirements || null,
  );

  return [
    `# ${data.document.title}`,
    "",
    `- 分数: ${data.score}`,
    `- 类型: ${data.document.source_type}`,
    `- 主题相关度: ${(data.topic_relevance_ratio * 100).toFixed(1)}%`,
    `- 分数来源: ${formatScoreSource(data.metadata?.score_source || "local")}`,
    `- 内容评价来源: ${formatCommentSource(data.metadata?.comment_source || "fallback")}`,
    `- 格式检测与评价: ${formatRoleSource(data.metadata?.evaluation_roles?.format_evaluation || "local")}`,
    "",
    "## 内容评价",
    "",
    data.comment,
    "",
    "## 统计结果",
    "",
    statistics,
    "",
    "## 评分明细",
    "",
    scoreDetail,
    "",
    "## 格式检测",
    "",
    issues,
    "",
    "## 软件技术栈",
    "",
    softwareTech,
    "",
    "## 硬件技术栈",
    "",
    hardwareTech,
    "",
    "## 格式要求",
    "",
    formatRequirements,
  ].join("\n");
}

function buildTechnologyStackMarkdown(technologyStack) {
  return technologyStack.length
    ? technologyStack.map((item) => `- ${item}`).join("\n")
    : "- 无";
}

function buildScoreDetailMarkdown(scoreDetail) {
  if (!scoreDetail || !scoreDetail.criteria || !scoreDetail.criteria.length) {
    return "- 未返回评分明细";
  }
  return scoreDetail.criteria
    .map((item) => {
      const evidence = (item.evidence || []).join("；") || "暂无证据";
      const deductions = (item.deductions || []).join("；") || "无明显扣分项";
      const suggestions = (item.suggestions || []).join("；") || "暂无建议";
      return [
        `- ${item.name}: ${item.score}/${item.max_score}`,
        `  - 方法: ${formatEvaluationMethod(item)}`,
        `  - 证据: ${evidence}`,
        `  - 扣分: ${deductions}`,
        `  - 建议: ${suggestions}`,
      ].join("\n");
    })
    .concat([
      `- 分数来源: ${formatScoreSource(scoreDetail.score_source || "local")}`,
      `- 原始分: ${scoreDetail.raw_score}/${scoreDetail.raw_total}`,
      `- 百分制: ${scoreDetail.score}`,
    ])
    .join("\n");
}

function buildFormatRequirementsMarkdown(formatRequirements) {
  if (
    !formatRequirements ||
    !formatRequirements.items ||
    !formatRequirements.items.length
  ) {
    return "- 未上传格式要求";
  }
  return formatRequirements.items
    .map((item) => `- ${item.label}: ${item.value}`)
    .join("\n");
}

function downloadFile(filename, mimeType, content) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function formatScoreSource(source) {
  if (source === "local") {
    return "分数: 本地程序";
  }
  if (source === "rule_engine") {
    return "分数: 规则引擎";
  }
  return source || "-";
}

function formatCommentSource(source) {
  if (source === "llm") {
    return "内容评价: LLM";
  }
  if (source === "fallback") {
    return "内容评价: 本地回退";
  }
  return source || "-";
}

function formatRoleSource(source) {
  if (source === "local") {
    return "本地程序";
  }
  if (source === "llm") {
    return "LLM";
  }
  if (source === "fallback") {
    return "本地回退";
  }
  return source || "-";
}

function formatEvaluationMethod(item) {
  const method = item.evaluation || item.source || "llm";
  if (method === "llm") {
    return "LLM";
  }
  if (method === "local") {
    return "本地程序";
  }
  if (method === "llm_fallback_local") {
    return "LLM 回退本地";
  }
  return method;
}

function setLoading(loading, message) {
  submitButton.disabled = loading;
  exportMdButton.disabled = loading;
  statusNode.textContent = message;
}
