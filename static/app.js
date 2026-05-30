const providerDefaults = {
  deepseek: "deepseek-chat",
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
  google_genai: "gemini-2.5-flash",
};

const state = {
  result: null,
  lastUpload: null,
  activeIssueFilter: "all",
  activeSectionId: null,
  activeIssueKey: null,
};

const form = document.getElementById("evaluate-form");
const providerInput = document.getElementById("provider");
const modelInput = document.getElementById("model");
const fileInput = document.getElementById("file");
const rubricFileInput = document.getElementById("rubric-file");
const formatFileInput = document.getElementById("format-file");
const submitButton = document.getElementById("submit-button");
const statusNode = document.getElementById("status");
const resultsNode = document.getElementById("results");
const exportMdButton = document.getElementById("export-md");
const refreshHistoryButton = document.getElementById("refresh-history");
const lastUploadListNode = document.getElementById("last-upload-list");
const reuseHintNode = document.getElementById("reuse-hint");

providerInput.addEventListener("change", () => {
  modelInput.value = providerDefaults[providerInput.value] || "";
});

exportMdButton.addEventListener("click", () => exportResult("md"));
refreshHistoryButton.addEventListener("click", () => {
  void loadHistory();
});

void loadHistory();
void loadLastUpload();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true, "正在分析论文，请稍候...");

  try {
    const response = await submitEvaluation();
    if (!response.ok) {
      throw new Error(response.detail || "请求失败");
    }
    state.result = response.data;
    state.activeIssueFilter = "all";
    state.activeIssueKey = null;
    state.activeSectionId =
      response.data.document.root_sections[0]?.identifier || null;
    renderResults();
    void loadHistory();
    setLoading(false, "分析完成");
  } catch (error) {
    setLoading(false, error.message || "处理失败");
  }
});

async function submitEvaluation() {
  const formData = new FormData();
  formData.append("provider", providerInput.value);
  formData.append("model", modelInput.value);
  formData.append("temperature", document.getElementById("temperature").value);
  formData.append("max_tokens", document.getElementById("max_tokens").value);

  const file = fileInput.files[0];
  if (file) {
    formData.append("file", file);
  }
  const rubricFile = rubricFileInput.files[0];
  if (rubricFile) {
    formData.append("rubric_file", rubricFile);
  }
  const formatFile = formatFileInput.files[0];
  if (formatFile) {
    formData.append("format_file", formatFile);
  }
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

async function loadLastUpload() {
  const response = await fetch("/last-upload");
  const payload = await response.json();
  state.lastUpload = payload.data?.items || null;
  renderLastUpload(state.lastUpload);
}

function renderResults() {
  const data = state.result;
  if (!data) {
    return;
  }

  resultsNode.classList.remove("hidden");
  document.getElementById("comment").textContent = data.comment || "暂无评语";
  document.getElementById("score-badge").textContent = `${data.score} 分`;
  document.getElementById("doc-title").textContent = data.document.title;
  document.getElementById("doc-type").textContent = data.document.source_type;
  document.getElementById("doc-sections").textContent = String(
    data.document.sections.length,
  );
  document.getElementById("topic-ratio").textContent = `${(
    data.topic_relevance_ratio * 100
  ).toFixed(1)}%`;

  const modelMeta = data.metadata?.model || {};
  const scoreSource = data.metadata?.score_source || "rule_engine";
  const commentSource = data.metadata?.comment_source || "fallback";
  document.getElementById("model-meta").innerHTML = [
    `<span class="meta-chip">模型 ${escapeHtml(modelMeta.provider || "-")} / ${escapeHtml(modelMeta.model || "-")}</span>`,
    `<span class="meta-chip ${modelMeta.available ? "is-ready" : "is-muted"}">Key ${modelMeta.available ? "可用" : "不可用"}</span>`,
    `<span class="meta-chip is-score">${formatScoreSource(scoreSource)}</span>`,
    `<span class="meta-chip ${commentSource === "llm" ? "is-llm" : "is-fallback"}">${formatCommentSource(commentSource)}</span>`,
  ].join("");

  renderStatistics(data.statistics);
  renderIssueFilters(data.issues);
  renderIssues(data.issues);
  renderTechList(data.technology_stack);
  renderSectionTree(data.document.root_sections);
  renderRubric(data.metadata?.rubric || null);
  renderFormatRequirements(data.metadata?.format_requirements || null);
  if (data.metadata?.last_upload) {
    state.lastUpload = data.metadata.last_upload;
    renderLastUpload(state.lastUpload);
  }
}

function renderStatistics(statistics) {
  renderTextList(
    "statistics-list",
    statistics.map((item) => `${item.label}: ${item.value}`),
  );
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
    button.className = `filter-chip${
      state.activeIssueFilter === key ? " active" : ""
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
    li.innerHTML =
      `<strong>[${issue.category}] ${issue.section_identifier} ${issue.section_title}</strong>` +
      `<div>${issue.message}</div>` +
      `<div class="issue-subline">建议：${issue.suggestion}</div>`;
    li.addEventListener("click", () => {
      state.activeIssueKey = key;
      state.activeSectionId = issue.section_identifier;
      renderIssues(issues);
      renderSectionTree(state.result.document.root_sections);
    });
    node.appendChild(li);
  });
}

function renderTechList(technologyStack) {
  renderTextList(
    "tech-list",
    technologyStack.length ? technologyStack : ["未识别到明确技术栈"],
  );
}

function renderSectionTree(rootSections) {
  const node = document.getElementById("section-tree");
  node.innerHTML = "";
  if (!rootSections.length) {
    node.textContent = "未识别到章节结构";
    return;
  }
  const list = document.createElement("ul");
  list.className = "tree-list";
  rootSections.forEach((section) => {
    list.appendChild(buildSectionNode(section));
  });
  node.appendChild(list);
}

function buildSectionNode(section) {
  const item = document.createElement("li");
  item.className = "tree-item";

  const button = document.createElement("button");
  button.type = "button";
  button.className = `tree-button${
    state.activeSectionId === section.identifier ? " active" : ""
  }`;
  button.innerHTML =
    `<div class="tree-header"><strong>${section.identifier} ${section.title}</strong>` +
    `<span>${(section.ratio * 100).toFixed(1)}% / ${section.word_count} 字</span></div>` +
    `<div class="tree-meta">层级 L${section.level} · 段落 ${section.paragraphs.length} · ` +
    `主题相关 ${(section.topic_relevance_score * 100).toFixed(1)}%</div>`;
  button.addEventListener("click", () => {
    state.activeSectionId = section.identifier;
    state.activeIssueKey = null;
    renderSectionTree(state.result.document.root_sections);
    renderIssues(state.result.issues);
  });
  item.appendChild(button);

  if (section.children && section.children.length) {
    const childList = document.createElement("ul");
    childList.className = "tree-list";
    section.children.forEach((child) => {
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
    li.innerHTML =
      `<strong>${item.title}</strong>` +
      `<div class="history-subline">${item.created_at} · ${item.source_type} · ${item.score} 分 · ` +
      `${Math.round((item.topic_relevance_ratio || 0) * 100)}% 主题相关</div>` +
      `<div class="history-subline">${item.comment}</div>`;
    node.appendChild(li);
  });
}

function renderLastUpload(lastUpload) {
  lastUploadListNode.innerHTML = "";
  const labels = {
    thesis: "论文",
    rubric: "评分标准",
    format_requirements: "格式要求",
  };
  const order = ["thesis", "rubric", "format_requirements"];
  let availableCount = 0;

  order.forEach((key) => {
    const entry = lastUpload?.[key] || null;
    const li = document.createElement("li");
    if (!entry || !entry.available) {
      li.textContent = `${labels[key]}: 暂无`;
      lastUploadListNode.appendChild(li);
      return;
    }
    availableCount += 1;
    li.textContent =
      `${labels[key]}: ${entry.filename} · ${entry.size} bytes · ${entry.updated_at}`;
    lastUploadListNode.appendChild(li);
  });

  reuseHintNode.textContent =
    availableCount > 0
      ? "未重新选择文件时，系统会自动复用上次上传文件。重新上传后会覆盖对应类型的上次记录。"
      : "当前还没有可复用的上传记录，请先至少上传一次论文文件。";
}

function renderRubric(rubric) {
  const metaNode = document.getElementById("rubric-meta");
  const listNode = document.getElementById("rubric-list");
  listNode.innerHTML = "";

  if (!rubric || !rubric.items || !rubric.items.length) {
    metaNode.textContent = "未上传评分标准";
    const li = document.createElement("li");
    li.textContent = "暂无评分标准";
    listNode.appendChild(li);
    return;
  }

  metaNode.textContent =
    `来源: ${rubric.source_name || "rubric.json"} | 总分: ${rubric.total_score}`;
  rubric.items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.criterion}: ${item.score}`;
    listNode.appendChild(li);
  });
}

function renderFormatRequirements(formatRequirements) {
  const metaNode = document.getElementById("format-meta");
  const listNode = document.getElementById("format-list");
  listNode.innerHTML = "";

  if (
    !formatRequirements ||
    !formatRequirements.items ||
    !formatRequirements.items.length
  ) {
    metaNode.textContent = "未上传格式要求";
    const li = document.createElement("li");
    li.textContent = "暂无格式要求";
    listNode.appendChild(li);
    return;
  }

  metaNode.textContent =
    `来源: ${formatRequirements.source_name || "format_requirements.json"} | ` +
    `条目数: ${formatRequirements.item_count || formatRequirements.items.length}`;
  formatRequirements.items.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.label}: ${item.value}`;
    listNode.appendChild(li);
  });
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
            `- [${issue.category}] ${issue.section_identifier} ${issue.section_title}: ${issue.message} 建议：${issue.suggestion}`,
        )
        .join("\n")
    : "- 未检测到明显问题";
  const tech = data.technology_stack.length
    ? data.technology_stack.map((item) => `- ${item}`).join("\n")
    : "- 未识别到明确技术栈";
  const rubric = buildRubricMarkdown(data.metadata?.rubric || null);
  const formatRequirements = buildFormatRequirementsMarkdown(
    data.metadata?.format_requirements || null,
  );

  return [
    `# ${data.document.title}`,
    "",
    `- 分数: ${data.score}`,
    `- 类型: ${data.document.source_type}`,
    `- 主题相关占比: ${(data.topic_relevance_ratio * 100).toFixed(1)}%`,
    `- 分数来源: ${formatScoreSource(data.metadata?.score_source || "rule_engine")}`,
    `- 评语来源: ${formatCommentSource(data.metadata?.comment_source || "fallback")}`,
    "",
    "## 总评语",
    "",
    data.comment,
    "",
    "## 统计结果",
    "",
    statistics,
    "",
    "## 识别问题",
    "",
    issues,
    "",
    "## 技术栈",
    "",
    tech,
    "",
    "## 评分标准",
    "",
    rubric,
    "",
    "## 格式要求",
    "",
    formatRequirements,
  ].join("\n");
}

function buildRubricMarkdown(rubric) {
  if (!rubric || !rubric.items || !rubric.items.length) {
    return "- 未上传评分标准";
  }
  return rubric.items
    .map((item) => `- ${item.criterion}: ${item.score}`)
    .concat([`- 总分: ${rubric.total_score}`])
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
  if (source === "rule_engine") {
    return "分数: 规则引擎";
  }
  return source || "-";
}

function formatCommentSource(source) {
  if (source === "llm") {
    return "评语: LLM";
  }
  if (source === "fallback") {
    return "评语: 规则回退";
  }
  return source || "-";
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setLoading(loading, message) {
  submitButton.disabled = loading;
  exportMdButton.disabled = loading;
  statusNode.textContent = message;
}
