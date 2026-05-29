const providerDefaults = {
  deepseek: "deepseek-chat",
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
  google_genai: "gemini-2.5-flash",
};

const state = {
  result: null,
  activeIssueFilter: "all",
  activeSectionId: null,
  activeIssueKey: null,
};

const form = document.getElementById("evaluate-form");
const providerInput = document.getElementById("provider");
const modelInput = document.getElementById("model");
const textInput = document.getElementById("text");
const fileInput = document.getElementById("file");
const filenameInput = document.getElementById("filename");
const submitButton = document.getElementById("submit-button");
const statusNode = document.getElementById("status");
const resultsNode = document.getElementById("results");
const exportJsonButton = document.getElementById("export-json");
const exportMdButton = document.getElementById("export-md");
const refreshHistoryButton = document.getElementById("refresh-history");

providerInput.addEventListener("change", () => {
  modelInput.value = providerDefaults[providerInput.value] || "";
});

exportJsonButton.addEventListener("click", () => exportResult("json"));
exportMdButton.addEventListener("click", () => exportResult("md"));
refreshHistoryButton.addEventListener("click", () => {
  void loadHistory();
});

void loadHistory();

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
    state.activeSectionId = response.data.document.root_sections[0]?.identifier || null;
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
  const text = textInput.value.trim();
  if (file) {
    formData.append("file", file);
    return postForm("/evaluate/upload", formData);
  }
  if (!text) {
    throw new Error("请先粘贴论文文本或上传文件");
  }
  formData.append("text", text);
  formData.append("filename", filenameInput.value || "submission.md");
  return postForm("/evaluate/text", formData);
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
  document.getElementById("model-meta").textContent =
    `模型: ${modelMeta.provider || "-"} / ${modelMeta.model || "-"} | ` +
    `Key可用: ${modelMeta.available ? "是" : "否"}`;

  renderStatistics(data.statistics);
  renderIssueFilters(data.issues);
  renderIssues(data.issues);
  renderTechList(data.technology_stack);
  renderSectionTree(data.document.root_sections);
  renderFocusPanel();
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
      renderFocusPanel();
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
    renderFocusPanel();
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

function renderFocusPanel() {
  const node = document.getElementById("focus-panel");
  node.innerHTML = "";
  if (!state.result) {
    return;
  }

  const activeIssue = findActiveIssue();
  if (activeIssue) {
    renderIssueFocus(node, activeIssue);
    return;
  }

  const activeSection = findSectionByIdentifier(
    state.result.document.sections,
    state.activeSectionId,
  );
  if (activeSection) {
    renderSectionFocus(node, activeSection);
    return;
  }

  node.innerHTML =
    '<p class="focus-empty">点击章节树或问题项后，这里会显示对应的段落和上下文。</p>';
}

function renderIssueFocus(node, issue) {
  const block = document.createElement("div");
  block.className = "focus-block";
  block.innerHTML =
    `<div class="focus-kicker">问题定位</div>` +
    `<h3>${issue.section_identifier} ${issue.section_title}</h3>` +
    `<p class="focus-summary">${issue.message}</p>` +
    `<p class="focus-meta">规则: ${issue.rule_id} · 严重程度: ${issue.severity}</p>` +
    `<div class="excerpt-card">${highlightText(issue.excerpt, issue.matched_text)}</div>` +
    `<p class="focus-meta">建议：${issue.suggestion}</p>`;
  node.appendChild(block);
}

function renderSectionFocus(node, section) {
  const block = document.createElement("div");
  block.className = "focus-block";
  const previewParagraphs = section.paragraphs.slice(0, 3);
  block.innerHTML =
    `<div class="focus-kicker">章节详情</div>` +
    `<h3>${section.identifier} ${section.title}</h3>` +
    `<p class="focus-summary">章节占比 ${(section.ratio * 100).toFixed(1)}%，` +
    `主题相关 ${(section.topic_relevance_score * 100).toFixed(1)}%，` +
    `共 ${section.word_count} 字。</p>`;
  previewParagraphs.forEach((paragraph) => {
    const p = document.createElement("p");
    p.className = "focus-paragraph";
    p.textContent = paragraph.text;
    block.appendChild(p);
  });
  if (!previewParagraphs.length) {
    const empty = document.createElement("p");
    empty.className = "focus-empty";
    empty.textContent = "该章节暂无可展示段落。";
    block.appendChild(empty);
  }
  node.appendChild(block);
}

function buildIssueKey(issue) {
  return [
    issue.section_identifier,
    issue.paragraph_index,
    issue.sentence_index,
    issue.rule_id,
    issue.matched_text,
  ].join(":");
}

function findActiveIssue() {
  if (!state.result || !state.activeIssueKey) {
    return null;
  }
  return (
    state.result.issues.find((issue) => buildIssueKey(issue) === state.activeIssueKey) ||
    null
  );
}

function findSectionByIdentifier(sections, identifier) {
  return sections.find((section) => section.identifier === identifier) || null;
}

function highlightText(text, matchedText) {
  if (!matchedText || !text.includes(matchedText)) {
    return escapeHtml(text);
  }
  return escapeHtml(text).replaceAll(
    escapeHtml(matchedText),
    `<mark>${escapeHtml(matchedText)}</mark>`,
  );
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

function exportResult(format) {
  if (!state.result) {
    setLoading(false, "暂无可导出的结果");
    return;
  }
  if (format === "json") {
    downloadFile(
      "thesisev-result.json",
      "application/json;charset=utf-8",
      JSON.stringify(state.result, null, 2),
    );
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

  return [
    `# ${data.document.title}`,
    "",
    `- 分数: ${data.score}`,
    `- 类型: ${data.document.source_type}`,
    `- 主题相关占比: ${(data.topic_relevance_ratio * 100).toFixed(1)}%`,
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
  ].join("\n");
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

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function setLoading(loading, message) {
  submitButton.disabled = loading;
  exportJsonButton.disabled = loading;
  exportMdButton.disabled = loading;
  statusNode.textContent = message;
}
