const providerDefaults = {
  deepseek: "deepseek-chat",
  openai: "gpt-4o-mini",
  anthropic: "claude-3-5-haiku-latest",
  google_genai: "gemini-2.5-flash",
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

providerInput.addEventListener("change", () => {
  modelInput.value = providerDefaults[providerInput.value] || "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setLoading(true, "正在分析论文，请稍候...");

  try {
    const response = await submitEvaluation();
    if (!response.ok) {
      throw new Error(response.detail || "请求失败");
    }
    renderResults(response.data);
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

function renderResults(data) {
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

  renderList(
    "statistics-list",
    data.statistics.map((item) => `${item.label}: ${item.value}`),
  );
  renderList(
    "issues-list",
    data.issues.length
      ? data.issues.map(
        (issue) =>
          `[${issue.category}] ${issue.message} 建议: ${issue.suggestion}`,
      )
      : ["未检测到明显问题"],
  );
  renderList(
    "tech-list",
    data.technology_stack.length ? data.technology_stack : ["未识别到明确技术栈"],
  );
}

function renderList(id, items) {
  const node = document.getElementById(id);
  node.innerHTML = "";
  for (const item of items) {
    const li = document.createElement("li");
    li.textContent = item;
    node.appendChild(li);
  }
}

function setLoading(loading, message) {
  submitButton.disabled = loading;
  statusNode.textContent = message;
}
