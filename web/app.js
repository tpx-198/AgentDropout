"use strict";

const $ = (selector) => document.querySelector(selector);
const names = { workspace: "实验工作台", models: "智能体与模型", experiments: "训练与评测", history: "运行记录" };
const topologyNames = { full: "全连接", random: "随机", layered: "分层", chain: "链式", star: "星形", debate: "辩论式" };
const datasetNames = { gsm8k: "GSM8K", aqua: "AQuA", svamp: "SVAMP", multiarith: "MultiArith" };
const roleNames = { "Math Solver": "数学求解", "Mathematical Analyst": "数学分析", "Programming Expert": "编程专家", Inspector: "检查员", FinalRefer: "最终汇总" };
const statusNames = { queued: "准备中", running: "执行中", cancelling: "正在取消", completed: "已完成", cancelled: "已取消", failed: "失败", interrupted: "已中断", success: "完成" };
const state = { config: null, token: "", roles: [], datasets: [], keyPresent: {}, pendingKeys: {}, clearKeys: new Set(),
  jobs: [], checkpoints: [], kind: "train", page: "workspace", selectedJob: null, job: null, preview: null,
  dirty: false, polling: false, previewTimer: null };

function jobTitle(job) {
  const time = new Date(job.created_at).toLocaleString("zh-CN", {
    timeZone: "Asia/Shanghai", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false
  });
  return `${job.kind === "train" ? "拓扑训练" : "测试集评测"} · ${time}`;
}

function escape(value) {
  return String(value ?? "").replace(/[&<>"']/g, (character) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]));
}

function toast(message, error = false) {
  const element = $("#toast");
  element.textContent = message;
  element.classList.toggle("error", error);
  element.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { element.hidden = true; }, error ? 6500 : 3500);
}

async function api(path, body) {
  const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-ADL-Token": state.token }, body: JSON.stringify(body) };
  const response = await fetch(path, options);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "操作失败，请稍后重试");
  return result;
}

function dirty() {
  state.dirty = true;
  $("#dirty-status").textContent = "有未保存的修改";
  updateNumbers();
}

function go(page) {
  state.page = page;
  document.querySelectorAll(".page").forEach((element) => { element.hidden = element.id !== `page-${page}`; });
  document.querySelectorAll("[data-page]").forEach((element) => element.classList.toggle("active", element.dataset.page === page));
  $("#breadcrumb").textContent = names[page];
  if (page === "models") renderModels();
  if (page === "history") renderHistory();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setKind(kind) {
  state.kind = kind;
  document.querySelectorAll("[data-kind]").forEach((element) => {
    const active = element.dataset.kind === kind;
    element.classList.toggle("active", active);
    element.setAttribute("aria-selected", String(active));
  });
  $("#training-fields").hidden = kind !== "train";
  $("#evaluation-fields").hidden = kind !== "evaluate";
  $("#experiment-form-title").textContent = kind === "train" ? "训练配置" : "评测配置";
  $("#split-label").textContent = kind === "train" ? "TRAIN SPLIT" : "TEST SPLIT";
  updateNumbers();
}

function newModel() {
  return { backend: "mock", model: "local-fixed-response", base_url: null, temperature: null, max_tokens: null,
    timeout: 60, mock_response: "[MOCK fixed response]\nThe answer is 5" };
}

function resizeTeam(count) {
  const models = {};
  const roles = [];
  for (let index = 0; index < count - 1; index++) {
    const node = `agent_${index + 1}`;
    models[node] = state.config.models[node] || newModel();
    roles.push(state.config.roles[index] || state.roles[index % state.roles.length]);
  }
  models.final = state.config.models.final || newModel();
  state.config.models = models;
  state.config.roles = roles;
  state.config.topology.agents = count;
  state.config.evaluation.checkpoint = "";
  syncForms();
  dirty();
  schedulePreview();
}

const bindings = {
  rounds: ["topology", "rounds"], topology: ["topology", "topology"], temporal: ["topology", "temporal"],
  "initial-weight": ["topology", "initial_weight"], seed: ["topology", "seed"], density: ["topology", "density"],
  layers: ["topology", "layers"], "train-limit": ["training", "limit"], "train-steps": ["training", "steps"],
  "batch-size": ["training", "batch_size"], "learning-rate": ["training", "lr"], sparsity: ["training", "sparsity"],
  "prune-rate": ["training", "prune_rate"], "eval-limit": ["evaluation", "limit"],
  "eval-mode": ["evaluation", "mode"], "eval-threshold": ["evaluation", "threshold"]
};

function syncForms() {
  for (const [id, [section, field]] of Object.entries(bindings)) $("#" + id).value = state.config[section][field];
  $("#agents").value = state.config.topology.agents;
  $("#sampling-temperature").value = state.config.sampling_temperature;
  $("#compare").checked = state.config.evaluation.compare;
  const roundSelect = $("#preview-round");
  const previousRound = Number(roundSelect.value || 0);
  roundSelect.innerHTML = Array.from({ length: state.config.topology.rounds }, (_, index) => `<option value="${index}">第 ${index + 1} 轮</option>`).join("");
  roundSelect.value = Math.min(previousRound, state.config.topology.rounds - 1);
  renderDatasetChoices();
  renderCheckpoints();
  renderModels();
  updateNumbers();
}

function renderDatasetChoices() {
  for (const [prefix, section, split] of [["train", "training", "train"], ["eval", "evaluation", "test"]]) {
    $("#" + prefix + "-datasets").innerHTML = state.datasets.map((item) => `<label class="dataset-choice"><input type="checkbox" data-dataset="${item.name}" data-section="${section}" ${state.config[section].datasets.includes(item.name) ? "checked" : ""}>${datasetNames[item.name]} <small> · ${item.splits[split] ?? 0}</small></label>`).join("");
  }
}

function renderDatasetTable() {
  $("#dataset-table").innerHTML = state.datasets.map((item) => `<tr><td class="dataset-name">${datasetNames[item.name]}</td><td>${(item.splits.train ?? 0).toLocaleString()}</td><td>${(item.splits.test ?? 0).toLocaleString()}</td><td class="dataset-type">${item.name === "aqua" ? "多项选择" : "数值推理"}</td><td><span class="badge teal">● 已就绪</span></td></tr>`).join("");
}

function updateNumbers() {
  if (!state.config) return;
  const config = state.config;
  const workers = config.topology.agents - 1;
  const rounds = config.topology.rounds;
  $("#stat-agents").textContent = config.topology.agents;
  $("#stat-rounds").textContent = rounds;
  $("#stat-calls").textContent = workers * rounds + 1;
  $("#stat-datasets").textContent = state.datasets.length;
  $("#stat-call-formula").textContent = `${workers} 个讨论节点 × ${rounds} 轮 + 1`;
  $("#nav-agent-count").textContent = config.topology.agents;
  $("#weight-value").textContent = Number(config.topology.initial_weight).toFixed(2);
  $("#graph-label").textContent = `${topologyNames[config.topology.topology]}候选图`;
  $("#density").disabled = config.topology.topology !== "random";
  $("#layers").disabled = config.topology.topology !== "layered";
  $("#eval-threshold").disabled = config.evaluation.mode !== "threshold";
  const groups = config.evaluation.checkpoint && config.evaluation.compare ? 2 : 1;
  const examples = state.kind === "train" ? config.training.steps * config.training.batch_size : config.evaluation.limit * config.evaluation.datasets.length * groups;
  const total = examples * (workers * rounds + 1);
  const realPerQuestion = Object.entries(config.models).reduce((sum, [node, spec]) => sum + (spec.backend === "mock" ? 0 : node === "final" ? 1 : rounds), 0);
  $("#budget-total").innerHTML = `${Number.isFinite(total) ? total.toLocaleString() : "—"}<small>次</small>`;
  $("#budget-team").textContent = `${config.topology.agents} / ${rounds}`;
  $("#budget-examples").textContent = examples;
  $("#budget-real").textContent = (examples * realPerQuestion).toLocaleString();
  $("#budget-backend").textContent = realPerQuestion ? "API ENABLED" : "MOCK";
  $("#run-environment").textContent = realPerQuestion ? "独立模型 API" : "本地模拟";
  $("#real-consent").hidden = realPerQuestion === 0;
  $("#budget-note").textContent = realPerQuestion ? "真实请求使用各节点已保存的 API 配置。取消任务将在当前请求结束后生效。" : "模拟回答用于验证流程，结果标记为模拟诊断。";
  const active = state.jobs.some((job) => ["queued", "running", "cancelling"].includes(job.status));
  $("#start-job").disabled = active;
  $("#start-job").textContent = active ? "任务执行中…" : state.kind === "train" ? "开始拓扑训练 ↗" : "开始测试集评测 ↗";
}

function renderModels(highlight) {
  if (!state.config) return;
  $("#model-cards").innerHTML = Object.entries(state.config.models).map(([node, spec], index) => {
    const final = node === "final";
    const role = final ? "FinalRefer" : state.config.roles[index];
    const present = !state.clearKeys.has(node) && state.keyPresent[node];
    const real = spec.backend !== "mock";
    return `<article class="card model-card ${highlight === node ? "highlight" : ""}" id="model-${node}"><div class="card-header"><div class="node-heading"><span class="node-chip ${final ? "final" : ""}">${final ? "Σ" : String(index + 1).padStart(2, "0")}</span><div><h2>${escape(roleNames[role])}</h2><small>${node} · ${escape(role)}</small></div></div><span class="badge ${real ? "purple" : "neutral"}">${real ? "API" : "MOCK"}</span></div><div class="card-body">
      <div class="two-columns"><label>角色<select data-node="${node}" data-model-field="role" ${final ? "disabled" : ""}>${(final ? ["FinalRefer"] : state.roles).map((item) => `<option value="${escape(item)}" ${role === item ? "selected" : ""}>${escape(roleNames[item])}</option>`).join("")}</select></label><label>模型接口<select data-node="${node}" data-model-field="backend">${[["mock", "本地模拟"], ["deepseek", "DeepSeek"], ["openai", "OpenAI 兼容接口"]].map(([value, label]) => `<option value="${value}" ${spec.backend === value ? "selected" : ""}>${label}</option>`).join("")}</select></label></div>
      <label>模型名称<input data-node="${node}" data-model-field="model" value="${escape(spec.model)}" placeholder="填写服务商支持的模型名称" autocomplete="off"></label>
      <div ${real ? "" : "hidden"}><label>API 根地址<input data-node="${node}" data-model-field="base_url" value="${escape(spec.base_url || "")}" placeholder="https://api.example.com/v1" autocomplete="off"></label><div class="key-label-row"><span>API 密钥 <span class="key-status ${present ? "" : "missing"}">${present ? "已保存" : "待填写"}</span></span><button class="clear-key" data-clear-key="${node}">清除已保存密钥</button></div><input type="password" data-node="${node}" data-model-field="api_key" value="${escape(state.pendingKeys[node] || "")}" aria-label="${node} API 密钥" placeholder="${present ? "留空沿用已保存密钥" : "输入 API 密钥，仅保存在本机"}" autocomplete="new-password"></div>
      <label ${real ? "hidden" : ""}>固定模拟回答<textarea data-node="${node}" data-model-field="mock_response">${escape(spec.mock_response || "")}</textarea></label>
      <details><summary>生成参数与请求超时</summary><div class="three-columns"><label>Temperature<input type="number" min="0" max="2" step="0.1" data-node="${node}" data-model-field="temperature" value="${spec.temperature ?? ""}" placeholder="提供方默认"></label><label>最大输出 Token<input type="number" min="1" data-node="${node}" data-model-field="max_tokens" value="${spec.max_tokens ?? ""}" placeholder="提供方默认"></label><label>超时 / 秒<input type="number" min="1" data-node="${node}" data-model-field="timeout" value="${spec.timeout}"></label></div></details>
      </div></article>`;
  }).join("");
}

function renderCheckpoints() {
  const select = $("#checkpoint");
  const signature = JSON.stringify(state.checkpoints);
  if (select.dataset.options !== signature) {
    select.innerHTML = '<option value="">当前初始拓扑</option>' + state.checkpoints.map((item) => {
      const job = state.jobs.find((value) => item.id === `job:${value.id}`);
      const label = job ? `${jobTitle(job)} · ${job.progress} 步` : item.label;
      return `<option value="${escape(item.id)}">${escape(label)}</option>`;
    }).join("");
    select.dataset.options = signature;
  }
  select.value = state.config.evaluation.checkpoint;
}

function schedulePreview() {
  clearTimeout(state.previewTimer);
  state.previewTimer = setTimeout(refreshPreview, 400);
}

async function refreshPreview() {
  try {
    const signature = JSON.stringify(state.config);
    const preview = await api("/api/preview", { config: state.config });
    if (signature !== JSON.stringify(state.config)) return;
    state.preview = preview;
    renderGraph();
  } catch (error) {
    $("#graph-note").textContent = error.message;
  }
}

function renderGraph() {
  if (!state.preview) return;
  const count = state.config.topology.agents - 1;
  const round = Math.min(Number($("#preview-round").value || 0), state.preview.execution.rounds.length - 1);
  const graph = state.preview.execution.rounds[round];
  const positions = Array.from({ length: count }, (_, index) => {
    const angle = count === 1 ? 0 : -Math.PI / 2 + index * Math.PI * 2 / count;
    return { x: 255 + (count === 1 ? 0 : 148 * Math.cos(angle)), y: 182 + (count === 1 ? 0 : 108 * Math.sin(angle)) };
  });
  const final = { x: 585, y: 182 };
  function edge(source, target, weight, summary) {
    const dx = target.x - source.x;
    const dy = target.y - source.y;
    const distance = Math.hypot(dx, dy);
    const from = { x: source.x + dx / distance * 31, y: source.y + dy / distance * 31 };
    const to = { x: target.x - dx / distance * 36, y: target.y - dy / distance * 36 };
    const midX = (from.x + to.x) / 2;
    const midY = (from.y + to.y) / 2;
    return `<line x1="${from.x}" y1="${from.y}" x2="${to.x}" y2="${to.y}" stroke="${summary ? "#9daeb1" : "#248b76"}" stroke-width="1.25" opacity="${summary ? .5 : .2 + weight * .65}" ${summary ? 'stroke-dasharray="4 5"' : ""} marker-end="url(#${summary ? "arrow-summary" : "arrow"})"/>${!summary && count <= 6 ? `<rect x="${midX - 13}" y="${midY - 7}" width="26" height="13" rx="3" fill="white"/><text x="${midX}" y="${midY + 3}" text-anchor="middle" fill="#73a491" font-size="9">${weight.toFixed(2)}</text>` : ""}`;
  }
  let markup = '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#248b76"/></marker><marker id="arrow-summary" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="#9daeb1"/></marker></defs>';
  for (const [source, target] of graph.spatial) markup += edge(positions[source], positions[target], state.preview.weights.spatial[round][source][target], false);
  for (const index of state.preview.execution.summary) markup += edge(positions[index], final, state.preview.weights.summary[index], true);
  positions.forEach((point, index) => {
    const role = state.config.roles[index];
    markup += `<g class="graph-node" data-graph-node="agent_${index + 1}" tabindex="0" role="button" aria-label="配置 agent_${index + 1}"><circle cx="${point.x}" cy="${point.y}" r="29" fill="#f0f8f4" stroke="#c8dfd4"/><text x="${point.x}" y="${point.y + 5}" text-anchor="middle" fill="#33876b" font-size="13" font-weight="600">A${index + 1}</text><rect x="${point.x - 43}" y="${point.y + 34}" width="86" height="34" rx="4" fill="white" opacity=".9"/><text x="${point.x}" y="${point.y + 46}" text-anchor="middle" fill="#527267" font-size="10">${escape(roleNames[role])}</text><text x="${point.x}" y="${point.y + 59}" text-anchor="middle" fill="#a2b0b4" font-size="8">agent_${index + 1}</text></g>`;
  });
  markup += `<g class="graph-node" data-graph-node="final" tabindex="0" role="button" aria-label="配置最终汇总者"><rect x="${final.x - 36}" y="${final.y - 32}" width="72" height="64" rx="16" fill="#f0edf8" stroke="#dcd5eb"/><text x="${final.x}" y="${final.y + 7}" text-anchor="middle" font-size="24" fill="#8d76b4">Σ</text><text x="${final.x}" y="${final.y + 52}" text-anchor="middle" font-size="11" fill="#73638b">最终汇总</text><text x="${final.x}" y="${final.y + 66}" text-anchor="middle" font-size="8" fill="#a5a0b4">FinalRefer</text></g>`;
  $("#topology-svg").innerHTML = markup;
  $("#graph-edge-count").textContent = `${graph.spatial.length} 条轮内边 · ${graph.temporal.length} 条跨轮边`;
  $("#graph-note").textContent = state.config.evaluation.checkpoint ? "当前展示所选检查点保留的通信边，边上数字为已学习的权重。" : "预览展示保留候选边对应的无环执行图。实际运行按权重采样，跨轮边读取上一轮回答。";
}

async function saveSettings(silent = false) {
  const result = await api("/api/config", { config: state.config, keys: state.pendingKeys, clear_keys: [...state.clearKeys] });
  state.config = result.config;
  state.keyPresent = result.key_present;
  state.pendingKeys = {};
  state.clearKeys.clear();
  state.dirty = false;
  $("#dirty-status").textContent = "配置已同步";
  renderModels();
  if (!silent) toast("配置已保存在本机");
}

async function startJob() {
  try {
    $("#start-job").disabled = true;
    await saveSettings(true);
    const job = await api("/api/jobs", { kind: state.kind, config: state.config, allow_real_api: $("#allow-real-api").checked });
    state.selectedJob = job.id;
    state.job = job;
    $("#allow-real-api").checked = false;
    renderJob();
    toast("实验已启动");
    await poll();
  } catch (error) {
    toast(error.message, true);
  } finally {
    updateNumbers();
  }
}

function statusBadge(status) {
  const color = ["completed", "success"].includes(status) ? "teal" : ["failed", "interrupted"].includes(status) ? "red" : ["running", "queued", "cancelling"].includes(status) ? "purple" : "neutral";
  return `<span class="badge ${color}">${statusNames[status] || escape(status)}</span>`;
}

function formatRatio(value) {
  return value == null ? "—" : `${(value * 100).toFixed(1)}%`;
}

function metricCards(summary, labelPrefix = "") {
  const diagnostic = summary.metrics_kind === "mock_or_mixed_diagnostic";
  return [[labelPrefix + (diagnostic ? "模拟诊断命中率" : "正确率"), formatRatio(diagnostic ? summary.diagnostic_ratio : summary.accuracy)],
    ["模型调用", (summary.model_calls || 0).toLocaleString()],
    ["总 Token", summary.usage ? summary.usage.total_tokens.toLocaleString() : "未测得"],
    ["累计答题耗时", `${(summary.elapsed_seconds || 0).toFixed(2)} s`]]
    .map(([label, value]) => `<div class="result-stat"><span>${label}</span><strong>${value}</strong></div>`).join("");
}

function renderJob() {
  const job = state.job;
  $("#job-panel").hidden = !job;
  $("#no-job").hidden = Boolean(job);
  if (!job) return;
  $("#job-name").textContent = jobTitle(job);
  $("#job-status").outerHTML = statusBadge(job.status).replace('<span class=', '<span id="job-status" class=');
  $("#job-message").textContent = job.message;
  $("#job-progress-label").textContent = `${job.progress} / ${job.total}`;
  $("#job-progress").max = job.total || 1;
  $("#job-progress").value = job.progress;
  $("#cancel-job").hidden = !["queued", "running", "cancelling"].includes(job.status);
  $("#cancel-job").disabled = job.status === "cancelling";
  $("#download-result").href = `/api/jobs/${job.id}/export`;
  $("#download-checkpoint").hidden = job.kind !== "train" || !job.completed_steps;
  $("#download-checkpoint").href = `/api/jobs/${job.id}/checkpoint`;
  const groups = job.kind === "train" ? { training: job.summary } : job.summary;
  const summary = { ...(groups.trained || groups.current || groups.training || groups.baseline || {}) };
  if (Object.keys(groups).length > 1) {
    const values = Object.values(groups);
    summary.model_calls = values.reduce((total, value) => total + value.model_calls, 0);
    summary.elapsed_seconds = values.reduce((total, value) => total + value.elapsed_seconds, 0);
    summary.usage = values.every((value) => value.usage) ? {
      total_tokens: values.reduce((total, value) => total + value.usage.total_tokens, 0)
    } : null;
  }
  $("#result-stats").innerHTML = metricCards(summary, groups.trained ? "训练后 · " : "");
  const groupNames = { training: "训练", current: "当前拓扑", baseline: "初始拓扑", trained: "训练后拓扑" };
  $("#comparison-table").innerHTML = Object.keys(groups).length > 1 ? `<div class="comparison table-wrap"><table><thead><tr><th>对照组</th><th>正确率 / 诊断比例</th><th>已执行题数</th><th>模型调用</th><th>真实调用</th><th>总 Token</th></tr></thead><tbody>${Object.entries(groups).map(([name, value]) => `<tr><td>${groupNames[name]}</td><td>${formatRatio(value.accuracy ?? value.diagnostic_ratio)}</td><td>${value.attempted}</td><td>${value.model_calls}</td><td>${value.real_api_calls}</td><td>${value.usage ? value.usage.total_tokens : "未测得"}</td></tr>`).join("")}</tbody></table></div>` : "";
  $("#result-rows").innerHTML = job.rows.slice(-100).map((row, index) => `<tr data-result-index="${Math.max(0, job.rows.length - 100) + index}"><td><span>${escape(row.sample_id)}</span><small>${datasetNames[row.dataset]}</small></td><td>${groupNames[row.group] || row.group}</td><td>${escape(row.target)}</td><td>${escape(row.prediction ?? "—")}</td><td>${row.status === "success" ? `<span class="badge ${row.correct ? "teal" : "neutral"}">${row.correct ? "命中" : "未命中"}</span>` : statusBadge(row.status)}</td><td>${row.elapsed_seconds.toFixed(2)} s</td></tr>`).join("");
  $("#rows-note").textContent = job.rows.length > 100 ? "页面展示最近 100 条记录，完整结果可导出。点击样本查看详情。" : "点击样本行查看完整回答与逐节点调用记录。";
  $("#loss-area").hidden = !job.losses.length;
  if (job.losses.length) {
    const values = job.losses.map((item) => item.loss);
    const low = Math.min(...values);
    const high = Math.max(...values);
    const span = high - low || 1;
    const points = values.map((value, index) => `${50 + index * 800 / Math.max(1, values.length - 1)},${110 - (value - low) / span * 80}`).join(" ");
    $("#loss-chart").innerHTML = `<line x1="50" y1="120" x2="850" y2="120" stroke="#e5eeea"/><polyline points="${points}" fill="none" stroke="#439e7d" stroke-width="2"/>${values.length === 1 ? `<circle cx="50" cy="110" r="4" fill="#439e7d"/>` : ""}<text x="50" y="145" fill="#98a8a0" font-size="10">STEP 1</text><text x="850" y="145" fill="#98a8a0" font-size="10" text-anchor="end">STEP ${values.length}</text><text x="50" y="16" fill="#6a9280" font-size="10">最新损失 ${values.at(-1).toFixed(5)}</text>`;
  }
}

function renderHistory() {
  $("#nav-job-count").textContent = state.jobs.length;
  if (!state.jobs.length) {
    $("#history-list").innerHTML = '<div class="empty-state"><span>◷</span><h3>还没有实验记录</h3><p>启动一次训练或评测，记录会自动保存在这里。</p></div>';
    return;
  }
  $("#history-list").innerHTML = state.jobs.map((job) => `<article class="history-card"><span class="action-icon ${job.kind === "evaluate" ? "purple" : ""}">${job.kind === "train" ? "↗" : "◎"}</span><div><h3>${escape(jobTitle(job))}</h3><p>${new Date(job.created_at).toLocaleString("zh-CN", { timeZone: "Asia/Shanghai" })} · 进度 ${job.progress}/${job.total}</p></div>${statusBadge(job.status)}<button class="button small" data-open-job="${job.id}">查看结果 →</button></article>`).join("");
}

async function selectJob(identity) {
  state.selectedJob = identity;
  state.job = await api(`/api/jobs/${identity}`);
  renderJob();
  go("experiments");
  setTimeout(() => $("#job-panel").scrollIntoView({ behavior: "smooth", block: "start" }), 100);
}

async function showRow(index) {
  try {
    const result = await api(`/api/jobs/${state.selectedJob}/export`);
    const row = result.details.rows?.[index];
    if (!row) throw new Error("该记录仍在写入，请稍后查看");
    $("#detail-title").textContent = row.sample_id;
    $("#detail-content").innerHTML = `<h3>题目</h3><pre>${escape(row.result.task)}</pre><h3>最终回答 · 标准答案 ${escape(row.target)}</h3><pre>${escape(row.result.answer || "任务尚未生成最终回答")}</pre><h3>通信与模型调用</h3>${row.result.trace.map((call) => `<details><summary>第 ${call.round} 轮 · ${escape(call.node_id)} · ${escape(call.role)}${call.error_type ? ` · ${escape(call.error_type)}` : ""}</summary><p>轮内可见 ${escape(Object.keys(call.spatial).join(", ") || "无")} · 跨轮可见 ${escape(Object.keys(call.temporal).join(", ") || "无")}</p><pre>${escape(call.output ?? "调用失败")}</pre><details><summary>查看发送的完整消息</summary><pre>${escape(call.messages.map((message) => `${message.role.toUpperCase()}\n${message.content}`).join("\n\n"))}</pre></details></details>`).join("")}`;
    $("#detail-dialog").showModal();
  } catch (error) { toast(error.message, true); }
}

async function poll() {
  if (state.polling) return;
  state.polling = true;
  try {
    const result = await api("/api/jobs");
    state.jobs = result.jobs;
    state.checkpoints = result.checkpoints;
    renderCheckpoints();
    renderHistory();
    if (state.selectedJob) {
      const job = await api(`/api/jobs/${state.selectedJob}`);
      if (JSON.stringify(job) !== JSON.stringify(state.job)) {
        state.job = job;
        renderJob();
      }
    }
    $("#connection-status").textContent = "本地引擎已连接";
    updateNumbers();
  } catch (_) {
    $("#connection-status").textContent = "连接中断，请重新启动平台";
  } finally { state.polling = false; }
}

function bindEvents() {
  document.querySelectorAll("[data-page]").forEach((element) => element.addEventListener("click", () => go(element.dataset.page)));
  document.querySelectorAll("[data-kind]").forEach((element) => element.addEventListener("click", () => setKind(element.dataset.kind)));
  document.querySelectorAll("[data-action]").forEach((element) => element.addEventListener("click", () => { setKind(element.dataset.action); go("experiments"); }));
  $("#go-experiments").addEventListener("click", () => go("experiments"));
  $("#agents").addEventListener("input", (event) => {
    const count = Number(event.target.value);
    if (Number.isInteger(count) && count >= 2 && count <= 32) resizeTeam(count);
  });
  $("#agents").addEventListener("change", (event) => resizeTeam(Math.min(32, Math.max(2, Number(event.target.value) || 2))));
  for (const [id, [section, field]] of Object.entries(bindings)) {
    $("#" + id).addEventListener("input", (event) => {
      state.config[section][field] = event.target.tagName === "SELECT" ? event.target.value : Number(event.target.value);
      if (section === "topology") {
        state.config.evaluation.checkpoint = "";
        if (field === "rounds" && Number.isInteger(state.config.topology.rounds) && state.config.topology.rounds > 0 && state.config.topology.rounds <= 10) syncForms();
        schedulePreview();
      }
      dirty();
    });
  }
  $("#sampling-temperature").addEventListener("input", (event) => { state.config.sampling_temperature = Number(event.target.value); dirty(); });
  $("#compare").addEventListener("change", (event) => { state.config.evaluation.compare = event.target.checked; dirty(); });
  $("#checkpoint").addEventListener("change", (event) => { state.config.evaluation.checkpoint = event.target.value; dirty(); schedulePreview(); });
  $("#preview-round").addEventListener("change", renderGraph);
  $("#refresh-preview").addEventListener("click", refreshPreview);
  $("#topology-svg").addEventListener("click", (event) => {
    const node = event.target.closest("[data-graph-node]");
    if (!node) return;
    go("models"); renderModels(node.dataset.graphNode);
    $("#model-" + node.dataset.graphNode).scrollIntoView({ behavior: "smooth", block: "center" });
  });
  $("#topology-svg").addEventListener("keydown", (event) => {
    if ((event.key === "Enter" || event.key === " ") && event.target.closest("[data-graph-node]")) {
      event.preventDefault();
      event.target.closest("[data-graph-node]").dispatchEvent(new MouseEvent("click", { bubbles: true }));
    }
  });
  document.addEventListener("change", (event) => {
    const input = event.target.closest("[data-dataset]");
    if (!input) return;
    const section = state.config[input.dataset.section];
    section.datasets = state.datasets.filter((item) => $( `[data-section="${input.dataset.section}"][data-dataset="${item.name}"]`).checked).map((item) => item.name);
    dirty();
  });
  $("#model-cards").addEventListener("input", (event) => {
    const input = event.target.closest("[data-model-field]");
    if (!input) return;
    const node = input.dataset.node;
    const field = input.dataset.modelField;
    const spec = state.config.models[node];
    if (field === "api_key") { state.pendingKeys[node] = input.value; state.clearKeys.delete(node); }
    else if (field === "role") state.config.roles[Number(node.split("_")[1]) - 1] = input.value;
    else if (["temperature", "max_tokens", "timeout"].includes(field)) spec[field] = input.value === "" ? null : Number(input.value);
    else if (field === "base_url") spec[field] = input.value.trim() || null;
    else spec[field] = input.value;
    if (field === "backend") {
      if (spec.backend === "mock") { spec.model = "local-fixed-response"; spec.base_url = null; }
      else { spec.model = ""; spec.base_url = spec.backend === "deepseek" ? "https://api.deepseek.com" : "https://api.openai.com/v1"; }
      renderModels(node);
    }
    dirty();
  });
  $("#model-cards").addEventListener("click", (event) => {
    const button = event.target.closest("[data-clear-key]");
    if (!button) return;
    state.clearKeys.add(button.dataset.clearKey);
    delete state.pendingKeys[button.dataset.clearKey];
    renderModels(button.dataset.clearKey); dirty();
  });
  for (const id of ["save-top", "save-models"]) $("#" + id).addEventListener("click", () => saveSettings().catch((error) => toast(error.message, true)));
  $("#import-config").addEventListener("click", async () => {
    try {
      const result = await api("/api/import-config", {});
      state.config = result.config; state.keyPresent = result.key_present; state.pendingKeys = {}; state.clearKeys.clear();
      state.dirty = false; $("#dirty-status").textContent = "配置已同步";
      syncForms(); schedulePreview(); toast("已导入项目中的模型配置及本地密钥");
    } catch (error) { toast(error.message, true); }
  });
  $("#apply-checkpoint").addEventListener("click", async () => {
    try {
      const identity = state.config.evaluation.checkpoint;
      if (!identity) throw new Error("请先选择一个已训练检查点");
      const values = await api("/api/checkpoint-settings", { checkpoint: identity });
      resizeTeam(values.topology.agents);
      state.config.topology = values.topology; state.config.roles = values.roles;
      state.config.evaluation.checkpoint = identity;
      syncForms(); schedulePreview(); dirty(); toast("已应用检查点对应的团队与拓扑参数");
    } catch (error) { toast(error.message, true); }
  });
  $("#start-job").addEventListener("click", startJob);
  $("#check-plan").addEventListener("click", async () => {
    try {
      const plan = await api("/api/plan", { config: state.config, kind: state.kind });
      toast(`数据校验通过 · 样本池 ${plan.selected} 题 · 模型调用上限 ${plan.model_calls} 次 · 真实调用 ${plan.real_api_calls} 次`);
    } catch (error) { toast(error.message, true); }
  });
  $("#cancel-job").addEventListener("click", async () => {
    try { state.job = await api(`/api/jobs/${state.selectedJob}/cancel`, {}); renderJob(); }
    catch (error) { toast(error.message, true); }
  });
  $("#result-rows").addEventListener("click", (event) => {
    const row = event.target.closest("[data-result-index]");
    if (row) showRow(Number(row.dataset.resultIndex));
  });
  $("#history-list").addEventListener("click", (event) => {
    const button = event.target.closest("[data-open-job]");
    if (button) selectJob(button.dataset.openJob).catch((error) => toast(error.message, true));
  });
  $("#refresh-history").addEventListener("click", poll);
  $("#close-dialog").addEventListener("click", () => $("#detail-dialog").close());
}

async function bootstrap() {
  try {
    const result = await api("/api/bootstrap");
    state.config = result.config; state.token = result.token; state.roles = result.roles;
    state.datasets = result.datasets; state.keyPresent = result.key_present; state.jobs = result.jobs; state.checkpoints = result.checkpoints;
    bindEvents(); syncForms(); setKind(state.kind); renderDatasetTable(); renderHistory(); await refreshPreview();
    $("#connection-status").textContent = "本地引擎已连接";
    const active = state.jobs.find((job) => ["queued", "running", "cancelling"].includes(job.status));
    if (active) { state.selectedJob = active.id; state.job = await api(`/api/jobs/${active.id}`); renderJob(); }
    setInterval(poll, 1500);
  } catch (error) {
    $("#connection-status").textContent = "连接失败，请重新启动平台";
    toast(error.message, true);
  }
}

bootstrap();
