/* reasoning-lab front end: zones, not a chat box.
   left  = problem bank + run form
   mid   = live reasoning tree (SVG)
   right = step inspector with the intervention controls
   below = event stream, candidate ranking, metrics, trace replay            */

"use strict";

const COL_W = 150;
const ROW_H = 58;
const BOX_W = 128;
const BOX_H = 30;

const els = {};
["status-line", "problem-list", "problem-count", "filter-subject", "filter-level",
  "run-strategy", "run-prm", "btn-start", "tree", "tree-count", "canvas-hint",
  "btn-pause", "btn-resume", "btn-stop", "node-id", "inspector", "event-log",
  "candidate-table", "metrics-box", "run-list", "replay-key", "btn-replay"]
  .forEach((id) => (els[id] = document.getElementById(id)));

const state = {
  problems: [],
  problem: null,
  session: null,
  snapshot: null,
  selected: null,
  stream: null,
  refreshTimer: null,
  mode: "live",
  runId: "",
  viewBox: { x: 0, y: 0, w: 900, h: 600 },
};

const api = {
  get: (path) => fetch(path).then(handle),
  post: (path, body) =>
    fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body || {}),
    }).then(handle),
};

async function handle(res) {
  const text = await res.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch (err) {
    throw new Error(`server returned non-JSON (${res.status})`);
  }
  if (!res.ok) throw new Error((data && data.detail) || res.statusText);
  return data;
}

function say(message, isError) {
  els["status-line"].textContent = message;
  els["status-line"].style.color = isError ? "var(--fail)" : "";
}

/* ---------------------------------------------------------------- problems */

async function loadProblems() {
  const subject = els["filter-subject"].value;
  const level = els["filter-level"].value;
  const query = new URLSearchParams();
  if (subject) query.set("subject", subject);
  if (level) query.set("level", level);
  const data = await api.get(`/api/problems?${query.toString()}`);
  state.problems = data.problems;
  els["problem-count"].textContent = `${data.count} problems`;
  els["problem-list"].innerHTML = "";
  data.problems.forEach((p) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="pid">${p.id}</span><span class="pq"></span><span class="plvl">${p.level}</span>`;
    li.querySelector(".pq").textContent = p.question;
    li.onclick = () => {
      state.problem = p;
      document.querySelectorAll(".problem-list li").forEach((n) => n.classList.remove("selected"));
      li.classList.add("selected");
      say(`selected ${p.id} · ${p.subject}/${p.level}`);
    };
    if (state.problem && p.id === state.problem.id) li.classList.add("selected");
    els["problem-list"].append(li);
  });
  const levels = [...new Set(state.problems.map((p) => p.level))].sort();
  const select = els["filter-level"];
  if (select.dataset.filled !== String(levels.length)) {
    select.innerHTML = '<option value="">all levels</option>' + levels.map((l) => `<option>${l}</option>`).join("");
    select.dataset.filled = String(levels.length);
    select.onchange = loadProblems;
  }
}

async function loadRunForm() {
  const data = await api.get("/api/strategies");
  els["run-strategy"].innerHTML = data.strategies
    .map((s) => `<option value="${s.name}" title="${s.description}">${s.name}</option>`)
    .join("");
  els["run-prm"].innerHTML = data.prm
    .map((p) => `<option${p === "symbolic" ? " selected" : ""}>${p}</option>`)
    .join("");
}

/* ----------------------------------------------------------------- session */

function configFromForm() {
  const numeric = ["n-samples", "branch-factor", "max-depth", "beam-width", "max-llm-calls"];
  const out = {};
  numeric.forEach((key) => {
    const input = document.getElementById(`cfg-${key}`);
    if (input && input.value !== "") out[key.replace(/-/g, "_")] = Number(input.value);
  });
  const temp = document.getElementById("cfg-temperature");
  if (temp && temp.value !== "") out.temperature = Number(temp.value);
  return out;
}

async function startSearch() {
  if (!state.problem) return say("pick a problem first", true);
  stopStream();
  try {
    const session = await api.post("/api/sessions", {
      problem_id: state.problem.id,
      strategy: els["run-strategy"].value,
      prm: els["run-prm"].value,
      config: configFromForm(),
    });
    state.session = session;
    state.snapshot = null;
    state.selected = null;
    els["event-log"].innerHTML = "";
    els["btn-pause"].disabled = false;
    say(`session ${session.id} · ${session.strategy} · prm ${els["run-prm"].value}`);
    openStream(session.id);
    scheduleRefresh(0);
  } catch (err) {
    say(`start failed: ${err.message}`, true);
  }
}

function openStream(id) {
  state.stream = new EventSource(`/api/sessions/${id}/events?last_seq=0`);
  state.stream.onmessage = (evt) => {
    let event;
    try {
      event = JSON.parse(evt.data);
    } catch (err) {
      return;
    }
    if (event.kind === "stream_end") {
      state.stream.close();
      state.stream = null;
      scheduleRefresh(0);
      return;
    }
    logEvent(event);
    scheduleRefresh(300);
  };
  state.stream.onerror = () => scheduleRefresh(1000);
}

function stopStream() {
  if (state.stream) {
    state.stream.close();
    state.stream = null;
  }
  if (state.refreshTimer) {
    clearTimeout(state.refreshTimer);
    state.refreshTimer = null;
  }
}

function scheduleRefresh(delay) {
  if (state.refreshTimer) return;
  state.refreshTimer = setTimeout(async () => {
    state.refreshTimer = null;
    if (!state.session) return;
    try {
      const [snapshot, session] = await Promise.all([
        api.get(`/api/sessions/${state.session.id}/tree`),
        api.get(`/api/sessions/${state.session.id}`),
      ]);
      state.snapshot = snapshot;
      state.session = session;
      syncControls();
      renderTree();
      renderMonitor();
      // keep an in-progress edit on screen; clicking a button blurs the textarea
      if (document.activeElement !== document.getElementById("edit-text")) renderInspector();
      if (session.state === "finished" || session.state === "stopped" || session.state === "failed") {
        finishSession(session);
      }
    } catch (err) {
      if (String(err.message).includes("no such session")) return;
      say(`refresh failed: ${err.message}`, true);
    }
  }, delay);
}

function finishSession(session) {
  stopStream();
  const verdict = session.error
    ? `error: ${session.error}`
    : `answer ${session.answer ?? "none"} · gold ${session.gold ?? "?"} · ${session.summary ? session.summary.nodes : 0} nodes`;
  say(`${session.state} · ${verdict}`, Boolean(session.error));
}

async function control(action, body) {
  if (!state.session) return;
  try {
    const result = await api.post(`/api/sessions/${state.session.id}/${action}`, body || {});
    // pause and resume report the new state: applying it now keeps the buttons honest
    if (result && result.id === state.session.id && result.state) {
      state.session = result;
      syncControls();
    }
    scheduleRefresh(0);
  } catch (err) {
    say(`${action} refused: ${err.message}`, true);
  }
}

/* --------------------------------------------------------------- tree view */

function flatten(node, list) {
  list.push(node);
  (node.children || []).forEach((child) => flatten(child, list));
  return list;
}

function nestFromFlat(tree) {
  const byId = {};
  Object.values(tree.nodes || {}).forEach((n) => (byId[n.id] = { ...n, childIds: n.children || [], children: [] }));
  Object.values(byId).forEach((n) => {
    n.childIds.forEach((c) => {
      if (byId[c]) byId[c].parent = byId[n.id];
    });
  });
  const root = byId[tree.root_id];
  Object.values(byId).forEach((n) => {
    const parent = n.parent;
    if (parent && parent !== n) parent.children.push(n);
  });
  return root || { id: tree.root_id, children: [] };
}

function layoutTree(root) {
  let cursor = 0;
  const place = (node, depth) => {
    node.depthRow = depth;
    const kids = node.children || [];
    if (!kids.length) {
      node.slot = cursor++;
      return node.slot;
    }
    const slots = kids.map((k) => place(k, depth + 1));
    node.slot = (Math.min(...slots) + Math.max(...slots)) / 2;
    return node.slot;
  };
  place(root, 0);
  return cursor;
}

function nodeClass(node, selectedId) {
  const parts = ["node"];
  const status = node.verification && node.verification.status;
  if (node.state === "pruned") parts.push("pruned");
  else if (status === "fail") parts.push("fail");
  else if (status === "pass") parts.push("pass");
  else parts.push("unknown");
  if (node.final_answer) parts.push("answer");
  if (node.id === selectedId) parts.push("selected");
  return parts.join(" ");
}

function clip(text, max) {
  const one = String(text || "").replace(/\s+/g, " ").trim();
  if (!one) return "…";
  return one.length > max ? `${one.slice(0, max - 1)}…` : one;
}

function renderTree() {
  const svg = els.tree;
  svg.innerHTML = "";
  const root = state.snapshot ? state.snapshot.tree : state.replayTree;
  if (!root) return;
  els["canvas-hint"].classList.add("hidden");
  const leaves = layoutTree(root);
  const nodes = flatten(root, []);
  const rows = Math.max(...nodes.map((n) => n.depthRow));
  const w = Math.max(6, leaves) * COL_W + BOX_W;
  const h = (rows + 1) * ROW_H + 40;
  state.viewBox = { x: -20, y: 10, w: Math.max(w, 600), h: Math.max(h, 320) };
  svg.setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${state.viewBox.w} ${state.viewBox.h}`);

  const NS = "http://www.w3.org/2000/svg";
  const pos = (n) => ({ x: n.slot * COL_W + BOX_W / 2, y: n.depthRow * ROW_H + 20 });
  nodes.forEach((n) => {
    (n.children || []).forEach((c) => {
      const a = pos(n);
      const b = pos(c);
      const path = document.createElementNS(NS, "path");
      const dead = c.state === "pruned" ? " edge dead" : " edge";
      path.setAttribute("class", dead.trim());
      path.setAttribute("d", `M${a.x} ${a.y + BOX_H / 2} C${a.x} ${a.y + ROW_H * 0.7}, ${b.x} ${b.y - ROW_H * 0.3}, ${b.x} ${b.y - BOX_H / 2}`);
      svg.append(path);
    });
  });

  nodes.forEach((n) => {
    const { x, y } = pos(n);
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", nodeClass(n, state.selected));
    const rect = document.createElementNS(NS, "rect");
    rect.setAttribute("x", x - BOX_W / 2);
    rect.setAttribute("y", y - BOX_H / 2);
    rect.setAttribute("width", BOX_W);
    rect.setAttribute("height", BOX_H);
    rect.setAttribute("rx", 4);
    const label = document.createElementNS(NS, "text");
    label.setAttribute("x", x - BOX_W / 2 + 6);
    label.setAttribute("y", y - 1);
    label.textContent = clip(n.text || n.id, 17);
    const score = document.createElementNS(NS, "text");
    score.setAttribute("class", "score");
    score.setAttribute("x", x - BOX_W / 2 + 6);
    score.setAttribute("y", y + 11);
    const bits = [];
    if (n.prm_score != null) bits.push(n.prm_score.toFixed(2));
    if (n.branch_score != null) bits.push(`b${n.branch_score.toFixed(2)}`);
    if (n.visits) bits.push(`v${n.visits}`);
    if (n.final_answer) bits.push(`=${clip(n.final_answer, 8)}`);
    score.textContent = bits.join(" ");
    const title = document.createElementNS(NS, "title");
    title.textContent = `${n.id} · d${n.depthRow} · ${n.state}\n${n.text || ""}`;
    g.append(rect, label, score, title);
    g.onclick = (event) => {
      event.stopPropagation();
      state.selected = n.id;
      renderTree();
      renderInspector();
    };
    svg.append(g);
  });

  const summary = state.snapshot ? state.snapshot.tree : null;
  els["tree-count"].textContent = state.snapshot
    ? `${(state.snapshot.metrics || {}).nodes_created ?? 0} nodes · ${nodes.length} shown`
    : `replay · ${nodes.length} nodes`;
  if (state.session) els["node-id"].textContent = state.session.id;
}

/* --------------------------------------------------------------- inspector */

function findNode(root, id) {
  if (!root) return null;
  if (root.id === id) return root;
  for (const child of root.children || []) {
    const hit = findNode(child, id);
    if (hit) return hit;
  }
  return null;
}

function renderInspector() {
  const host = els.inspector;
  host.innerHTML = "";
  const root = state.snapshot ? state.snapshot.tree : state.replayTree;
  const node = findNode(root, state.selected);
  if (!node) {
    host.innerHTML = '<p class="empty">click a node in the tree</p>';
    return;
  }
  const check = (tag, value) => `<div>${tag}</div><div>${value ?? "–"}</div>`;
  const v = node.verification;
  const head = document.createElement("div");
  head.className = "kv";
  head.innerHTML = [
    check("node", node.id),
    check("state", node.state),
    check("depth", node.depth),
    check("prm", node.prm_score != null ? node.prm_score.toFixed(3) : null),
    check("branch", node.branch_score != null ? node.branch_score.toFixed(3) : null),
    check("visits", node.visits),
    check("answer", node.final_answer),
    check("status", v ? `<span class="status-${v.status}">${v.status}</span>` : "not checked"),
  ].join("");
  host.append(head);

  const body = document.createElement("div");
  body.className = "step-text";
  body.textContent = node.text || "(root)";
  host.append(body);

  if (v && v.checks && v.checks.length) {
    const table = document.createElement("table");
    table.className = "tbl";
    table.innerHTML =
      "<thead><tr><th>kind</th><th>expression</th><th>status</th><th>detail</th></tr></thead>" +
      v.checks
        .map(
          (c) =>
            `<tr><td>${c.kind}</td><td>${c.expression || ""}</td>` +
            `<td class="status-${c.status}">${c.status}</td><td>${c.detail || ""}</td></tr>`
        )
        .join("");
    host.append(document.createElement("h3"), table);
  }
  if (v && v.notes) {
    const notes = document.createElement("p");
    notes.className = "status-unknown";
    notes.textContent = v.notes;
    host.append(notes);
  }

  if (state.mode === "replay") {
    const hint = document.createElement("p");
    hint.className = "empty";
    hint.textContent = "replayed trace: read-only";
    host.append(hint);
    return;
  }

  const editor = document.createElement("textarea");
  editor.id = "edit-text";
  editor.value = node.text || "";
  host.append(document.createElement("h3"), editor);

  const actions = document.createElement("div");
  actions.className = "actions";
  const button = (label, handler, kind) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = handler;
    if (kind) b.className = kind;
    actions.append(b);
  };
  button("re-verify edit", () => control("edit", { node_id: node.id, text: editor.value }), "primary");
  button("inject as child", () => control("inject", { parent_id: node.id, text: editor.value }));
  button("prune branch", () => control("prune", { node_id: node.id }));
  button("take this answer", () => control("answer", { node_id: node.id }));
  host.append(actions);
}

/* ---------------------------------------------------------------- monitoring */

function logEvent(event) {
  const li = document.createElement("li");
  const body = JSON.stringify(event.payload || {}, null, 0);
  li.innerHTML =
    `<span class="seq">${event.seq}</span><span class="kind">${event.kind}</span>` +
    `<span class="body"></span>`;
  li.querySelector(".body").textContent = `${event.node_id || "-"} ${body}`;
  els["event-log"].append(li);
  while (els["event-log"].children.length > 400) els["event-log"].firstChild.remove();
}

function renderMonitor() {
  const rows = (state.snapshot && state.snapshot.candidates) || [];
  els["candidate-table"].tBodies[0].innerHTML = rows
    .slice(0, 40)
    .map(
      (c) =>
        `<tr><td>${c.answer}</td><td>${c.branch_score == null ? "–" : c.branch_score.toFixed(3)}</td>` +
        `<td class="status-${c.status || "not_checked"}">${c.status || "–"}</td><td>${c.depth}</td></tr>`
    )
    .join("");
  els["metrics-box"].textContent = JSON.stringify(
    { session: state.session && state.session.state, votes: state.session && state.session.votes, metrics: state.snapshot && state.snapshot.metrics },
    null,
    2
  );
}

/* ------------------------------------------------------------------- replay */

async function loadRuns() {
  const data = await api.get("/api/runs");
  els["run-list"].innerHTML = "";
  data.runs.forEach((run) => {
    const li = document.createElement("li");
    li.textContent = `${run.run_id} · ${run.records} records`;
    li.onclick = async () => {
      state.runId = run.run_id;
      try {
        const jobs = await api.get(`/api/runs/${run.run_id}/jobs`);
        els["replay-key"].innerHTML = jobs.jobs
          .map((job) => `<option value="${job.key}">${job.key}</option>`)
          .join("");
        say(`run ${run.run_id}: ${jobs.total} traced jobs`);
      } catch (err) {
        say(`jobs failed: ${err.message}`, true);
      }
    };
    els["run-list"].append(li);
  });
  if (!data.runs.length) els["run-list"].innerHTML = "<li>no runs yet — use `reasoning-lab eval`</li>";
}

async function loadTrace() {
  const key = els["replay-key"].value;
  if (!state.runId || !key) return;
  try {
    const data = await api.get(`/api/runs/${state.runId}/jobs/${encodeURIComponent(key)}`);
    state.replayTree = nestFromFlat(data.tree || {});
    state.snapshot = null;
    state.selected = null;
    setMode("replay");
    renderTree();
    renderInspector();
    say(`replayed ${key}`);
  } catch (err) {
    say(`replay failed: ${err.message}`, true);
  }
}

/* --------------------------------------------------------------- ui wiring */

function syncControls() {
  const live = state.mode === "live";
  const status = (state.session || {}).state;
  const done = ["finished", "stopped", "failed"].includes(status);
  els["btn-pause"].disabled = !live || status !== "running";
  els["btn-resume"].disabled = !live || status !== "paused";
  els["btn-stop"].disabled = !live || !state.session || done;
  document.querySelectorAll("#zone-tree,#zone-inspect").forEach((z) => z.classList.toggle("readonly", !live));
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("active", b.dataset.mode === mode));
  syncControls();
}

document.querySelectorAll(".mode").forEach((b) => {
  b.onclick = () => {
    setMode(b.dataset.mode);
    if (b.dataset.mode === "replay") loadRuns();
    renderTree();
    renderInspector();
  };
});

document.querySelectorAll(".tab").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === b));
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("hidden", p.id !== `pane-${b.dataset.tab}`));
  };
});

els["btn-start"].onclick = startSearch;
els["btn-pause"].onclick = () => control("pause");
els["btn-resume"].onclick = () => control("resume");
els["btn-stop"].onclick = () => control("stop");
els["btn-replay"].onclick = loadTrace;
els["filter-subject"].onchange = loadProblems;

/* pan and zoom on the tree canvas */
(function enableCanvas(svg) {
  let drag = null;
  svg.addEventListener("wheel", (event) => {
    event.preventDefault();
    const box = { ...state.viewBox };
    const factor = event.deltaY > 0 ? 1.12 : 0.9;
    const point = new DOMPoint(event.offsetX, event.offsetY).matrixTransform(svg.getScreenCTM().inverse());
    box.w *= factor;
    box.h *= factor;
    box.x = point.x - (point.x - box.x) * factor;
    box.y = point.y - (point.y - box.y) * factor;
    state.viewBox = box;
    svg.setAttribute("viewBox", `${box.x} ${box.y} ${box.w} ${box.h}`);
  });
  svg.addEventListener("mousedown", (event) => (drag = { x: event.clientX, y: event.clientY, box: { ...state.viewBox } }));
  window.addEventListener("mouseup", () => (drag = null));
  window.addEventListener("mousemove", (event) => {
    if (!drag) return;
    const scale = state.viewBox.w / svg.clientWidth;
    state.viewBox.x = drag.box.x - (event.clientX - drag.x) * scale;
    state.viewBox.y = drag.box.y - (event.clientY - drag.y) * scale;
    svg.setAttribute("viewBox", `${state.viewBox.x} ${state.viewBox.y} ${state.viewBox.w} ${state.viewBox.h}`);
  });
})(els.tree);

loadProblems().catch((err) => say(err.message, true));
loadRunForm().catch((err) => say(err.message, true));
setMode("live");
say("offline demo backend: start a search with prm=symbolic");
