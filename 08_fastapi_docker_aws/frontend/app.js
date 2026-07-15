/*
 * Commercial Assistant — frontend logic (vanilla, no build step).
 *
 * The service is stateless: the browser owns the full conversation and replays
 * it on every request. Here we:
 *   - load GET /models to build the provider -> model selectors and learn, per
 *     provider, whether a server-side key exists (server_keys) — which decides
 *     if the BYOK field is Required or Optional;
 *   - stream POST /invoke/stream over fetch (EventSource can't POST a body nor
 *     send the X-LLM-API-Key header), parsing the SSE frames by hand;
 *   - persist settings, per-provider BYOK keys, and the conversation history in
 *     localStorage (fully compatible with the stateless server);
 *   - mirror the persisted pipeline via GET /leads.
 *
 * The BYOK key is only ever sent in the X-LLM-API-Key header — never logged.
 */

"use strict";

// Empty string = same origin. deploy/frontend_deploy.sh rewrites window.API_BASE_URL.
const API_BASE = (window.API_BASE_URL || "").replace(/\/+$/, "");

const LS = {
  theme: "ca:theme",
  engine: "ca:engine",
  provider: "ca:provider",
  model: (p) => `ca:model:${p}`,
  key: (p) => `ca:key:${p}`,
  history: "ca:history",
};

// ------------------------------------------------------------------ //
// State                                                               //
// ------------------------------------------------------------------ //
let modelsData = null; // { providers, default_models, server_keys }
let history = []; // [{ role, content }]  — replayed to the server each turn
let streaming = false;

// ------------------------------------------------------------------ //
// DOM                                                                 //
// ------------------------------------------------------------------ //
const $ = (id) => document.getElementById(id);
const providerSel = $("provider");
const modelSel = $("model");
const engineEl = $("engine");
const engineBtns = [...engineEl.querySelectorAll(".seg")];
const keyInput = $("apiKey");
const keyToggle = $("keyToggle");
const keyBadge = $("keyBadge");
const keyHint = $("keyHint");
const messagesEl = $("messages");
const emptyState = $("emptyState");
const banner = $("banner");
const form = $("composer");
const input = $("input");
const sendBtn = $("sendBtn");
const clearBtn = $("clearBtn");
const themeToggle = $("themeToggle");
const leadsList = $("leadsList");
const leadsCount = $("leadsCount");
const leadsRefresh = $("leadsRefresh");
const docsLink = $("docsLink");

// Small DOM helper: element with class and text (text set via textContent — safe).
function el(tag, cls, txt) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
}

// ------------------------------------------------------------------ //
// Theme                                                               //
// ------------------------------------------------------------------ //
function currentTheme() {
  return (
    document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")
  );
}
themeToggle.addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem(LS.theme, next);
});

// ------------------------------------------------------------------ //
// Settings: engine / provider / model / key                          //
// ------------------------------------------------------------------ //
let engine = localStorage.getItem(LS.engine) || "langgraph";

function setEngine(value) {
  engine = value;
  localStorage.setItem(LS.engine, value);
  for (const b of engineBtns) {
    const on = b.dataset.engine === value;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  }
}
engineBtns.forEach((b) => b.addEventListener("click", () => setEngine(b.dataset.engine)));

function keyRequired(provider) {
  return !(modelsData && modelsData.server_keys && modelsData.server_keys[provider]);
}
function getKey(provider) {
  return localStorage.getItem(LS.key(provider)) || "";
}
function updateKeyBadge() {
  const p = providerSel.value;
  const req = keyRequired(p);
  keyBadge.textContent = req ? "Required" : "Optional";
  keyBadge.className = "badge " + (req ? "req" : "opt");
  keyHint.textContent = req
    ? "No server-side key for this provider — enter your own to use it."
    : "A server-side key is configured; yours is optional and overrides it.";
}

function onProviderChange() {
  const p = providerSel.value;
  localStorage.setItem(LS.provider, p);

  // Rebuild the model list, filtered to this provider; preselect the default.
  const models = (modelsData.providers && modelsData.providers[p]) || [];
  modelSel.replaceChildren();
  for (const m of models) modelSel.appendChild(new Option(m, m));
  const savedModel = localStorage.getItem(LS.model(p));
  modelSel.value =
    savedModel && models.includes(savedModel)
      ? savedModel
      : (modelsData.default_models && modelsData.default_models[p]) || models[0] || "";

  keyInput.value = getKey(p);
  updateKeyBadge();
}

providerSel.addEventListener("change", onProviderChange);
modelSel.addEventListener("change", () =>
  localStorage.setItem(LS.model(providerSel.value), modelSel.value),
);
keyInput.addEventListener("input", () => {
  const p = providerSel.value;
  const v = keyInput.value;
  if (v) localStorage.setItem(LS.key(p), v);
  else localStorage.removeItem(LS.key(p));
});
keyToggle.addEventListener("click", () => {
  const show = keyInput.type === "password";
  keyInput.type = show ? "text" : "password";
  keyToggle.textContent = show ? "Hide" : "Show";
});

// ------------------------------------------------------------------ //
// Banner                                                              //
// ------------------------------------------------------------------ //
function showBanner(msg) {
  banner.textContent = msg;
  banner.hidden = false;
}
function clearBanner() {
  banner.hidden = true;
}

// ------------------------------------------------------------------ //
// /models                                                             //
// ------------------------------------------------------------------ //
async function loadModels() {
  let data;
  try {
    const r = await fetch(`${API_BASE}/models`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch {
    showBanner("Could not load /models — is the API running?");
    input.disabled = true;
    return;
  }
  modelsData = data;

  providerSel.replaceChildren();
  for (const p of Object.keys(data.providers || {})) providerSel.appendChild(new Option(p, p));
  const savedProvider = localStorage.getItem(LS.provider);
  if (savedProvider && data.providers && data.providers[savedProvider]) {
    providerSel.value = savedProvider;
  }
  onProviderChange();
  setEngine(engine);
  updateSendState();
}

// ------------------------------------------------------------------ //
// /leads                                                              //
// ------------------------------------------------------------------ //
async function refreshLeads() {
  let leads;
  try {
    const r = await fetch(`${API_BASE}/leads`);
    if (!r.ok) return;
    leads = await r.json();
  } catch {
    return;
  }
  leadsCount.textContent = String(leads.length);
  leadsList.replaceChildren();
  if (!leads.length) {
    leadsList.appendChild(el("p", "leads-empty", "No leads yet."));
    return;
  }
  for (const lead of leads) {
    const item = el("div", "lead-item");
    const top = el("div", "lead-top");
    top.appendChild(el("span", "lead-name", lead.name || lead.id));
    top.appendChild(el("span", `lead-status s-${lead.status}`, lead.status));
    item.appendChild(top);
    if (lead.company) item.appendChild(el("div", "lead-company", lead.company));
    leadsList.appendChild(item);
  }
}
leadsRefresh.addEventListener("click", refreshLeads);

// ------------------------------------------------------------------ //
// Chat rendering                                                      //
// ------------------------------------------------------------------ //
function scrollToBottom() {
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function addUserTurn(text) {
  emptyState.hidden = true;
  const turn = el("div", "turn user");
  turn.appendChild(el("div", "bubble", text));
  messagesEl.appendChild(turn);
  scrollToBottom();
}

// Create an assistant turn. When `done` is true it renders a finished bubble
// (used when restoring history); otherwise it returns a live handle to stream into.
function addAssistantTurn(text, done) {
  emptyState.hidden = true;
  const turn = el("div", "turn asst");
  const bubble = el("div", "bubble");
  const content = el("span", "content");
  if (text) content.textContent = text;
  bubble.appendChild(content);
  turn.appendChild(bubble);
  const meta = el("div", "meta");
  turn.appendChild(meta);
  messagesEl.appendChild(turn);

  let cursor = null;
  if (!done) {
    cursor = el("span", "cursor");
    bubble.appendChild(cursor);
  }
  scrollToBottom();

  return {
    text: text || "",
    appendToken(delta) {
      this.text += delta;
      content.textContent = this.text;
      scrollToBottom();
    },
    finalize(final) {
      if (cursor) cursor.remove();
      if (!this.text && final.reply) content.textContent = final.reply;
      renderMeta(meta, final);
      scrollToBottom();
    },
    fail(detail) {
      if (cursor) cursor.remove();
      bubble.classList.add("error");
      content.textContent = this.text ? `${this.text}\n\n⚠ ${detail}` : `⚠ ${detail}`;
      scrollToBottom();
    },
  };
}

function renderMeta(meta, final) {
  // Tool-actions trace (collapsible).
  const calls = final.tool_calls || [];
  if (calls.length) {
    const det = el("details", "actions");
    const sum = el("summary", null, `Actions · ${calls.length} tool call${calls.length > 1 ? "s" : ""}`);
    det.appendChild(sum);
    for (const tc of calls) {
      const row = el("div", "tool-row");
      const line = el("div");
      line.appendChild(el("span", "tool-name", tc.name));
      line.appendChild(el("span", "tool-args", `(${JSON.stringify(tc.args)})`));
      row.appendChild(line);
      if (tc.result) row.appendChild(el("div", "tool-res", `→ ${tc.result}`));
      det.appendChild(row);
    }
    meta.appendChild(det);
  }

  // Email draft card.
  const d = final.email_draft;
  if (d) {
    const card = el("div", "draft");
    card.appendChild(el("div", "draft-head", "✉ Email draft"));
    const to = el("div", "draft-line");
    to.appendChild(el("b", null, "To: "));
    to.appendChild(document.createTextNode(d.to || ""));
    card.appendChild(to);
    const subj = el("div", "draft-line");
    subj.appendChild(el("b", null, "Subject: "));
    subj.appendChild(document.createTextNode(d.subject || ""));
    card.appendChild(subj);
    card.appendChild(el("div", "draft-body", d.body || ""));
    meta.appendChild(card);
  }

  // Leads touched.
  const touched = final.leads_touched || [];
  if (touched.length) {
    const row = el("div", "touched");
    row.appendChild(el("span", "touched-label", "Leads touched:"));
    for (const id of touched) row.appendChild(el("span", "lead-chip", id));
    meta.appendChild(row);
  }

  // Token usage.
  const u = final.usage;
  if (u) {
    meta.appendChild(
      el("div", "usage", `in ${u.input_tokens} · out ${u.output_tokens} · total ${u.total_tokens} tokens`),
    );
  }
}

// ------------------------------------------------------------------ //
// History persistence                                                 //
// ------------------------------------------------------------------ //
function saveHistory() {
  localStorage.setItem(LS.history, JSON.stringify(history));
}
function restoreHistory() {
  try {
    history = JSON.parse(localStorage.getItem(LS.history) || "[]");
  } catch {
    history = [];
  }
  if (!Array.isArray(history)) history = [];
  for (const m of history) {
    if (m.role === "user") addUserTurn(m.content);
    else if (m.role === "assistant") addAssistantTurn(m.content, true);
  }
}
function clearConversation() {
  history = [];
  saveHistory();
  [...messagesEl.querySelectorAll(".turn")].forEach((n) => n.remove());
  emptyState.hidden = false;
  clearBanner();
}
clearBtn.addEventListener("click", clearConversation);

// ------------------------------------------------------------------ //
// SSE streaming over fetch                                            //
// ------------------------------------------------------------------ //
// Parse one raw SSE frame (lines separated by \n) into { event, data }.
function parseSseFrame(raw) {
  let event = "message";
  const dataLines = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith(":")) continue; // comment / keep-alive ping
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
  }
  return { event, data: dataLines.join("\n") };
}

async function streamInvoke(body, apiKey, onToken, onFinal, onError) {
  const headers = { "Content-Type": "application/json" };
  if (apiKey) headers["X-LLM-API-Key"] = apiKey;

  let resp;
  try {
    resp = await fetch(`${API_BASE}/invoke/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    });
  } catch {
    onError("Network error: could not reach the API.");
    return;
  }

  // Errors resolved before the stream starts (401/422/429/…) arrive as a normal
  // JSON response, not SSE.
  if (!resp.ok || !resp.body) {
    let detail = `Request failed (HTTP ${resp.status}).`;
    try {
      const j = await resp.json();
      if (j.detail) detail = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* keep the generic message */
    }
    onError(detail);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    // Strip raw CR bytes: sse-starlette separates frames with CRLF+CRLF, so the
    // frame boundary on the wire is "\r\n\r\n". Removing the (structural-only)
    // CRs lets us split on a plain "\n\n". A real CR inside token text arrives
    // JSON-escaped, never as a raw 0x0D byte, so this is safe.
    buffer += decoder.decode(value, { stream: true }).replace(/\r/g, "");

    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, sep);
      buffer = buffer.slice(sep + 2);
      const frame = parseSseFrame(raw);
      if (!frame.data) continue;

      // Provider failure mid-stream: server sends `event: error`.
      if (frame.event === "error") {
        let detail = "The LLM provider returned an error.";
        try {
          detail = JSON.parse(frame.data).detail || detail;
        } catch {
          /* keep the generic message */
        }
        onError(detail);
        return;
      }

      let payload;
      try {
        payload = JSON.parse(frame.data);
      } catch {
        continue;
      }
      if (payload.type === "token") onToken(payload.content || "");
      else if (payload.type === "final") onFinal(payload);
    }
  }
}

// ------------------------------------------------------------------ //
// Send flow                                                           //
// ------------------------------------------------------------------ //
function updateSendState() {
  sendBtn.disabled = streaming || !input.value.trim() || !modelsData;
}
function setStreaming(on) {
  streaming = on;
  input.disabled = on;
  updateSendState();
}

async function send() {
  const text = input.value.trim();
  if (!text || streaming || !modelsData) return;

  const provider = providerSel.value;
  const apiKey = getKey(provider);
  if (keyRequired(provider) && !apiKey) {
    showBanner(`An API key is required for "${provider}". Add it in the sidebar.`);
    keyInput.focus();
    return;
  }
  clearBanner();

  addUserTurn(text);
  history.push({ role: "user", content: text });
  saveHistory();

  input.value = "";
  autoGrow();
  const asst = addAssistantTurn("", false);
  setStreaming(true);

  let committed = false;
  await streamInvoke(
    { engine, provider, model: modelSel.value || null, messages: history },
    apiKey,
    (token) => asst.appendToken(token),
    (final) => {
      asst.finalize(final);
      history.push({ role: "assistant", content: asst.text || final.reply || "" });
      saveHistory();
      committed = true;
    },
    (detail) => {
      asst.fail(detail);
      // Roll the user turn back out of the replayed history so a resend starts
      // from the prior context instead of stacking a dead turn.
      history.pop();
      saveHistory();
      committed = true;
    },
  );

  if (!committed) {
    // Stream ended without a final event (unexpected): keep what streamed, if any.
    if (asst.text) {
      history.push({ role: "assistant", content: asst.text });
    } else {
      asst.fail("The response ended unexpectedly.");
      history.pop();
    }
    saveHistory();
  }

  setStreaming(false);
  refreshLeads();
  input.focus();
}

// ------------------------------------------------------------------ //
// Composer wiring                                                     //
// ------------------------------------------------------------------ //
function autoGrow() {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 180) + "px";
}
input.addEventListener("input", () => {
  autoGrow();
  updateSendState();
});
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});
form.addEventListener("submit", (e) => {
  e.preventDefault();
  send();
});

// Suggestion chips (empty state): fill the composer and send.
$("suggestions").addEventListener("click", (e) => {
  const btn = e.target.closest(".chip-btn");
  if (!btn) return;
  input.value = btn.dataset.q;
  autoGrow();
  send();
});

// ------------------------------------------------------------------ //
// Init                                                                //
// ------------------------------------------------------------------ //
docsLink.href = `${API_BASE}/docs`;
restoreHistory();
loadModels();
refreshLeads();
