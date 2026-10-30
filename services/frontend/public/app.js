// страница демо: запросы идут на /api (nginx пересылает их в бэкенд), письма пользователя выводятся только через textContent
(() => {
  "use strict";

  const API = "/api";
  const ALL = "__all__";
  const STORE_KEY = "pg_conversations";
  const MAX_CONVERSATIONS = 30;

  const $ = (selector, root = document) => root.querySelector(selector);

  // ---------- разметка без innerHTML: текст письма попадает в страницу только как текст ----------
  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props || {})) {
      if (value == null || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "text") el.textContent = value;
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : value);
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  // значки в стиле lucide (константы, не пользовательский ввод)
  const SHIELD = '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>';
  const ICONS = {
    shield: SHIELD,
    "shield-alert": SHIELD + '<path d="M12 8v4"/><path d="M12 16h.01"/>',
    "shield-check": SHIELD + '<path d="m9 12 2 2 4-4"/>',
    plus: '<path d="M5 12h14"/><path d="M12 5v14"/>',
    x: '<path d="M18 6 6 18"/><path d="m6 6 12 12"/>',
    trash: '<path d="M3 6h18"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    menu: '<path d="M4 12h16"/><path d="M4 6h16"/><path d="M4 18h16"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    "arrow-up": '<path d="m5 12 7-7 7 7"/><path d="M12 19V5"/>',
    "triangle-alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "eye-off": '<path d="M10.733 5.076a10.744 10.744 0 0 1 11.205 6.575 1 1 0 0 1 0 .696 10.747 10.747 0 0 1-1.444 2.49"/><path d="M14.084 14.158a3 3 0 0 1-4.242-4.242"/><path d="M17.479 17.499a10.75 10.75 0 0 1-15.417-5.151 1 1 0 0 1 0-.696 10.75 10.75 0 0 1 4.446-5.143"/><path d="m2 2 20 20"/>',
  };

  function icon(name) {
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    for (const [k, v] of Object.entries({ viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", "stroke-width": "2", "stroke-linecap": "round", "stroke-linejoin": "round", "aria-hidden": "true" })) {
      svg.setAttribute(k, v);
    }
    svg.innerHTML = ICONS[name] || "";
    return svg;
  }

  // ---------- названия детекторов ----------
  const FIXED_LABELS = {
    keyword: "Ключевые слова",
    "judge-haiku": "Судья Haiku 5.5",
    "lora-xlmr": "LoRA (xlm-roberta)",
    "prompt-tuning-xlmr": "Prompt tuning (xlm-roberta)",
  };

  function labelOf(name) {
    if (FIXED_LABELS[name]) return FIXED_LABELS[name];
    if (name.startsWith("cascade-")) return "Каскад «" + labelOf(name.slice("cascade-".length)) + " → судья»";
    const m = /^embeddings-e5-(small|base|large)(?:-(knn|mlp))?$/.exec(name);
    if (m) return `e5-${m[1]} + ${m[2] === "knn" ? "kNN" : m[2] === "mlp" ? "MLP" : "логрегрессия"}`;
    return name;
  }

  // ---------- форматирование ----------
  const percent = (score) => `${Math.round(score * 100)}%`;
  function duration(ms) {
    if (ms < 1) return "меньше 1 мс";
    return ms < 1000 ? `${Math.round(ms)} мс` : `${(ms / 1000).toFixed(1)} с`;
  }
  const verdictText = (label) => (label === "attack" ? "Атака" : "Безопасно");

  function verdictBadge(label, large) {
    const attack = label === "attack";
    return h("span", { class: `badge ${attack ? "badge-attack" : "badge-benign"}${large ? " badge-lg" : ""}` },
      icon(attack ? "shield-alert" : "shield-check"), verdictText(label));
  }

  function meter(score, label, caption) {
    const pct = Math.round(score * 100);
    return h("div", { class: "meter" },
      caption ? h("span", { class: "meter-label", text: caption }) : null,
      h("div", { class: "meter-track", role: "progressbar", "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": String(pct), "aria-label": "Уверенность в атаке" },
        h("div", { class: `meter-fill ${label}`, style: `width:${pct}%` })),
      h("span", { class: "meter-value", text: `${pct}%` }));
  }

  // ---------- состояние и хранение ----------
  const state = { conversations: [], currentId: null, detectors: null, busy: false };

  function loadConversations() {
    try {
      const saved = JSON.parse(localStorage.getItem(STORE_KEY) || "[]");
      return Array.isArray(saved) ? saved : [];
    } catch (e) {
      return [];
    }
  }

  function saveConversations() {
    try {
      const keep = state.conversations.slice(0, MAX_CONVERSATIONS).map((c) => ({ ...c, checks: c.checks.filter((k) => k.kind !== "pending") }));
      localStorage.setItem(STORE_KEY, JSON.stringify(keep));
    } catch (e) {
      // хранилище недоступно или переполнено: история просто не сохранится
    }
  }

  const current = () => state.conversations.find((c) => c.id === state.currentId) || null;
  const uid = () => Math.random().toString(36).slice(2, 10);

  function conversationTitle(check) {
    const text = (check.subject || "").trim() || (check.body || "").trim().replace(/\s+/g, " ");
    return text.length > 48 ? text.slice(0, 47) + "…" : text || "Без темы";
  }

  function conversationTone(conv) {
    const last = conv.checks[conv.checks.length - 1];
    if (!last) return "";
    if (last.kind === "single") return last.data.label;
    if (last.kind === "all") {
      const { attack, benign } = last.data.summary;
      return attack && benign ? "mixed" : attack ? "attack" : benign ? "benign" : "";
    }
    return "";
  }

  // ---------- запросы к бэкенду ----------
  function friendlyError(status, json) {
    if (status === 504) return "Таймаут";
    const detail = json && typeof json.detail === "string" ? json.detail : "";
    if (status === 503) return `${detail || "Детектор недоступен"}. Выберите другой детектор.`;
    if (status === 404) return detail || "Детектор не найден. Обновите список.";
    if (status === 422) return "Проверьте, что текст письма не пустой.";
    return `Бэкенд ответил с ошибкой ${status}${detail ? ": " + detail : ""}.`;
  }

  async function api(path, payload) {
    let response;
    try {
      response = await fetch(API + path, payload
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
        : undefined);
    } catch (e) {
      throw new Error("Не удаётся связаться с бэкендом. Запустите сервисы командой make up и обновите страницу.");
    }
    let json = null;
    try { json = await response.json(); } catch (e) { /* ответ без тела */ }
    if (!response.ok) throw new Error(friendlyError(response.status, json));
    return json;
  }

  // ---------- отрисовка ----------
  function renderIcons(root) {
    root.querySelectorAll("[data-icon]").forEach((el) => {
      if (!el.querySelector("svg")) el.prepend(icon(el.dataset.icon));
    });
  }

  function renderHistory() {
    const box = $("#history");
    box.replaceChildren();
    if (!state.conversations.length) {
      box.append(h("p", { class: "history-empty", text: "Проверок пока нет" }));
      return;
    }
    for (const conv of state.conversations) {
      const tone = conversationTone(conv);
      const active = conv.id === state.currentId;
      box.append(h("div", { class: `history-row${active ? " is-active" : ""}` },
        h("button", {
          class: "history-item", type: "button", "aria-current": conv.id === state.currentId ? "true" : "false",
          onclick: () => { state.currentId = conv.id; closeSidebar(); renderAllViews(); },
        }, h("span", { class: `dot ${tone ? "dot-" + tone : ""}`, "aria-hidden": "true" }), h("span", { class: "history-title", text: conv.title })),
        h("button", {
          class: "icon-btn history-delete", type: "button", title: "Удалить чат", "aria-label": `Удалить чат «${conv.title}»`,
          onclick: () => deleteConversation(conv.id),
        }, icon("x"))));
    }
  }

  function deleteConversation(id) {
    const conv = state.conversations.find((c) => c.id === id);
    if (!conv || state.busy) return;
    if (!window.confirm(`Удалить чат «${conv.title}»? Историю нельзя будет восстановить.`)) return;
    state.conversations = state.conversations.filter((c) => c.id !== id);
    if (state.currentId === id) state.currentId = null;
    saveConversations();
    renderAllViews();
  }

  function renderEmpty() {
    const grid = h("div", { class: "examples" });
    for (const ex of window.EXAMPLES || []) {
      grid.append(h("button", { class: "example", type: "button", onclick: () => fillExample(ex) },
        h("span", { class: "example-title", text: ex.title }), h("span", { class: "example-hint", text: ex.hint })));
    }
    return h("section", { class: "empty" },
      h("div", { class: "empty-mark" }, icon("shield")),
      h("h2", { text: "Проверка писем на Prompt Injection" }),
      h("p", { text: "Вставьте письмо или выберите пример. Детектор скажет, пытается ли текст командовать вашим ИИ-ассистентом." }),
      grid);
  }

  function renderMail(check) {
    const body = check.body || "";
    const long = body.split("\n").length > 6 || body.length > 420;
    const bodyEl = h("div", { class: `mail-body${long ? " clamped" : ""}`, text: body });
    const detectorName = check.detector === ALL ? "Все детекторы" : check.detector ? labelOf(check.detector) : "Авто";
    const meta = h("div", { class: "mail-meta" }, h("span", { text: `Детектор: ${detectorName}` }));
    if (long) {
      const toggle = h("button", { class: "link-btn", type: "button", text: "Показать полностью" });
      toggle.addEventListener("click", () => {
        const clamped = bodyEl.classList.toggle("clamped");
        toggle.textContent = clamped ? "Показать полностью" : "Свернуть";
      });
      meta.append(toggle);
    }
    return h("div", { class: "msg-user" }, h("div", { class: "mail" },
      check.subject ? h("div", { class: "mail-subject", text: check.subject }) : null, bodyEl, meta));
  }

  function hiddenCallout(data) {
    if (!data.hidden_found) return null;
    return h("div", { class: "callout", role: "note" }, icon("eye-off"),
      h("div", {}, h("div", { class: "callout-title", text: "В письме есть текст, скрытый от читателя" }),
        h("pre", { text: data.hidden_text })));
  }

  function renderSingle(check) {
    const d = check.data;
    const facts = h("div", { class: "facts" },
      h("span", {}, "Действие шлюза ", h("b", { text: d.action === "block" ? "заблокировать" : "пропустить" })),
      h("span", {}, "Время ", h("b", { text: duration(d.latency_ms) })));
    if (d.stage && d.stage !== d.detector) {
      facts.prepend(h("span", {}, "Решила ступень ", h("b", { text: labelOf(d.stage) })));
    }
    return h("div", { class: "card" },
      h("div", { class: "card-head" }, verdictBadge(d.label, true), h("span", { class: "card-title", text: labelOf(d.detector) })),
      h("div", { class: "card-body" }, meter(d.score, d.label, "Уверенность в атаке"), facts, hiddenCallout(d)));
  }

  function renderCompare(check) {
    const d = check.data;
    const rows = d.results;
    const answered = rows.filter((r) => !r.error);

    const vote = h("div", {
      class: "vote", role: "img",
      "aria-label": `Атакой назвали ${d.summary.attack} из ${d.summary.total} детекторов`,
    }, rows.map((r) => h("span", {
      class: `vote-seg ${r.error ? "error" : r.label}`,
      title: `${labelOf(r.detector)}: ${r.error ? (r.timeout ? "таймаут" : "ошибка") : verdictText(r.label)}`,
    })));

    const head = h("div", { class: "card-head" },
      h("span", { class: "card-title", text: "Все детекторы" }),
      h("span", { class: "badge badge-attack" }, icon("shield-alert"), `Атака: ${d.summary.attack}`),
      h("span", { class: "badge badge-benign" }, icon("shield-check"), `Безопасно: ${d.summary.benign}`),
      d.summary.errors ? h("span", { class: "badge badge-outline", text: `Ошибок: ${d.summary.errors}` }) : null);

    const tbody = h("tbody", {}, rows.map((r) => {
      const name = h("td", {}, h("span", { class: "det-name", text: labelOf(r.detector) }),
        !r.error && r.stage !== r.detector ? h("span", { class: "det-stage", text: `решила ступень: ${labelOf(r.stage)}` }) : null,
        h("span", { class: "det-raw", text: r.detector }));
      if (r.error) {
        return h("tr", {}, name, h("td", { colspan: "3" }, h("span", { class: "badge badge-outline", title: r.error, text: r.timeout ? "Таймаут" : "Ошибка" })));
      }
      return h("tr", {}, name,
        h("td", {}, verdictBadge(r.label)),
        h("td", {}, h("div", { class: "cell-meter" }, h("div", { class: "meter-track" }, h("div", { class: `meter-fill ${r.label}`, style: `width:${Math.round(r.score * 100)}%` })), h("span", { class: "meter-value", text: percent(r.score) }))),
        h("td", { class: "num", text: duration(r.latency_ms) }));
    }));

    const table = h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Детектор", "Вердикт", "Уверенность в атаке", "Время"].map((t, i) => h("th", { text: t, class: i === 3 ? "num" : "" })))),
      tbody));

    const unavailable = d.unavailable && d.unavailable.length
      ? h("details", { class: "details" }, h("summary", { text: `Недоступны: ${d.unavailable.length}` }),
        h("ul", {}, d.unavailable.map((u) => h("li", {}, h("code", { text: u.detector }), ": " + u.reason))))
      : null;

    return h("div", { class: "card" }, head, h("div", { class: "vote-box" }, vote), answered.length ? table : null,
      d.hidden_found ? h("div", { class: "card-body", style: "padding-top:12px" }, hiddenCallout(d)) : null, unavailable);
  }

  function renderBot(check) {
    let content;
    if (check.kind === "pending") {
      content = h("div", { class: "skeleton" }, h("span", { class: "spinner" }), "Проверяю письмо…");
    } else if (check.kind === "error") {
      content = h("div", { class: "error-state", role: "alert" },
        h("div", { class: "error-row" }, icon("triangle-alert"), h("span", { text: check.error })),
        h("button", { class: "link-btn", type: "button", disabled: state.busy, onclick: () => retry(check), text: "Повторить" }));
    } else {
      content = check.kind === "all" ? renderCompare(check) : renderSingle(check);
    }
    return h("div", { class: "msg-bot" }, content);
  }

  function renderThread() {
    const thread = $("#thread");
    thread.replaceChildren();
    const conv = current();
    $("#topbar-title").textContent = conv ? conv.title : "Новая проверка";
    if (!conv || !conv.checks.length) {
      thread.append(renderEmpty());
      return;
    }
    for (const check of conv.checks) thread.append(renderMail(check), renderBot(check));
  }

  function renderAllViews() {
    renderHistory();
    renderThread();
    updateSend();
  }

  function scrollToEnd() {
    const box = $("#scroll");
    box.scrollTop = box.scrollHeight;
  }

  // ---------- форма ----------
  function updateSend() {
    $("#send").disabled = state.busy || !$("#body").value.trim();
  }

  function autosize() {
    const body = $("#body");
    body.style.height = "auto";
    body.style.height = Math.min(body.scrollHeight, 240) + "px";
  }

  function fillExample(ex) {
    $("#subject").value = ex.subject;
    $("#body").value = ex.body;
    autosize();
    updateSend();
    $("#body").focus();
  }

  function populateDetectors() {
    const select = $("#detector");
    const info = state.detectors;
    select.replaceChildren();
    select.append(h("option", { value: ALL, text: "Все детекторы" }));
    select.append(h("option", { value: "", text: "Авто (лучший из готовых)" }));
    if (!info) return;
    const cascades = info.available.filter((n) => n.startsWith("cascade-"));
    const single = info.available.filter((n) => !n.startsWith("cascade-"));
    if (cascades.length) select.append(h("optgroup", { label: "Каскады" }, cascades.map((n) => h("option", { value: n, text: labelOf(n) }))));
    if (single.length) select.append(h("optgroup", { label: "Одиночные детекторы" }, single.map((n) => h("option", { value: n, text: labelOf(n) }))));
    select.value = ALL;  // по умолчанию проверяем всеми детекторами
  }

  async function submit(event) {
    if (event) event.preventDefault();
    const subject = $("#subject").value.trim();
    const body = $("#body").value;
    if (state.busy) return;
    if (!body.trim()) { $("#body").focus(); return; }
    const detector = $("#detector").value;

    let conv = current();
    if (!conv) {
      conv = { id: uid(), title: conversationTitle({ subject, body }), checks: [] };
      state.conversations.unshift(conv);
      state.currentId = conv.id;
    }
    const check = { id: uid(), subject, body, detector, kind: "pending" };
    conv.checks.push(check);
    $("#subject").value = "";
    $("#body").value = "";
    autosize();
    await runCheck(check);
  }

  async function runCheck(check) {
    check.kind = "pending";
    delete check.error;
    state.busy = true;
    renderAllViews();
    scrollToEnd();
    try {
      if (check.detector === ALL) {
        check.data = await api("/compare", { subject: check.subject, body: check.body });
        check.kind = "all";
      } else {
        const payload = { subject: check.subject, body: check.body };
        if (check.detector) payload.detector = check.detector;
        check.data = await api("/check", payload);
        check.kind = "single";
      }
    } catch (e) {
      check.kind = "error";
      check.error = e.message;
    }
    state.busy = false;
    saveConversations();
    renderAllViews();
    scrollToEnd();
  }

  function retry(check) {
    if (!state.busy) runCheck(check);
  }

  // ---------- боковая панель, тема, статус ----------
  function openSidebar() { document.body.classList.add("sidebar-open"); }
  function closeSidebar() { document.body.classList.remove("sidebar-open"); }

  async function checkBackend() {
    const status = $("#status");
    try {
      await api("/health");
      status.dataset.state = "ok";
      status.title = "Бэкенд работает";
      status.setAttribute("aria-label", "Бэкенд работает");
    } catch (e) {
      status.dataset.state = "error";
      status.title = "Бэкенд недоступен";
      status.setAttribute("aria-label", "Бэкенд недоступен");
    }
  }

  async function loadDetectors() {
    try {
      state.detectors = await api("/detectors");
    } catch (e) {
      state.detectors = null;
    }
    populateDetectors();
  }

  // ---------- запуск ----------
  function init() {
    renderIcons(document);
    state.conversations = loadConversations();
    populateDetectors();
    renderAllViews();

    $("#composer").addEventListener("submit", submit);
    $("#body").addEventListener("input", () => { autosize(); updateSend(); });
    $("#composer").addEventListener("keydown", (e) => {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e);
    });
    $("#new-chat").addEventListener("click", () => {
      state.currentId = null;
      closeSidebar();
      renderAllViews();
      $("#body").focus();
    });
    $("#menu-btn").addEventListener("click", openSidebar);
    $("#sidebar-close").addEventListener("click", closeSidebar);
    $("#scrim").addEventListener("click", closeSidebar);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });

    const templates = $("#template");
    templates.append(h("option", { value: "", text: "Шаблон" }));
    (window.EXAMPLES || []).forEach((ex, i) => templates.append(h("option", { value: String(i), text: ex.title, title: ex.hint })));
    templates.addEventListener("change", () => {
      const ex = (window.EXAMPLES || [])[Number(templates.value)];
      templates.value = "";  // список всегда показывает «Шаблон», выбор только подставляет текст
      if (ex) fillExample(ex);
    });

    checkBackend();
    loadDetectors();
  }

  init();
})();
