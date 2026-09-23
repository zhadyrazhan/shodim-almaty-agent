const STARTER_QUESTIONS = [
  "Куда сходить на выходных?",
  "Посоветуй место для свидания",
  "Куда сводить ребенка?",
  "Какие концерты будут?",
  "Где вкусно поесть?",
];

const THEME_KEY = "sxodim-theme";

const messagesEl = document.getElementById("messages");
const emptyStateEl = document.getElementById("empty-state");
const composerEl = document.getElementById("composer");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");
const chipsEl = document.getElementById("chips");
const statusEl = document.getElementById("status");
const statusTextEl = document.getElementById("status-text");
const themeToggle = document.getElementById("theme-toggle");

/* --- theme ---------------------------------------------------------------- */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  try {
    localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Private browsing or blocked storage — the toggle still works per-session.
  }
}

function initTheme() {
  // Dark by default — only an explicit toggle switches to light, so the app
  // looks the same regardless of the visitor's OS setting.
  let saved = null;
  try {
    saved = localStorage.getItem(THEME_KEY);
  } catch {
    // ignore
  }
  applyTheme(saved === "light" ? "light" : "dark");
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme");
  applyTheme(current === "dark" ? "light" : "dark");
});

/* --- status --------------------------------------------------------------- */
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (data.ready) {
      statusEl.classList.add("ready");
      statusTextEl.textContent = `${data.records} мест · ${data.backend}`;
    } else {
      statusEl.classList.add("down");
      statusTextEl.textContent = "нет данных";
    }
  } catch {
    statusEl.classList.add("down");
    statusTextEl.textContent = "сервер недоступен";
  }
}

/* --- chat ----------------------------------------------------------------- */
function renderChips() {
  for (const q of STARTER_QUESTIONS) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.textContent = q;
    chip.addEventListener("click", () => ask(q));
    chipsEl.appendChild(chip);
  }
}

function setBusy(busy) {
  inputEl.disabled = busy;
  sendBtn.disabled = busy;
  for (const chip of chipsEl.querySelectorAll(".chip")) chip.disabled = busy;
}

function addMessage(role, text) {
  emptyStateEl?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

function addTyping() {
  const el = document.createElement("div");
  el.className = "typing";
  el.innerHTML = "<span></span><span></span><span></span>";
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return el;
}

async function ask(question) {
  const trimmed = question.trim();
  if (!trimmed) return;

  addMessage("user", trimmed);
  inputEl.value = "";
  setBusy(true);
  const typingEl = addTyping();

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed }),
    });
    const data = await res.json().catch(() => null);
    typingEl.remove();

    if (!res.ok) {
      addMessage("error", data?.detail ?? `Ошибка сервера (${res.status})`);
      return;
    }
    addMessage("bot", data.answer);
  } catch {
    typingEl.remove();
    addMessage("error", "Не удалось связаться с сервером. Проверь, что server.py запущен.");
  } finally {
    setBusy(false);
    inputEl.focus();
  }
}

composerEl.addEventListener("submit", (e) => {
  e.preventDefault();
  ask(inputEl.value);
});

initTheme();
renderChips();
checkHealth();
inputEl.focus();
