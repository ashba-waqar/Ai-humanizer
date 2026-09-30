"use strict";

const els = {
  source: document.getElementById("sourceText"),
  output: document.getElementById("outputText"),
  sourceCounts: document.getElementById("sourceCounts"),
  outputCounts: document.getElementById("outputCounts"),
  humanizeBtn: document.getElementById("humanizeBtn"),
  copyBtn: document.getElementById("copyBtn"),
  error: document.getElementById("errorBanner"),
  themeToggle: document.getElementById("themeToggle"),
};

// --- Counts ---------------------------------------------------------------
function countText(str) {
  const chars = str.length;
  const words = str.trim() ? str.trim().split(/\s+/).length : 0;
  return `${words} ${words === 1 ? "word" : "words"} · ${chars} ${chars === 1 ? "char" : "chars"}`;
}

function updateSourceCounts() {
  els.sourceCounts.textContent = countText(els.source.value);
}
els.source.addEventListener("input", updateSourceCounts);

// --- Errors ---------------------------------------------------------------
function showError(message) {
  els.error.textContent = message;
  els.error.hidden = false;
}
function clearError() {
  els.error.hidden = true;
  els.error.textContent = "";
}

// --- Loading state --------------------------------------------------------
function setLoading(loading) {
  els.humanizeBtn.disabled = loading;
  els.humanizeBtn.classList.toggle("is-loading", loading);
  els.output.classList.toggle("is-busy", loading);
  els.humanizeBtn.querySelector(".humanize-btn__label").textContent = loading
    ? "Rewriting… this takes a few minutes"
    : "Humanize";
}

// --- Main action ----------------------------------------------------------
async function humanize() {
  const text = els.source.value.trim();
  clearError();

  if (!text) {
    showError("Add some text to the Source panel first.");
    els.source.focus();
    return;
  }

  setLoading(true);
  try {
    const res = await fetch("/api/humanize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await res.json().catch(() => ({}));

    if (!res.ok) {
      throw new Error(data.error || `Request failed (${res.status}).`);
    }

    renderResult(data.result || "");
  } catch (err) {
    showError(err.message || "Something went wrong. Please try again.");
  } finally {
    setLoading(false);
  }
}

function renderResult(result) {
  els.output.textContent = result;
  els.output.dataset.value = result;
  els.outputCounts.textContent = countText(result);
  els.copyBtn.disabled = !result;
}

els.humanizeBtn.addEventListener("click", humanize);

// Ctrl/Cmd + Enter to run from the textarea.
els.source.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    e.preventDefault();
    humanize();
  }
});

// --- Copy -----------------------------------------------------------------
els.copyBtn.addEventListener("click", async () => {
  const value = els.output.dataset.value || "";
  if (!value) return;
  try {
    await navigator.clipboard.writeText(value);
    els.copyBtn.textContent = "Copied ✓";
    els.copyBtn.classList.add("is-copied");
    setTimeout(() => {
      els.copyBtn.textContent = "Copy";
      els.copyBtn.classList.remove("is-copied");
    }, 1600);
  } catch {
    showError("Couldn't access the clipboard. Select the text and copy manually.");
  }
});

// --- Theme toggle ---------------------------------------------------------
(function initTheme() {
  const stored = localStorage.getItem("humanizer-theme");
  if (stored) document.documentElement.setAttribute("data-theme", stored);

  els.themeToggle.addEventListener("click", () => {
    const current =
      document.documentElement.getAttribute("data-theme") ||
      (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("humanizer-theme", next);
  });
})();

updateSourceCounts();
