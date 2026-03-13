/* ============================================================
   FraudShield — main.js
   Handles form submission, validation, API call, and result rendering
   ============================================================ */

"use strict";

// ── DOM references ────────────────────────────────────────────────────────────
const form        = document.getElementById("fraud-form");
const submitBtn   = document.getElementById("submit-btn");
const formError   = document.getElementById("form-error");

const resultIdle    = document.getElementById("result-idle");
const resultLoading = document.getElementById("result-loading");
const resultOutput  = document.getElementById("result-output");

const verdictIcon  = document.getElementById("verdict-icon");
const verdictLabel = document.getElementById("verdict-label");
const verdictText  = document.getElementById("verdict-text");
const probValue    = document.getElementById("prob-value");
const probBar      = document.getElementById("prob-bar");
const riskBadge    = document.getElementById("risk-badge");
const featureGrid  = document.getElementById("feature-grid");
const resetBtn     = document.getElementById("reset-btn");

// ── Label helpers ─────────────────────────────────────────────────────────────
const DAY_LABELS  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"];
const TYPE_LABELS = ["UPI Payment","ATM Withdrawal","Debit/Credit Card (POS)","NEFT/RTGS/IMPS"];
const CAT_LABELS  = ["Kirana/Grocery","Electronics","Travel & Transport","Restaurant/Dining","Entertainment/OTT",
                     "Healthcare/Pharmacy","Clothing & Fashion","Petrol/CNG","Utilities & Bills","Other"];

function friendlyLabel(key, val) {
  switch (key) {
    case "amount":                 return `₹${parseFloat(val).toLocaleString("en-IN", {minimumFractionDigits:2,maximumFractionDigits:2})}`;
    case "hour_of_day":            return `${val}:00`;
    case "day_of_week":            return DAY_LABELS[val] || val;
    case "transaction_type":       return TYPE_LABELS[val] || val;
    case "merchant_category":      return CAT_LABELS[val] || val;
    case "distance_from_home":     return `${parseFloat(val).toFixed(1)} km`;
    case "num_prev_transactions":  return `${val} txns`;
    case "account_age_days":       return `${val} days`;
    case "failed_attempts":        return `${val} attempts`;
    case "is_international":       return val == 1 ? "Yes" : "No";
    default: return val;
  }
}

function friendlyKey(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

// ── State helpers ─────────────────────────────────────────────────────────────
function showPanel(which) {
  resultIdle.hidden    = (which !== "idle");
  resultLoading.hidden = (which !== "loading");
  resultOutput.hidden  = (which !== "output");
}

function clearErrors() {
  formError.hidden = true;
  formError.textContent = "";
  form.querySelectorAll(".invalid").forEach(el => el.classList.remove("invalid"));
}

function showError(msg, fieldId = null) {
  formError.textContent = `⚠ ${msg}`;
  formError.hidden = false;
  if (fieldId) {
    const el = document.getElementById(fieldId);
    if (el) el.classList.add("invalid");
  }
}

// ── Client-side validation ────────────────────────────────────────────────────
function validateForm(data) {
  const checks = [
    { id:"amount",               min:1,     max:4000000, label:"Amount" },
    { id:"hour_of_day",          min:0,     max:23,     label:"Hour" },
    { id:"day_of_week",          min:0,     max:6,      label:"Day" },
    { id:"transaction_type",     min:0,     max:3,      label:"Transaction type" },
    { id:"merchant_category",    min:0,     max:9,      label:"Merchant category" },
    { id:"distance_from_home",   min:0,     max:20000,  label:"Distance" },
    { id:"num_prev_transactions",min:0,     max:500,    label:"Previous transactions" },
    { id:"account_age_days",     min:0,     max:36500,  label:"Account age" },
    { id:"failed_attempts",      min:0,     max:10,     label:"Failed attempts" },
    { id:"is_international",     min:0,     max:1,      label:"International field" },
  ];

  for (const { id, min, max, label } of checks) {
    const raw = data[id];
    if (raw === "" || raw === null || raw === undefined) {
      return { ok: false, msg: `${label} is required.`, field: id };
    }
    const num = parseFloat(raw);
    if (isNaN(num) || num < min || num > max) {
      return { ok: false, msg: `${label} must be between ${min} and ${max}.`, field: id };
    }
  }
  return { ok: true };
}

// ── Render result ─────────────────────────────────────────────────────────────
function renderResult(data, formData) {
  const isFraud = data.prediction === 1;
  const pct     = data.probability; // 0–100

  // Verdict
  verdictIcon.textContent = isFraud ? "🚨" : "✅";
  verdictLabel.textContent = "Prediction";
  verdictText.textContent  = isFraud ? "Fraudulent" : "Legitimate";
  verdictText.className    = "verdict-text " + (isFraud ? "fraud" : "legitimate");

  // Probability bar
  probValue.textContent = `${pct.toFixed(1)}%`;
  // Offset the gradient so colour corresponds to risk
  probBar.style.backgroundPositionX = `${100 - pct}%`;
  requestAnimationFrame(() => {
    probBar.style.width = `${pct}%`;
  });

  // Risk badge
  riskBadge.textContent = data.risk_level;
  riskBadge.className   = `risk-badge ${data.risk_level}`;

  // Feature snapshot
  featureGrid.innerHTML = "";
  for (const key of data.features_used) {
    const raw = formData[key];
    const item = document.createElement("div");
    item.className = "feat-item";
    item.innerHTML = `
      <span class="feat-name">${friendlyKey(key)}</span>
      <span class="feat-value">${friendlyLabel(key, raw)}</span>
    `;
    featureGrid.appendChild(item);
  }

  showPanel("output");
}

// ── Submit handler ────────────────────────────────────────────────────────────
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearErrors();

  // Collect raw form data
  const raw = {};
  const fd  = new FormData(form);
  fd.forEach((val, key) => { raw[key] = val; });

  // Client-side validation
  const check = validateForm(raw);
  if (!check.ok) {
    showError(check.msg, check.field);
    return;
  }

  // UI → loading
  submitBtn.disabled = true;
  showPanel("loading");

  try {
    const res = await fetch("/predict", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(raw),
    });

    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || `Server error (${res.status})`);
    }

    renderResult(data, raw);

  } catch (err) {
    showPanel("idle");
    showError(err.message || "Network error. Please try again.");
  } finally {
    submitBtn.disabled = false;
  }
});

// ── Reset ─────────────────────────────────────────────────────────────────────
resetBtn.addEventListener("click", () => {
  form.reset();
  clearErrors();
  probBar.style.width = "0%";
  showPanel("idle");
});

// ── Live input clear on focus ─────────────────────────────────────────────────
form.querySelectorAll("input, select").forEach(el => {
  el.addEventListener("focus", () => el.classList.remove("invalid"));
});