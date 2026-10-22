document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const browseBtn = document.getElementById("browseBtn");
  const dropPrompt = document.getElementById("dropPrompt");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const clearBtn = document.getElementById("clearBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const analyzeSpinner = document.getElementById("analyzeSpinner");

  const emptyState = document.getElementById("emptyState");
  const resultsContent = document.getElementById("resultsContent");
  const diagName = document.getElementById("diagName");
  const diagFullName = document.getElementById("diagFullName");
  const diagSeverityBadge = document.getElementById("diagSeverityBadge");
  const diagConfidenceNumber = document.getElementById("diagConfidenceNumber");
  const diagDescription = document.getElementById("diagDescription");
  const diagActionText = document.getElementById("diagActionText");
  const inferenceLatency = document.getElementById("inferenceLatency");
  const topBarsContainer = document.getElementById("topBarsContainer");

  // State
  let currentFile = null;
  let currentPreviewDataUrl = "";

  // Initialize
  initApp();

  async function initApp() {
    initTheme();
    initHistory();
    await fetchModelInfo();
    setupEventListeners();
  }

  // Theme Management
  function initTheme() {
    const themeToggleBtn = document.getElementById("themeToggleBtn");
    const themeToggleText = document.getElementById("themeToggleText");
    const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
    updateThemeUI(currentTheme);

    if (themeToggleBtn) {
      themeToggleBtn.addEventListener("click", () => {
        const activeTheme = document.documentElement.getAttribute("data-theme") || "light";
        const newTheme = activeTheme === "dark" ? "light" : "dark";
        document.documentElement.setAttribute("data-theme", newTheme);
        try {
          localStorage.setItem("dermvit_theme", newTheme);
        } catch (e) {
          console.warn("Could not save theme preference:", e);
        }
        updateThemeUI(newTheme);
      });
    }

    // Respond to system theme preference change if user hasn't explicitly set preference
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      try {
        if (!localStorage.getItem("dermvit_theme")) {
          const sysTheme = e.matches ? "dark" : "light";
          document.documentElement.setAttribute("data-theme", sysTheme);
          updateThemeUI(sysTheme);
        }
      } catch (err) {}
    });

    function updateThemeUI(theme) {
      if (themeToggleText) {
        themeToggleText.textContent = theme === "dark" ? "Light Mode" : "Dark Mode";
      }
    }
  }

  async function fetchModelInfo() {
    try {
      const res = await fetch("/api/info");
      const data = await res.json();
      
      const statusIndicator = document.getElementById("modelStatusIndicator");
      const statusText = document.getElementById("modelStatusText");
      if (data.status === "ready") {
        statusIndicator.className = "status-indicator active";
        statusText.textContent = "model/vit-skin-disease";
      }
    } catch (err) {
      console.warn("Could not fetch model info:", err);
    }
  }

  function setupEventListeners() {
    // Browse button click
    browseBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      fileInput.click();
    });

    dropZone.addEventListener("click", () => {
      if (!currentFile) {
        fileInput.click();
      }
    });

    // File input change
    fileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelection(e.target.files[0]);
      }
    });

    // Drag and Drop
    ["dragenter", "dragover"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove("dragover");
      });
    });

    dropZone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      if (dt.files && dt.files[0]) {
        handleFileSelection(dt.files[0]);
      }
    });

    // Clear Button
    clearBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      resetInput();
    });

    // Run Analysis Button
    analyzeBtn.addEventListener("click", () => {
      runAnalysis();
    });
  }

  function handleFileSelection(file) {
    if (!file.type.startsWith("image/")) {
      alert("Please upload a valid image file (PNG, JPG, JPEG, WEBP)");
      return;
    }
    currentFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
      currentPreviewDataUrl = e.target.result;
      showPreview(e.target.result);
      analyzeBtn.disabled = false;
      // Automatically analyze for instant response
      runAnalysis();
    };
    reader.readAsDataURL(file);
  }

  function showPreview(src) {
    imagePreview.src = src;
    dropPrompt.classList.add("hidden");
    previewContainer.classList.remove("hidden");
  }

  function resetInput() {
    currentFile = null;
    currentPreviewDataUrl = "";
    fileInput.value = "";
    imagePreview.src = "";
    previewContainer.classList.add("hidden");
    dropPrompt.classList.remove("hidden");
    analyzeBtn.disabled = true;
    emptyState.classList.remove("hidden");
    resultsContent.classList.add("hidden");
  }

  async function runAnalysis() {
    if (!currentFile) return;

    // Loading State
    analyzeBtn.disabled = true;
    analyzeSpinner.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Server inference failed");
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      alert("Prediction Error: " + err.message);
    } finally {
      analyzeBtn.disabled = false;
      analyzeSpinner.classList.add("hidden");
    }
  }

  function renderResults(res, saveToHistory = true) {
    emptyState.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    const top = res.topPrediction;

    // Primary Diagnosis Header
    diagName.textContent = top.name;
    diagFullName.textContent = `${top.fullName} (${top.code})`;
    diagSeverityBadge.textContent = `${top.type} • ${top.severity}`;
    
    // Adaptive severity badge styles
    const severityLower = (top.severity || "").toLowerCase();
    diagSeverityBadge.className = "diag-badge";
    diagSeverityBadge.style.backgroundColor = "";
    diagSeverityBadge.style.color = "";
    diagSeverityBadge.style.borderColor = "";

    if (severityLower.includes("high") || severityLower.includes("critical")) {
      diagSeverityBadge.classList.add("severity-high");
    } else if (severityLower.includes("moderate")) {
      diagSeverityBadge.classList.add("severity-moderate");
    } else {
      diagSeverityBadge.classList.add("severity-low");
    }

    diagConfidenceNumber.textContent = `${top.percentage}%`;
    inferenceLatency.textContent = typeof res.inferenceTimeMs === "number" ? `${res.inferenceTimeMs}ms inference` : res.inferenceTimeMs;

    // Populate Detailed Clinical Dossier
    const dossier = res.dossier || {};

    // 1. Optical Visual Biomarkers Grid (The 4 Factors)
    const biomarkersEl = document.getElementById("dossierBiomarkers");
    if (biomarkersEl && dossier.biomarkers) {
      biomarkersEl.innerHTML = dossier.biomarkers.map(b => {
        const tagColor = (b.color && b.color !== "#1E293B") ? b.color : "var(--text-muted)";
        return `
        <div class="biomarker-card">
          <span class="biomarker-label">${b.label}</span>
          <strong class="biomarker-val">${b.value}</strong>
          <span class="biomarker-tag" style="color: ${tagColor};">${b.badge}</span>
        </div>
      `;
      }).join("");
    }

    // 2. Yeh Kya Hai? (Clinical Overview)
    const overviewEl = document.getElementById("conditionOverview");
    if (overviewEl) {
      overviewEl.textContent = dossier.overview || top.description || "Clinical pathology description of this classification.";
    }

    // 3. Kiski Wajah Se Hoti Hai? (Causes & Pathological Mechanism)
    const pathologyEl = document.getElementById("conditionPathology");
    if (pathologyEl) {
      pathologyEl.textContent = dossier.pathology || dossier.pathologyNarrative || top.description || "Underlying biological and histological etiology.";
    }

    // Top-5 Bars in Rose Gold to Coral gradient
    const topItems = res.top5 || res.top3 || (res.allPredictions ? res.allPredictions.slice(0, 5) : []);
    topBarsContainer.innerHTML = topItems.map(item => `
      <div class="bar-item">
        <div class="bar-info">
          <span><strong style="color:var(--text-heading)">${item.name}</strong> <small style="color:var(--text-dim)">(${item.code})</small></span>
          <span style="font-weight:700; color: #C4677C">${item.percentage}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width: 0%; background: linear-gradient(90deg, #E8C7EA 0%, #E3A5AF 50%, #C4677C 100%);" data-target-width="${item.percentage}%"></div>
        </div>
      </div>
    `).join("");

    // Animate bars after brief DOM paint
    requestAnimationFrame(() => {
      document.querySelectorAll(".bar-fill").forEach(fill => {
        fill.style.width = fill.dataset.targetWidth;
      });
    });

    // Save to History (if new scan)
    if (saveToHistory) {
      saveScanRecord({
        id: "scan_" + Date.now(),
        timestamp: getFormattedTimestamp(),
        name: top.name,
        fullName: top.fullName,
        code: top.code,
        severity: top.severity,
        type: top.type,
        percentage: top.percentage,
        overview: dossier.overview || top.description || "",
        pathology: dossier.pathology || dossier.pathologyNarrative || "",
        biomarkers: dossier.biomarkers || [],
        top5: topItems
      });
    }

    // Smooth scroll into results if viewport is compact
    if (window.innerWidth < 960) {
      resultsContent.scrollIntoView({ behavior: "smooth" });
    }
  }

  // ==========================================
  // Diagnostic Scan History (Last 25 Records)
  // ==========================================
  const HISTORY_KEY = "dermvit_scan_history_v2";

  function initHistory() {
    renderHistoryUI();
    const clearBtn = document.getElementById("clearHistoryBtn");
    if (clearBtn) {
      clearBtn.addEventListener("click", () => {
        if (confirm("Are you sure you want to clear your saved scan history?")) {
          localStorage.removeItem(HISTORY_KEY);
          renderHistoryUI();
        }
      });
    }
  }

  function getHistoryList() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch (e) {
      return [];
    }
  }

  function saveScanRecord(record) {
    createThumbnail(currentPreviewDataUrl, (thumb) => {
      record.thumbnail = thumb;
      let history = getHistoryList();
      // Add newest record at top
      history.unshift(record);
      // Strictly maintain maximum 25 records
      if (history.length > 25) {
        history = history.slice(0, 25);
      }
      try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
      } catch (e) {
        console.warn("Storage quota exceeded, trimming history:", e);
        history = history.slice(0, 10);
        try {
          localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        } catch (err) {}
      }
      renderHistoryUI();
    });
  }

  function renderHistoryUI() {
    const history = getHistoryList();
    const counterBadge = document.getElementById("historyCounterBadge");
    const emptyState = document.getElementById("historyEmpty");
    const historyGrid = document.getElementById("historyGrid");

    if (counterBadge) {
      counterBadge.textContent = `${history.length} / 25 Records Saved`;
    }

    if (!historyGrid || !emptyState) return;

    if (history.length === 0) {
      emptyState.classList.remove("hidden");
      historyGrid.classList.add("hidden");
      historyGrid.innerHTML = "";
      return;
    }

    emptyState.classList.add("hidden");
    historyGrid.classList.remove("hidden");

    historyGrid.innerHTML = history.map((rec) => {
      const sevClass = (rec.severity || "").toLowerCase().includes("high") || (rec.severity || "").toLowerCase().includes("critical")
        ? "severity-high"
        : (rec.severity || "").toLowerCase().includes("moderate")
        ? "severity-moderate"
        : "severity-low";

      return `
        <div class="history-item-card" data-scan-id="${rec.id}" title="Click to view details of this diagnostic record">
          <div class="history-thumb-wrap">
            <img src="${rec.thumbnail || ''}" alt="${rec.name}" class="history-thumb" />
          </div>
          <div class="history-info">
            <div class="history-top-row">
              <span class="history-name">${rec.name}</span>
              <span class="history-conf">${rec.percentage}%</span>
            </div>
            <div class="history-diag">${rec.fullName} (${rec.code})</div>
            <div class="history-meta-row">
              <span class="history-time">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                  <circle cx="12" cy="12" r="10"></circle>
                  <polyline points="12 6 12 12 16 14"></polyline>
                </svg>
                ${rec.timestamp}
              </span>
              <span class="diag-badge ${sevClass}" style="padding: 0.15rem 0.5rem; font-size: 0.68rem;">${rec.severity}</span>
            </div>
          </div>
          <button type="button" class="btn btn-outline btn-sm" style="flex-shrink:0;">View</button>
        </div>
      `;
    }).join("");

    // Add click listeners to history cards
    historyGrid.querySelectorAll(".history-item-card").forEach(card => {
      card.addEventListener("click", () => {
        const id = card.dataset.scanId;
        loadHistoricalRecord(id);
      });
    });
  }

  function loadHistoricalRecord(scanId) {
    const history = getHistoryList();
    const rec = history.find(h => h.id === scanId);
    if (!rec) return;

    if (rec.thumbnail) {
      showPreview(rec.thumbnail);
    }

    renderResults({
      topPrediction: {
        name: rec.name,
        fullName: rec.fullName,
        code: rec.code,
        severity: rec.severity,
        type: rec.type,
        percentage: rec.percentage,
        description: rec.overview
      },
      dossier: {
        overview: rec.overview,
        pathology: rec.pathology,
        biomarkers: rec.biomarkers
      },
      top5: rec.top5 || [],
      inferenceTimeMs: "Saved Record"
    }, false); // don't duplicate into history

    const resultsSection = document.getElementById("resultsSection");
    if (resultsSection) {
      resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  function createThumbnail(dataUrl, callback) {
    if (!dataUrl) {
      callback("");
      return;
    }
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      const size = 88;
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext("2d");
      const minDim = Math.min(img.width, img.height);
      const sx = (img.width - minDim) / 2;
      const sy = (img.height - minDim) / 2;
      ctx.drawImage(img, sx, sy, minDim, minDim, 0, 0, size, size);
      callback(canvas.toDataURL("image/jpeg", 0.7));
    };
    img.onerror = () => callback(dataUrl);
    img.src = dataUrl;
  }

  function getFormattedTimestamp() {
    const now = new Date();
    const dateStr = now.toLocaleDateString("en-US", { day: "numeric", month: "short", year: "numeric" });
    const timeStr = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: true });
    return `${dateStr} • ${timeStr}`;
  }
});
