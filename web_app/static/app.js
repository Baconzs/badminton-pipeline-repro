/* ShuttleVision local workbench — intentionally dependency-free. */
(function () {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const TERMINAL = new Set(["completed", "failed", "cancelled"]);
  const JOB_STATUSES = new Set(["queued", "running", "completed", "failed", "cancelled"]);
  const QUALITY_LABELS = {
    hit_anchored: "双击锚定",
    single_hit: "单击锚定",
    trajectory_fallback: "轨迹回退",
  };
  const HAWKEYE_REASON_LABELS = Object.freeze({
    no_rallies: "没有可复核的有效回合",
    no_confirmed_hit: "没有确认击球事件",
    no_post_hit_observation: "最后击球后没有真实球点",
    shot_continues_to_next_contact: "后续已检测到接触，请回看视频",
    later_hit_unconfirmed: "后续有未确认击球，请回看视频",
    discontinuous_terminal_track: "终局轨迹不连续",
    flight_too_short: "终局轨迹持续时间不足",
    no_landing_evidence: "未观察到可靠落地事件，请回看视频",
    projection_failed: "终局轨迹无法映射到球场",
    landing_out_of_bounds: "终点超出可信球场范围",
    wrong_target_half: "终点不在击球方的目标半场",
    low_confidence: "终局轨迹证据质量不足",
    invalid_calibration: "球场四点标定无效",
    source_unavailable: "轨迹数据不可用",
    read_failed: "轨迹数据读取失败",
  });

  const state = {
    video: null,
    points: [],
    videoWidth: 0,
    videoHeight: 0,
    job: null,
    eventSource: null,
    eventSourceJobId: null,
    pollTimer: null,
    uploading: false,
    importing: false,
    sourceRequestToken: 0,
    submitting: false,
    history: [],
    jobs: [],
    health: null,
    terminalNoticeFor: null,
    jobContextToken: 0,
    coachEvidenceStop: null,
  };

  function safeJobStatus(value) {
    const status = String(value || "").toLowerCase();
    return JOB_STATUSES.has(status) ? status : "failed";
  }

  function hasQueuedOrRunningJob() {
    return Boolean(state.job && !TERMINAL.has(safeJobStatus(state.job.status)));
  }

  function hasActiveJob() {
    return state.submitting || hasQueuedOrRunningJob();
  }

  function sourceBusy() {
    return state.uploading || state.importing;
  }

  function invalidateJobContext() {
    state.jobContextToken += 1;
    clearJobPoll();
    if (state.eventSource) state.eventSource.close();
    state.eventSource = null;
    state.eventSourceJobId = null;
    return state.jobContextToken;
  }

  const dom = {
    status: $("#systemStatus"),
    dropZone: $("#dropZone"),
    fileInput: $("#fileInput"),
    uploadProgress: $("#uploadProgress"),
    uploadProgressBar: $("#uploadProgress span"),
    uploadProgressText: $("#uploadProgressText"),
    pathForm: $("#pathImportForm"),
    pathInput: $("#pathInput"),
    pathSubmit: $("button", $("#pathImportForm")),
    selectedSource: $("#selectedSource"),
    sourceThumb: $("#sourceThumb"),
    sourceName: $("#sourceName"),
    sourceMeta: $("#sourceMeta"),
    clearSource: $("#clearSourceButton"),
    splitToggle: $("#splitToggle"),
    hawkeyeToggle: $("#hawkeyeToggle"),
    proToggle: $("#proToggle"),
    // The layout may render this as either the standard checkbox switch or a
    // real toggle button.  Keep the lookup optional so older cached HTML
    // bundles continue to work (they simply use normal-speed output).
    bulletTimeToggle: $("#bulletTimeToggle"),
    coachConfig: $("#coachConfig"),
    coachTarget: $("#coachTarget"),
    nearDominantHand: $("#nearDominantHand"),
    farDominantHand: $("#farDominantHand"),
    sceneCutToggle: $("#sceneCutToggle"),
    deviceSelect: $("#deviceSelect"),
    modeExplainer: $("#modeExplainer"),
    defaultCourt: $("#defaultCourtButton"),
    clearCourt: $("#clearCourtButton"),
    calibrationCount: $("#calibrationCount"),
    calibrationStatus: $("#calibrationStatus"),
    start: $("#startButton"),
    ctaNote: $("#ctaNote"),
    mediaShell: $("#mediaShell"),
    previewVideo: $("#previewVideo"),
    courtCanvas: $("#courtCanvas"),
    mediaEmpty: $("#mediaEmpty"),
    calibrationHint: $("#calibrationHint"),
    previewTitle: $("#previewTitle"),
    previewCaption: $("#previewCaption"),
    statusLed: $(".status-led"),
    resolution: $("#resolutionChip"),
    time: $("#timeChip"),
    progressPanel: $("#progressPanel"),
    jobStage: $("#jobStage"),
    jobStatus: $("#jobStatus"),
    cancel: $("#cancelButton"),
    progressBar: $("#progressBar"),
    progressPercent: $("#progressPercent"),
    stageTimeline: $("#stageTimeline"),
    logConsole: $("#logConsole"),
    resultsPanel: $("#resultsPanel"),
    completePill: $("#completePill"),
    sidecarLink: $("#sidecarLink"),
    sidecarLabel: $("#sidecarLabel"),
    kpiGrid: $("#kpiGrid"),
    analysisOutput: $("#analysisOutput"),
    analysisVideo: $("#analysisVideo"),
    analysisTitle: $("#analysisVideoTitle"),
    analysisCaption: $("#analysisVideoNotice") || $(".analysis-output .subheading small"),
    bulletTimeHint: $("#bulletTimeHint"),
    coachingSection: $("#coachingSection"),
    coachingGrid: $("#coachingGrid"),
    coachingNotice: $("#coachingNotice"),
    strokeTypesSection: $("#strokeTypesSection"),
    strokeTypesSummary: $("#strokeTypesSummary"),
    strokeTypesGrid: $("#strokeTypesGrid"),
    strokeTypesNotice: $("#strokeTypesNotice"),
    hawkeyeSection: $("#hawkeyeSection"),
    hawkeyeGrid: $("#hawkeyeGrid"),
    hawkeyeSummary: $("#hawkeyeSummary"),
    hawkeyeNotice: $("#hawkeyeNotice"),
    rallySection: $("#rallySection"),
    rallyGrid: $("#rallyGrid"),
    rallySummary: $("#rallySummary"),
    resultEmpty: $("#resultEmpty"),
    historyPanel: $("#historyPanel"),
    historyList: $("#historyList"),
    refresh: $("#refreshButton"),
    toastStack: $("#toastStack"),
  };

  function controlChecked(control) {
    if (!control) return false;
    if (typeof control.checked === "boolean") return control.checked;
    return control.getAttribute("aria-pressed") === "true";
  }

  function setControlChecked(control, value) {
    if (!control) return;
    const checked = Boolean(value);
    if (typeof control.checked === "boolean") control.checked = checked;
    else control.setAttribute("aria-pressed", checked ? "true" : "false");
  }

  function bulletTimeSelected() {
    return controlChecked(dom.bulletTimeToggle);
  }

  function professionalSelected() {
    // Selecting the FX option implicitly selects the professional pass.  This
    // keeps the standalone button useful while preserving the pipeline's
    // requirement that FX runs after pose/TrackNet rendering.
    return Boolean((dom.proToggle && dom.proToggle.checked) || bulletTimeSelected());
  }

  function bulletConfigEnabled(config) {
    if (!config || typeof config !== "object") return false;
    // Prefer the canonical field even when it is explicitly false.  Falling
    // through with `||` would let a stale legacy alias incorrectly turn FX
    // back on when an older manifest contains both fields.
    for (const key of ["bullet_time_enabled", "bullet_time_fx", "bullet_time"]) {
      if (Object.prototype.hasOwnProperty.call(config, key)) return config[key] === true;
    }
    return false;
  }

  function bulletTimeControlCard() {
    if (!dom.bulletTimeToggle) return null;
    return dom.bulletTimeToggle.closest(".mode-card, .bullet-option, [data-bullet-time-option]");
  }

  function toast(message, kind = "info", timeout = 4200) {
    const item = document.createElement("div");
    item.className = `toast ${kind}`;
    item.textContent = message;
    dom.toastStack.appendChild(item);
    window.setTimeout(() => {
      item.classList.add("out");
      window.setTimeout(() => item.remove(), 300);
    }, timeout);
  }

  async function requestJSON(url, options = {}) {
    const response = await fetch(url, options);
    let payload = null;
    try { payload = await response.json(); } catch (_) { /* non-JSON response */ }
    if (!response.ok) {
      const reason = payload && payload.error ? payload.error : `请求失败（${response.status}）`;
      throw new Error(reason);
    }
    return payload;
  }

  function escapeHTML(value) {
    return String(value == null ? "" : value).replace(/[&<>"']/g, (character) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[character]));
  }

  function safeMediaURL(value) {
    const url = String(value || "");
    return url.startsWith("/") && !url.startsWith("//") ? url : "";
  }

  function formatDuration(seconds) {
    const value = Number(seconds);
    if (!Number.isFinite(value) || value < 0) return "—";
    const total = Math.round(value);
    const hours = Math.floor(total / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    if (hours > 0) return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    return `${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
  }

  function formatBytes(bytes) {
    const value = Number(bytes);
    if (!Number.isFinite(value) || value <= 0) return "—";
    const units = ["B", "KB", "MB", "GB", "TB"];
    const index = Math.min(units.length - 1, Math.floor(Math.log(value) / Math.log(1024)));
    return `${(value / (1024 ** index)).toFixed(index ? 1 : 0)} ${units[index]}`;
  }

  function formatNumber(value, digits = 0) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "—";
    return number.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits });
  }

  function metadataOf(video) {
    const metadata = (video && video.metadata) || {};
    const dimensions = metadata.width && metadata.height ? `${metadata.width} × ${metadata.height}` : "—";
    return `${dimensions}  ·  ${formatDuration(metadata.duration)}  ·  ${formatBytes(metadata.size_bytes)}`;
  }

  function setSystemStatus(kind, message) {
    dom.status.classList.remove("ready", "error");
    if (kind) dom.status.classList.add(kind);
    const text = $("span", dom.status);
    if (text) text.textContent = message;
  }

  function missingCapabilities() {
    const split = dom.splitToggle.checked;
    const hawkeye = dom.hawkeyeToggle.checked;
    const professional = professionalSelected();
    const bulletTime = bulletTimeSelected();
    if (!split && !hawkeye && !professional) return [];
    if (!state.health) return ["正在检测分析引擎"];
    const weights = state.health.weights || {};
    const missing = [];
    if (!weights.tracknet) missing.push("TrackNet 权重");
    if (professional && !weights.pose) missing.push("姿态模型权重");
    if ((split || bulletTime) && !state.health.ffmpeg) missing.push("ffmpeg");
    const device = dom.deviceSelect.value;
    const accelerator = String(state.health.accelerator || "").toLowerCase();
    if (device === "cuda" && !accelerator.includes("cuda")) missing.push("可用的 CUDA 设备");
    if (device === "mps" && !accelerator.includes("mps")) missing.push("可用的 Apple MPS 设备");
    return missing;
  }

  function updateControlLock() {
    const busy = hasActiveJob();
    const sourceIsBusy = sourceBusy();
    const lockSource = busy || sourceIsBusy;
    // A new-job POST can take a moment before it returns the job id.  During
    // that window there is no cancellable server task yet; keeping the old
    // cancel button visible would make a click look successful while
    // `cancelJob()` has nothing to send to the server.
    const cancelUnavailable = (
      state.submitting
      || !state.job
      || TERMINAL.has(safeJobStatus(state.job.status))
    );
    dom.cancel.hidden = cancelUnavailable;
    dom.cancel.disabled = cancelUnavailable;
    dom.fileInput.disabled = lockSource;
    dom.dropZone.classList.toggle("disabled", lockSource);
    dom.dropZone.setAttribute("aria-disabled", lockSource ? "true" : "false");
    dom.pathInput.disabled = lockSource;
    dom.pathSubmit.disabled = lockSource;
    dom.clearSource.disabled = lockSource;
    dom.historyPanel.classList.toggle("source-busy", sourceIsBusy || busy);
    dom.splitToggle.disabled = busy;
    dom.hawkeyeToggle.disabled = busy;
    dom.proToggle.disabled = busy;
    if (dom.bulletTimeToggle) {
      const bulletDisabled = busy;
      dom.bulletTimeToggle.disabled = bulletDisabled;
      const bulletCard = bulletTimeControlCard();
      if (bulletCard) {
        bulletCard.classList.toggle("disabled", bulletDisabled);
        bulletCard.setAttribute("aria-disabled", bulletDisabled ? "true" : "false");
      }
    }
    const coachDisabled = busy || !professionalSelected();
    dom.coachTarget.disabled = coachDisabled;
    dom.nearDominantHand.disabled = coachDisabled;
    dom.farDominantHand.disabled = coachDisabled;
    dom.coachConfig.classList.toggle("disabled", coachDisabled);
    dom.sceneCutToggle.disabled = busy || (!dom.splitToggle.checked && !dom.hawkeyeToggle.checked && !professionalSelected());
    dom.deviceSelect.disabled = busy;
    dom.defaultCourt.disabled = busy || !state.video;
    dom.clearCourt.disabled = busy || !state.video;
  }

  function defaultPoints(width = state.videoWidth, height = state.videoHeight) {
    const metadata = (state.video && state.video.metadata) || {};
    const w = Math.max(1, Number(width) || Number(metadata.width) || 1);
    const h = Math.max(1, Number(height) || Number(metadata.height) || 1);
    return [
      { x: w * .14, y: h * .18, label: "TL" },
      { x: w * .86, y: h * .18, label: "TR" },
      { x: w * .86, y: h * .83, label: "BR" },
      { x: w * .14, y: h * .83, label: "BL" },
    ];
  }

  function syncCanvas() {
    const video = dom.previewVideo;
    if (!video.videoWidth || !video.videoHeight) return;
    state.videoWidth = video.videoWidth;
    state.videoHeight = video.videoHeight;
    dom.mediaShell.style.setProperty("--media-ratio", `${video.videoWidth} / ${video.videoHeight}`);
    // Keep interaction coordinates in original-video pixels, but cap the
    // backing store to the displayed resolution.  A 4K/8K source does not
    // need a 4K/8K RGBA canvas just to draw four calibration points.
    const rect = dom.mediaShell.getBoundingClientRect();
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    let canvasWidth = Math.max(1, Math.round((rect.width || Math.min(video.videoWidth, 1280)) * dpr));
    let canvasHeight = Math.max(1, Math.round(canvasWidth * video.videoHeight / video.videoWidth));
    const maxCanvasPixels = 3_000_000;
    if (canvasWidth * canvasHeight > maxCanvasPixels) {
      const scale = Math.sqrt(maxCanvasPixels / (canvasWidth * canvasHeight));
      canvasWidth = Math.max(1, Math.floor(canvasWidth * scale));
      canvasHeight = Math.max(1, Math.floor(canvasHeight * scale));
    }
    if (dom.courtCanvas.width !== canvasWidth || dom.courtCanvas.height !== canvasHeight) {
      dom.courtCanvas.width = canvasWidth;
      dom.courtCanvas.height = canvasHeight;
    }
    drawCourt();
    updateCalibrationUI();
  }

  function drawCourt() {
    const canvas = dom.courtCanvas;
    if (!canvas.width || !canvas.height) return;
    const context = canvas.getContext("2d");
    if (!context || !state.videoWidth || !state.videoHeight) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    const scaleX = canvas.width / state.videoWidth;
    const scaleY = canvas.height / state.videoHeight;
    const project = (point) => ({ x: point.x * scaleX, y: point.y * scaleY });
    const points = state.points;
    if (points.length >= 2) {
      context.save();
      context.beginPath();
      const first = project(points[0]);
      context.moveTo(first.x, first.y);
      points.slice(1).forEach((point) => {
        const projected = project(point);
        context.lineTo(projected.x, projected.y);
      });
      if (points.length === 4) context.closePath();
      context.fillStyle = "rgba(103, 233, 239, .06)";
      context.fill();
      context.strokeStyle = "rgba(103, 233, 239, .85)";
      context.lineWidth = Math.max(2, canvas.width / 700);
      context.setLineDash(points.length === 4 ? [] : [10, 8]);
      context.shadowColor = "rgba(103,233,239,.85)";
      context.shadowBlur = 12;
      context.stroke();
      context.restore();
    }
    points.forEach((point, index) => {
      const projected = project(point);
      const radius = Math.max(7, canvas.width / 100);
      context.save();
      context.beginPath();
      context.arc(projected.x, projected.y, radius + 5, 0, Math.PI * 2);
      context.fillStyle = "rgba(103,233,239,.13)";
      context.fill();
      context.beginPath();
      context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
      context.fillStyle = index === 2 || index === 3 ? "#c4f26b" : "#67e9ef";
      context.shadowColor = context.fillStyle;
      context.shadowBlur = 13;
      context.fill();
      context.shadowBlur = 0;
      context.fillStyle = "#081019";
      context.font = `700 ${Math.max(9, canvas.width / 115)}px DM Mono, monospace`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(point.label || ["TL", "TR", "BR", "BL"][index], projected.x, projected.y + .5);
      context.restore();
    });
  }

  function updateCalibrationUI() {
    const count = state.points.length;
    const needsCourt = dom.splitToggle.checked || dom.hawkeyeToggle.checked || professionalSelected();
    const canEditCourt = Boolean(state.video) && needsCourt && !hasActiveJob() && count < 4;
    dom.calibrationCount.textContent = `${count} / 4`;
    dom.calibrationStatus.classList.remove("ready", "warn");
    const statusText = $("span", dom.calibrationStatus);
    if (!state.video) {
      if (statusText) statusText.textContent = "选择视频后开始标定";
    } else if (!needsCourt) {
      if (statusText) statusText.textContent = "当前仅预览；启用分析后需要四点标定";
    } else if (count === 4) {
      dom.calibrationStatus.classList.add("ready");
      if (statusText) statusText.textContent = "四点已完成，可开始智能分析";
      dom.calibrationHint.innerHTML = "四点已锁定 · 使用 <b>清除</b> 可重新标定";
    } else {
      dom.calibrationStatus.classList.add("warn");
      if (statusText) statusText.textContent = `请在视频上点击第 ${count + 1} 个点（${["TL", "TR", "BR", "BL"][count]}）`;
      dom.calibrationHint.innerHTML = `点击视频中的 <b>${["TL", "TR", "BR", "BL"][count]}</b> 点位 · 依次完成四角`;
    }
    dom.mediaShell.classList.toggle("calibrating", canEditCourt);
    const unavailable = missingCapabilities();
    if (!state.video) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = "先导入视频即可开始";
    } else if (state.submitting) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = "正在创建任务，请稍候…";
    } else if (hasQueuedOrRunningJob()) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = "已有任务正在处理，请等待完成或取消";
    } else if (needsCourt && count !== 4) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = "智能分析需要完成四点标定";
    } else if (sourceBusy()) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = state.uploading ? "视频仍在上传…" : "正在导入服务器视频…";
    } else if (unavailable.length) {
      dom.start.disabled = true;
      dom.ctaNote.textContent = `分析引擎不可用：${unavailable.join("、")}`;
    } else {
      dom.start.disabled = false;
      dom.ctaNote.textContent = needsCourt ? "标定已锁定 · 可以开始分析" : "仅预览模式 · 不运行模型";
    }
    updateControlLock();
  }

  function updateModeExplainer() {
    const split = dom.splitToggle.checked;
    const hawkeye = dom.hawkeyeToggle.checked;
    const professional = professionalSelected();
    const bulletTime = bulletTimeSelected();
    const continueAfterSceneCuts = dom.sceneCutToggle.checked;
    let copy;
    if (split && professional) {
      copy = "完整模式：专业视频、可靠动作建议和回合短片。";
    } else if (professional) {
      copy = "专业模式：TrackNet、姿态、移动和 Motion Coach。";
    } else if (split) {
      copy = "轻量模式：建立球路时间轴并导出回合短片。";
    } else if (hawkeye) {
      copy = "鹰眼辅助：查看每个回合最后一拍后的终局轨迹。";
    } else {
      copy = "仅预览视频，不运行分析模型。";
    }
    if (bulletTime) copy += " 已开启子弹镜头，渲染时间更长。";
    else if (dom.bulletTimeHint) dom.bulletTimeHint.textContent = professional
      ? "可选 · 视频中加入冻结与慢动作 · 会增加渲染时间"
      : "点击后会自动开启专业分析 · 会增加渲染时间";
    if (dom.bulletTimeHint && bulletTime) {
      dom.bulletTimeHint.textContent = "已开启 · 视频中加入冻结与慢动作";
    }
    if (hawkeye) {
      copy += " 边线附近或终点不稳时，请回看原视频。";
    }
    if (split || hawkeye || professional) {
      copy += continueAfterSceneCuts
        ? " 跨镜头已开启；仅适用于同一球场全景。"
        : " 默认在第一次硬切处停止；固定全景可开启跨镜头。";
    }
    const accelerator = String((state.health && state.health.accelerator) || "").toLowerCase();
    const cpuProfessional = professional && (
      dom.deviceSelect.value === "cpu"
      || (dom.deviceSelect.value === "auto" && !accelerator.includes("cuda") && !accelerator.includes("mps"))
    );
    if (cpuProfessional) {
      copy += " CPU 模式使用稀疏姿态校验。";
    }
    const bst = state.health && state.health.bst;
    if (professional && bst && !bst.configured) {
      copy += " 未配置 BST 权重，将显示规则型击球结果。";
    } else if (professional && bst && bst.configured) {
      copy += " 已配置 BST，完成后显示逐拍击球类型 Top-3。";
    }
    const unavailable = missingCapabilities();
    if (unavailable.length && state.health) copy += ` 当前缺少：${unavailable.join("、")}。`;
    dom.modeExplainer.textContent = copy;
    updateCalibrationUI();
  }

  function stopMedia(video) {
    if (!video) return;
    try { video.pause(); } catch (_) { /* no-op */ }
    video.removeAttribute("src");
    try { video.load(); } catch (_) { /* no-op */ }
  }

  function resetResultMedia({ hideResults = true, hideProgress = false } = {}) {
    clearCoachEvidenceStop();
    stopMedia(dom.analysisVideo);
    $$("video", dom.rallyGrid).forEach(stopMedia);
    dom.rallyGrid.replaceChildren();
    dom.kpiGrid.replaceChildren();
    dom.analysisOutput.hidden = true;
    if (dom.analysisTitle) dom.analysisTitle.textContent = "专业分析视频";
    if (dom.analysisCaption) dom.analysisCaption.textContent = "含击球、姿态与移动视觉叠加";
    dom.coachingSection.hidden = true;
    dom.coachingGrid.replaceChildren();
    dom.coachingNotice.textContent = "仅展示通过证据门控的可靠建议";
    dom.strokeTypesSection.hidden = true;
    dom.strokeTypesSummary.replaceChildren();
    dom.strokeTypesGrid.replaceChildren();
    dom.strokeTypesNotice.textContent = "可选模型：逐拍击球类型 Top-3";
    dom.hawkeyeSection.hidden = true;
    dom.hawkeyeGrid.replaceChildren();
    dom.hawkeyeSummary.replaceChildren();
    dom.hawkeyeNotice.textContent = "单机位终局轨迹参考 · 请以原视频为准";
    dom.rallySection.hidden = true;
    dom.sidecarLink.hidden = true;
    dom.resultEmpty.hidden = true;
    dom.rallySummary.textContent = "智能回合 · 可下载微调";
    if (hideResults) dom.resultsPanel.hidden = true;
    if (hideProgress) dom.progressPanel.hidden = true;
  }

  function selectVideo(video, announce = true) {
    if (!video || !video.id) return;
    if (hasActiveJob()) {
      toast("当前任务仍在运行；请先等待完成或取消后再切换视频", "error");
      return;
    }
    invalidateJobContext();
    clearCoachEvidenceStop();
    state.video = video;
    state.points = [];
    state.videoWidth = 0;
    state.videoHeight = 0;
    dom.courtCanvas.width = 0;
    dom.courtCanvas.height = 0;
    state.job = null;
    resetResultMedia({ hideResults: true, hideProgress: true });
    dom.mediaShell.classList.remove("empty");
    dom.mediaEmpty.hidden = true;
    dom.previewTitle.textContent = video.name || "视频预览";
    dom.previewCaption.textContent = video.uploaded ? "UPLOADED / READY" : "LOCAL PATH / READY";
    dom.statusLed.classList.add("active");
    dom.sourceThumb.src = video.poster_url || "";
    dom.sourceName.textContent = video.name || "未命名视频";
    dom.sourceMeta.textContent = metadataOf(video);
    dom.selectedSource.hidden = false;
    dom.previewVideo.poster = video.poster_url || "";
    dom.previewVideo.src = video.content_url;
    dom.previewVideo.load();
    dom.resolution.textContent = video.metadata && video.metadata.width ? `${video.metadata.width} × ${video.metadata.height}` : "— × —";
    dom.time.textContent = `00:00 / ${formatDuration(video.metadata && video.metadata.duration)}`;
    updateCalibrationUI();
    if (announce) toast(`已载入：${video.name || "视频"}`, "success");
  }

  function clearVideo() {
    if (hasActiveJob() || sourceBusy()) {
      toast("当前任务仍在运行或正在导入视频，暂时不能更换来源", "error");
      return;
    }
    invalidateJobContext();
    clearCoachEvidenceStop();
    state.video = null;
    state.points = [];
    state.job = null;
    state.videoWidth = 0;
    state.videoHeight = 0;
    dom.courtCanvas.width = 0;
    dom.courtCanvas.height = 0;
    resetResultMedia({ hideResults: true, hideProgress: true });
    dom.previewVideo.pause();
    dom.previewVideo.removeAttribute("src");
    dom.previewVideo.load();
    dom.mediaShell.classList.add("empty");
    dom.mediaEmpty.hidden = false;
    dom.selectedSource.hidden = true;
    dom.previewTitle.textContent = "等待视频信号";
    dom.previewCaption.textContent = "PREVIEW / NO ANALYSIS";
    dom.statusLed.classList.remove("active", "done");
    dom.resolution.textContent = "— × —";
    dom.time.textContent = "00:00 / 00:00";
    updateCalibrationUI();
    drawCourt();
  }

  function uploadFile(file) {
    if (!file || sourceBusy()) return;
    if (hasActiveJob()) { toast("当前任务仍在运行，暂时不能导入新视频", "error"); return; }
    const allowed = /\.(mp4|mov|mkv|avi|webm|m4v)$/i.test(file.name);
    if (!allowed) { toast("请选择 MP4、MOV、MKV、AVI、WEBM 或 M4V 视频", "error"); return; }
    const requestToken = ++state.sourceRequestToken;
    state.uploading = true;
    dom.uploadProgress.classList.add("active");
    dom.uploadProgressBar.style.width = "0%";
    dom.uploadProgressText.textContent = "准备上传 0%";
    updateCalibrationUI();
    const form = new FormData();
    form.append("video", file, file.name);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/videos/upload");
    xhr.upload.onprogress = (event) => {
      if (!event.lengthComputable) return;
      const percent = Math.round(event.loaded / event.total * 100);
      dom.uploadProgressBar.style.width = `${percent}%`;
      dom.uploadProgressText.textContent = `正在上传 ${percent}% · ${formatBytes(event.loaded)} / ${formatBytes(event.total)}`;
    };
    xhr.onerror = () => finishUpload(null, new Error("网络连接中断"), requestToken);
    xhr.onload = () => {
      let payload = null;
      try { payload = JSON.parse(xhr.responseText); } catch (_) { /* handled below */ }
      if (xhr.status >= 200 && xhr.status < 300 && payload) finishUpload(payload, null, requestToken);
      else finishUpload(null, new Error(payload && payload.error ? payload.error : `上传失败（${xhr.status}）`), requestToken);
    };
    xhr.send(form);
  }

  function finishUpload(video, error, requestToken) {
    state.uploading = false;
    dom.uploadProgressBar.style.width = error ? "0%" : "100%";
    dom.uploadProgressText.textContent = error ? "上传失败，请重试" : "上传完成 · 正在读取视频信息";
    window.setTimeout(() => dom.uploadProgress.classList.remove("active"), 900);
    updateCalibrationUI();
    if (requestToken !== state.sourceRequestToken) return;
    if (error) { toast(error.message, "error", 6000); return; }
    if (video) {
      selectVideo(video);
      loadHistory();
    }
  }

  async function importPath(event) {
    event.preventDefault();
    if (hasActiveJob() || sourceBusy()) { toast("当前任务仍在运行或正在导入视频，暂时不能切换来源", "error"); return; }
    const path = dom.pathInput.value.trim();
    if (!path) { toast("请输入服务器本地视频路径", "error"); dom.pathInput.focus(); return; }
    const requestToken = ++state.sourceRequestToken;
    state.importing = true;
    updateCalibrationUI();
    try {
      const video = await requestJSON("/api/videos/import", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ path }) });
      if (requestToken === state.sourceRequestToken) {
        state.importing = false;
        selectVideo(video);
        dom.pathInput.value = "";
        loadHistory();
      }
    } catch (error) { toast(error.message, "error", 6000); }
    finally {
      if (requestToken === state.sourceRequestToken) state.importing = false;
      updateCalibrationUI();
    }
  }

  function courtClick(event) {
    if (hasActiveJob() || !state.video || !dom.courtCanvas.width || !dom.mediaShell.classList.contains("calibrating")) return;
    const rect = dom.courtCanvas.getBoundingClientRect();
    const x = (event.clientX - rect.left) * state.videoWidth / rect.width;
    const y = (event.clientY - rect.top) * state.videoHeight / rect.height;
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const labels = ["TL", "TR", "BR", "BL"];
    if (state.points.length >= 4) return;
    state.points.push({ x, y, label: labels[state.points.length] });
    drawCourt();
    updateCalibrationUI();
  }

  function useDefaultCourt() {
    if (!state.video) { toast("请先导入视频", "error"); return; }
    state.points = defaultPoints();
    drawCourt();
    updateCalibrationUI();
    toast("已载入建议球场范围；如需更精确，可清除后重新点击四角", "info");
  }

  function clearCourt() {
    state.points = [];
    drawCourt();
    updateCalibrationUI();
  }

  function currentStages(job) {
    const config = (job && job.config) || {};
    const stages = [{ key: "input", label: "输入校验" }];
    if (config.professional_analysis) {
      stages.push({ key: "track", label: "TrackNet 球路" }, { key: "pose", label: "姿态与移动" });
      if (bulletConfigEnabled(config)) stages.push({ key: "fx", label: "子弹镜头" });
      if (config.hawkeye_review) stages.push({ key: "hawkeye", label: "鹰眼复核" });
      stages.push({ key: "coach", label: "动作建议" });
    } else if (config.split_rallies || config.hawkeye_review) {
      stages.push({ key: "track", label: "球路时间轴" });
      if (config.hawkeye_review) stages.push({ key: "hawkeye", label: "鹰眼复核" });
    }
    if (config.split_rallies) stages.push({ key: "clips", label: "回合导出" });
    stages.push({ key: "done", label: "报告完成" });
    return stages;
  }

  function activeStageIndex(job, stages) {
    if (!job) return 0;
    if (safeJobStatus(job.status) === "completed") return stages.length - 1;
    const stage = String(job.stage || "");
    if (/回合短视频|导出回合|回合导出/.test(stage)) return Math.max(0, stages.findIndex((item) => item.key === "clips"));
    if (/特效|子弹镜头/.test(stage)) {
      const fxIndex = stages.findIndex((item) => item.key === "fx");
      if (fxIndex >= 0) return fxIndex;
    }
    if (/姿态|专业/.test(stage)) return Math.max(0, stages.findIndex((item) => item.key === "pose"));
    if (/鹰眼|落点复核|终局轨迹/.test(stage)) return Math.max(0, stages.findIndex((item) => item.key === "hawkeye"));
    if (/动作改进|动作建议/.test(stage)) return Math.max(0, stages.findIndex((item) => item.key === "coach"));
    if (/球路|TrackNet|相位|时间轴/.test(stage)) return Math.max(0, stages.findIndex((item) => item.key === "track"));
    // The normal-speed professional pass still has a Step 3 overlay stage,
    // but it is not named “特效”.  Keep that work marked in-progress instead
    // of matching the broad “整理/完成” fallback and lighting up the final
    // report stage too early.
    if (/普通速度分析|整理分析产物/.test(stage)) {
      const fxIndex = stages.findIndex((item) => item.key === "fx");
      if (fxIndex >= 0 && /视觉特效/.test(stage)) return fxIndex;
      const poseIndex = stages.findIndex((item) => item.key === "pose");
      if (poseIndex >= 0) return poseIndex;
    }
    if (/完成|整理/.test(stage)) return stages.length - 1;
    return 0;
  }

  function renderTimeline(job) {
    const stages = currentStages(job);
    const active = activeStageIndex(job, stages);
    const status = job ? safeJobStatus(job.status) : "queued";
    dom.stageTimeline.innerHTML = stages.map((stage, index) => {
      let className = "stage-item";
      if (job && (status === "completed" || index < active)) className += " done";
      else if (job && index === active && status === "running") className += " active";
      return `<div class="${className}"><span class="stage-dot"></span><span>${escapeHTML(stage.label)}</span></div>`;
    }).join("");
  }

  function renderLogs(logs) {
    const entries = Array.isArray(logs) ? logs.slice(-42) : [];
    dom.logConsole.innerHTML = `<span class="console-caret">›</span>${entries.length ? entries.map((entry) => escapeHTML(entry.message || "")).join("\n") : "等待子进程输出…"}`;
    dom.logConsole.scrollTop = dom.logConsole.scrollHeight;
  }

  function renderProgress(job) {
    if (!job) return;
    const status = safeJobStatus(job.status);
    dom.progressPanel.hidden = false;
    const progress = Math.max(0, Math.min(100, Number(job.progress) || 0));
    dom.jobStage.textContent = job.stage || "处理任务";
    dom.jobStatus.textContent = status.toUpperCase();
    dom.jobStatus.className = `job-status ${status}`;
    dom.progressBar.style.width = `${progress}%`;
    dom.progressPercent.textContent = `${Math.round(progress)}%`;
    dom.cancel.hidden = TERMINAL.has(status);
    renderTimeline(job);
    renderLogs(job.logs);
    if (status === "running") {
      dom.previewCaption.textContent = "ANALYSIS / PROCESSING";
      dom.statusLed.classList.add("active");
    }
    if (status === "failed") {
      dom.previewCaption.textContent = "ANALYSIS / FAILED";
    } else if (status === "cancelled") {
      dom.previewCaption.textContent = "ANALYSIS / CANCELLED";
    }
  }

  function kpi(label, value, unit, tone = "") {
    return `<div class="kpi ${tone}"><span class="kpi-label">${escapeHTML(label)}</span><strong class="kpi-value">${escapeHTML(value)}<em>${escapeHTML(unit || "")}</em></strong></div>`;
  }

  function setCompletePill(status) {
    const completed = status === "completed";
    const label = completed ? "COMPLETED" : status.toUpperCase();
    const led = document.createElement("i");
    dom.completePill.replaceChildren(led, document.createTextNode(` ${label}`));
    dom.completePill.style.color = completed ? "var(--lime)" : "var(--red)";
  }

  function clearCoachEvidenceStop() {
    if (typeof state.coachEvidenceStop === "function") state.coachEvidenceStop();
    state.coachEvidenceStop = null;
  }

  function playCoachEvidence(start, end) {
    if (!dom.previewVideo.src) {
      toast("原始视频不可用，无法播放动作证据", "info");
      return;
    }
    const from = Number(start);
    const to = Number(end);
    if (!Number.isFinite(from) || !Number.isFinite(to) || from < 0 || to < from) return;
    clearCoachEvidenceStop();
    const video = dom.previewVideo;
    const stopAt = Math.max(from + 0.25, to);
    const stop = () => {
      video.removeEventListener("timeupdate", onTime);
      video.removeEventListener("ended", stop);
      if (state.coachEvidenceStop === stop) state.coachEvidenceStop = null;
    };
    const onTime = () => {
      if (video.currentTime >= stopAt - 0.02) {
        try { video.pause(); } catch (_) { /* no-op */ }
        stop();
      }
    };
    state.coachEvidenceStop = stop;
    video.addEventListener("timeupdate", onTime);
    video.addEventListener("ended", stop, { once: true });
    video.currentTime = from;
    video.play().catch(() => { /* Browser may require a direct video interaction. */ });
    dom.mediaShell.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function cleanCoachText(value, fallback = "") {
    let text = String(value == null || value === "" ? fallback : value).trim();
    if (!text) return "";
    // Older job manifests can still contain the research-oriented labels.
    // Normalize them at the browser boundary so every report reads like a
    // short product insight, even when it was generated by an older server.
    [
      ["技术候选待复核", "技术信息不足"],
      ["正反手待确认", "正反手信息不足"],
      ["低置信候选", "重点关注"],
      ["前场球路待确认", "前场球路信息不足"],
      ["后场球路待确认", "后场球路信息不足"],
      ["待复核", "回看视频"],
      ["待确认", "信息不足"],
      ["可能有助于", "帮助"],
      ["请结合视频确认", "请回看视频"],
      ["由教练确认", "建议教练复看"],
    ].forEach(([from, to]) => { text = text.split(from).join(to); });
    return text.replace(/\s{2,}/g, " ").trim();
  }

  function cleanMotionCoachText(value, fallback = "") {
    // “候选” is an internal confidence state for legacy coach records, not
    // useful card copy.  Remove it only inside Motion Coach; BST and Hawk-Eye
    // still use “候选” in their own technical/reporting meaning.
    return cleanCoachText(value, fallback)
      .replace(/设置持拍手可区分正反手/g, "设置持拍手")
      .replace(/候选/g, "")
      .replace(/\s{2,}/g, " ")
      .trim();
  }

  function coachFriendlyStroke(stroke) {
    const family = String((stroke && stroke.family) || "unknown");
    const labels = {
      serve: "发球",
      push: "前场推球",
      lift: "前场挑球",
      cross_net: "前场对角球",
      net: "前场放网",
      net_kill: "前场扑球",
      clear: "后场高远球",
      drive_clear: "后场平高球",
      drop: "后场吊球",
      smash: "后场杀球",
      defensive_drive: "后场平抽",
      preparation: "准备衔接",
      overhead: "上手球",
      frontcourt: "前场球",
      backcourt: "后场球",
      recovery: "击球恢复",
      front_unknown: "前场球路信息不足",
      back_unknown: "后场球路信息不足",
      unknown: "球路信息不足",
    };
    return cleanMotionCoachText(
      (stroke && (stroke.display_label || stroke.label)) || labels[family],
      "这一拍信息不足",
    );
  }

  const RELIABLE_COACH_FAMILIES = new Set([
    "serve", "push", "lift", "cross_net", "net", "net_kill",
    "clear", "drive_clear", "drop", "smash", "defensive_drive",
    "preparation", "overhead", "frontcourt", "backcourt", "recovery",
  ]);
  const RELIABLE_COACH_SIGNAL_FAMILIES = new Map([
    ["serve_preparation", new Set(["serve"])],
    ["serve_first_step", new Set(["serve"])],
    ["front_support", new Set(["push", "lift", "cross_net", "net", "net_kill", "frontcourt"])],
    ["front_contact_space", new Set(["push", "lift", "cross_net", "net", "net_kill", "frontcourt"])],
    ["overhead_extension", new Set(["clear", "drive_clear", "drop", "smash", "overhead"])],
    ["back_support", new Set(["clear", "drive_clear", "drop", "smash", "defensive_drive", "backcourt"])],
    ["preparation_support", new Set(["preparation"])],
    ["overhead_preparation", new Set(["overhead"])],
    ["recovery_stability", new Set(["recovery"])],
  ]);
  const UNCERTAIN_COACH_COPY = /待确认|待复核|信息不足|不确定|疑似|可能|或许|候选|需要确认|建议复核|请复核/;

  function reliableCoachRecommendations(player) {
    const recommendations = player && Array.isArray(player.recommendations) ? player.recommendations : [];
    return recommendations.flatMap((item) => {
      if (!item || typeof item !== "object" || item.status !== "reliable") return [];
      const family = String(item.family || "");
      const signal = String(item.signal || "");
      const signalFamilies = RELIABLE_COACH_SIGNAL_FAMILIES.get(signal);
      const repeatCount = Math.floor(Number(item.repeat_count) || 0);
      const problem = String(item.problem || "").trim();
      const action = String(item.action || "").trim();
      // A reliable marker is necessary, but incomplete or hesitant copy is
      // still not useful to an athlete. Reject the whole card instead of
      // rewriting uncertainty into false confidence in the browser.
      if (!RELIABLE_COACH_FAMILIES.has(family) || !signalFamilies || !signalFamilies.has(family) || repeatCount < 3 || !problem || !action || UNCERTAIN_COACH_COPY.test(`${problem} ${action}`)) return [];
      return [{ ...item, family, signal, repeat_count: repeatCount, problem, action }];
    });
  }

  function coachEvidenceButtons(recommendation) {
    const evidence = Array.isArray(recommendation.evidence) ? recommendation.evidence : [];
    return evidence.slice(0, 3).map((point) => {
      if (!point || typeof point !== "object") return "";
      const seconds = Number(point.seconds);
      const timecode = String(point.timecode || "");
      const rawStart = Number(point.window_start);
      const rawEnd = Number(point.window_end);
      const start = Number.isFinite(rawStart) ? rawStart : seconds - .45;
      const end = Number.isFinite(rawEnd) ? rawEnd : seconds + .75;
      if (!Number.isFinite(seconds) || seconds < 0 || !/^\d{2,3}:\d{2}$/.test(timecode)) return "";
      return `<button class="coach-evidence-play" type="button" data-coach-start="${escapeHTML(String(Math.max(0, start)))}" data-coach-end="${escapeHTML(String(Math.max(start, end)))}">${escapeHTML(timecode)} ↗</button>`;
    }).join("");
  }

  function reliableCoachCard(recommendation) {
    const familyLabel = coachFriendlyStroke({ family: recommendation.family });
    const repeatCount = recommendation.repeat_count;
    const evidenceButtons = coachEvidenceButtons(recommendation);
    const basis = `依据 ${formatNumber(repeatCount)} 拍`;
    return `<article class="coach-recommendation-card"><header><strong>${escapeHTML(familyLabel)}</strong><small>${escapeHTML(basis)}</small></header><div class="coach-recommendation-row"><i>问题</i><p>${escapeHTML(recommendation.problem)}</p></div><div class="coach-recommendation-row training"><i>训练</i><p>${escapeHTML(recommendation.action)}</p></div>${evidenceButtons ? `<footer><span>证据</span><div>${evidenceButtons}</div></footer>` : ""}</article>`;
  }

  function renderCoaching(raw) {
    const report = raw && typeof raw === "object" ? raw : {};
    const players = Array.isArray(report.players) ? report.players.filter((player) => player && typeof player === "object") : [];
    const status = String(report.status || "");
    dom.coachingGrid.replaceChildren();
    const reliablePlayers = players.map((player) => ({
      player,
      recommendations: reliableCoachRecommendations(player),
    })).filter((entry) => entry.recommendations.length);
    const reliableCount = reliablePlayers.reduce((total, entry) => total + entry.recommendations.length, 0);
    dom.coachingNotice.textContent = `可靠建议 ${formatNumber(reliableCount)} 条${reliableCount ? " · 点击时间码回看证据" : ""}`;
    dom.coachingSection.hidden = !status;
    if (!status) return;
    if (status !== "ready" || !reliableCount) {
      dom.coachingGrid.innerHTML = '<div class="coaching-empty">可靠建议 0 条</div>';
      return;
    }
    dom.coachingGrid.innerHTML = reliablePlayers.map(({ player, recommendations }) => {
      const cards = recommendations.map(reliableCoachCard).join("");
      return `<article class="coach-player coach-player-reliable"><header><div><span>${escapeHTML(player.label || "球员")}</span><small>仅展示通过多拍证据门控的建议</small></div><b>${formatNumber(recommendations.length)}<em>条</em></b></header><div class="coach-recommendation-list">${cards}</div></article>`;
    }).join("");
    $$('[data-coach-start][data-coach-end]', dom.coachingGrid).forEach((button) => button.addEventListener("click", () => {
      playCoachEvidence(button.dataset.coachStart, button.dataset.coachEnd);
    }));
  }

  function renderStrokeTypes(raw) {
    const report = raw && typeof raw === "object" ? raw : {};
    const status = String(report.status || "");
    const hits = Array.isArray(report.hits) ? report.hits.filter((item) => item && typeof item === "object") : [];
    dom.strokeTypesGrid.replaceChildren();
    dom.strokeTypesSummary.replaceChildren();
    if (!status) {
      dom.strokeTypesSection.hidden = true;
      return;
    }
    dom.strokeTypesSection.hidden = false;
    dom.strokeTypesNotice.textContent = cleanCoachText(report.quality_notice || report.notice, "BST 只识别击球类型，不评价动作质量");
    const total = Math.max(0, Number(report.total_hit_count) || 0);
    const predicted = Math.max(0, Math.min(total || Number.MAX_SAFE_INTEGER, Number(report.predicted_hit_count) || hits.length));
    const model = String(report.model || "BST");
    const dataset = String(report.dataset || "—");
    dom.strokeTypesSummary.innerHTML = `<span>${escapeHTML(model)} · ${escapeHTML(dataset)}</span><b>${formatNumber(predicted)} / ${formatNumber(total)} 拍</b><i>类型置信度 ≠ 动作质量</i>`;
    if (status !== "ready" || !hits.length) {
      const message = cleanCoachText(report.notice, status === "not_configured" ? "未配置 BST 权重，将显示规则型结果。" : "信息不足，暂时没有逐拍模型结果。");
      dom.strokeTypesGrid.innerHTML = `<div class="stroke-types-empty ${escapeHTML(status)}">${escapeHTML(message)}</div>`;
      return;
    }
    const hitterLabel = (value) => value === "near" ? "下方球员" : value === "far" ? "上方球员" : "击球方信息不足";
    dom.strokeTypesGrid.innerHTML = hits.map((hit) => {
      const confidence = Math.max(0, Math.min(100, Number(hit.confidence) || 0));
      const level = hit.confidence_level === "ready" ? "ready" : "review";
      const top3 = Array.isArray(hit.top3) ? hit.top3.slice(0, 3) : [];
      const topRows = top3.map((candidate) => {
        const probability = Math.max(0, Math.min(100, Number(candidate.confidence) || Number(candidate.probability) * 100 || 0));
        return `<div class="stroke-type-prob"><span>${escapeHTML(cleanCoachText(candidate.label, "类型信息不足"))}</span><i><b style="width:${probability.toFixed(1)}%"></b></i><em>${formatNumber(probability, 1)}%</em></div>`;
      }).join("");
      const seconds = Number(hit.seconds);
      const start = Number(hit.window_start);
      const end = Number(hit.window_end);
      const timecode = String(hit.timecode || "");
      const canPlay = Number.isFinite(seconds) && Number.isFinite(start) && Number.isFinite(end) && start >= 0 && end >= start && /^\d{2,3}:\d{2}$/.test(timecode);
      const play = canPlay ? `<button class="coach-evidence-play" type="button" data-coach-start="${escapeHTML(String(start))}" data-coach-end="${escapeHTML(String(end))}">原视频 ${escapeHTML(timecode)} ↗</button>` : "";
      const rule = String(hit.rule_candidate || "");
      const sideWarning = hit.model_side && hit.hitter && hit.hitter !== "unknown" && hit.model_side !== hit.hitter
        ? `<small class="stroke-type-warning">模型侧别与事件归属不一致，请回看视频</small>` : "";
      const coverage = `${formatNumber(Math.max(0, Number(hit.pose_coverage) || 0) * 100, 0)}% 姿态 · ${formatNumber(Math.max(0, Number(hit.ball_coverage) || 0) * 100, 0)}% 球点`;
      return `<article class="stroke-type-card ${level}"><header><div><span>RALLY ${escapeHTML(String(hit.rally_id || 0).padStart(2, "0"))} · HIT ${escapeHTML(String(hit.hit_index || 0).padStart(2, "0"))}</span><strong>${escapeHTML(timecode || "未定位")} · ${escapeHTML(hitterLabel(hit.hitter))}</strong></div><b>${formatNumber(confidence, 1)}<em>%</em></b></header><div class="stroke-type-main"><strong>${escapeHTML(cleanCoachText(hit.model_label, "类型信息不足"))}</strong><span>${level === "ready" ? "模型结果" : "人工查看"}</span></div>${topRows ? `<div class="stroke-type-top3">${topRows}</div>` : ""}${rule ? `<div class="stroke-type-rule"><i>规则结果</i><span>${escapeHTML(cleanCoachText(rule))}</span></div>` : ""}${sideWarning}<footer><small>${escapeHTML(coverage)}${hit.review_reason ? ` · ${escapeHTML(cleanCoachText(hit.review_reason))}` : ""}</small>${play}</footer></article>`;
    }).join("");
    $$('[data-coach-start][data-coach-end]', dom.strokeTypesGrid).forEach((button) => button.addEventListener("click", () => {
      playCoachEvidence(button.dataset.coachStart, button.dataset.coachEnd);
    }));
  }

  function hawkeyeNumber(value, fallback, minimum, maximum) {
    const number = Number(value);
    if (!Number.isFinite(number)) return fallback;
    return Math.max(minimum, Math.min(maximum, number));
  }

  function hawkeyeEmptyText(report) {
    const fallback = cleanCoachText(report.notice, "连续轨迹信息不足，暂时没有落点结果。");
    const diagnostics = report.diagnostics && typeof report.diagnostics === "object" ? report.diagnostics : {};
    const reasons = Array.isArray(diagnostics.reason_counts) ? diagnostics.reason_counts : [];
    const primary = reasons.find((item) => item && typeof item === "object" && Object.prototype.hasOwnProperty.call(HAWKEYE_REASON_LABELS, String(item.code || "")));
    if (!primary) return fallback;
    return `未生成落点：${HAWKEYE_REASON_LABELS[String(primary.code)]}。`;
  }

  function hawkeyeCourt(candidate, court) {
    const width = hawkeyeNumber(court && court.width_m, 6.1, 5.5, 7.0);
    const length = hawkeyeNumber(court && court.length_m, 13.4, 12.0, 15.0);
    const landing = candidate && typeof candidate.landing === "object" ? candidate.landing : {};
    const x = hawkeyeNumber(landing.x_m, width / 2, -3, width + 3);
    const y = hawkeyeNumber(landing.y_m, length / 2, -3, length + 3);
    const markerX = Math.max(-0.62, Math.min(width + 0.62, x)).toFixed(3);
    const markerY = Math.max(-0.62, Math.min(length + 0.62, y)).toFixed(3);
    const decision = ["in", "out", "review"].includes(candidate.decision) ? candidate.decision : "review";
    const viewWidth = (width + 1.6).toFixed(2);
    const viewHeight = (length + 1.6).toFixed(2);
    const boundary = String(candidate && candidate.boundary || "");
    const boundaryLine = {
      far_baseline: `<line class="hawk-boundary out" x1="0" y1="0" x2="${width.toFixed(2)}" y2="0"/>`,
      near_baseline: `<line class="hawk-boundary out" x1="0" y1="${length.toFixed(2)}" x2="${width.toFixed(2)}" y2="${length.toFixed(2)}"/>`,
      left_sideline: `<line class="hawk-boundary out" x1="0" y1="0" x2="0" y2="${length.toFixed(2)}"/>`,
      right_sideline: `<line class="hawk-boundary out" x1="${width.toFixed(2)}" y1="0" x2="${width.toFixed(2)}" y2="${length.toFixed(2)}"/>`,
    }[boundary] || "";
    const recovered = candidate && candidate.method === "visible_terminal_rest_after_occlusion";
    return `<svg class="hawk-court" viewBox="-0.8 -0.8 ${viewWidth} ${viewHeight}" role="img" aria-label="${recovered ? "遮挡后恢复的底线出界示意" : "双打球场落点图"}"><rect class="hawk-court-fill" x="0" y="0" width="${width.toFixed(2)}" height="${length.toFixed(2)}" rx=".10"/><rect class="hawk-court-line" x="0" y="0" width="${width.toFixed(2)}" height="${length.toFixed(2)}" rx=".10"/><line class="hawk-net" x1="0" y1="${(length / 2).toFixed(2)}" x2="${width.toFixed(2)}" y2="${(length / 2).toFixed(2)}"/><line class="hawk-center" x1="${(width / 2).toFixed(2)}" y1="0" x2="${(width / 2).toFixed(2)}" y2="${length.toFixed(2)}"/>${boundaryLine}<circle class="hawk-marker ${decision}" cx="${markerX}" cy="${markerY}" r=".23"/><circle class="hawk-marker-core ${decision}" cx="${markerX}" cy="${markerY}" r=".085"/></svg>`;
  }

  function renderHawkeye(raw) {
    const report = raw && typeof raw === "object" ? raw : {};
    const status = String(report.status || "");
    const candidates = Array.isArray(report.candidates) ? report.candidates.filter((item) => item && typeof item === "object") : [];
    const summary = report.summary && typeof report.summary === "object" ? report.summary : {};
    const court = report.court && typeof report.court === "object" ? report.court : {};
    dom.hawkeyeGrid.replaceChildren();
    dom.hawkeyeSummary.replaceChildren();
    dom.hawkeyeNotice.textContent = cleanCoachText(report.notice, "单机位终局轨迹参考 · 请以原视频为准");
    if (!status) {
      dom.hawkeyeSection.hidden = true;
      return;
    }
    dom.hawkeyeSection.hidden = false;
    if (status !== "ready" || !candidates.length) {
      dom.hawkeyeGrid.innerHTML = `<div class="hawkeye-empty">${escapeHTML(hawkeyeEmptyText(report))}</div>`;
      return;
    }
    const safeCount = (key) => Math.max(0, Math.min(999, Number(summary[key]) || 0));
    dom.hawkeyeSummary.innerHTML = `<span>${formatNumber(safeCount("candidate_count"))} POINTS</span><b class="in">${formatNumber(safeCount("in_count"))} IN</b><b class="out">${formatNumber(safeCount("out_count"))} OUT</b><b class="review">${formatNumber(safeCount("review_count"))} CHECK</b><i>双打全场 · 线边 ${formatNumber(hawkeyeNumber(court.line_margin_cm, 20, 1, 100))} cm 内请看视频；遮挡后恢复的 OUT 不显示单点精度</i>`;
    dom.hawkeyeGrid.innerHTML = candidates.map((candidate) => {
      const decision = ["in", "out", "review"].includes(candidate.decision) ? candidate.decision : "review";
      const labels = { in: "IN", out: "OUT", review: "人工查看" };
      const recoveredLanding = candidate.method === "visible_terminal_rest_after_occlusion";
      const measuredLanding = candidate.method === "visible_rest" || recoveredLanding;
      const method = recoveredLanding ? "遮挡后落地已恢复" : measuredLanding ? "落地已观测" : "未确认落地";
      const hitter = candidate.hitter === "near" ? "下方球员最后一拍" : candidate.hitter === "far" ? "上方球员最后一拍" : "击球方信息不足";
      const confidence = Math.max(0, Math.min(100, Number(candidate.confidence) || 0));
      const seconds = Number(candidate.seconds);
      const timecode = String(candidate.timecode || "");
      const lineDistance = hawkeyeNumber(candidate.line_distance_cm, 0, 0, 1000);
      // Only a measured landing event may display a line distance.  This
      // guard also protects browsers that still hold an older cached report.
      const rawRange = Array.isArray(candidate.line_distance_range_cm) ? candidate.line_distance_range_cm : [];
      const rangeMin = hawkeyeNumber(rawRange[0], lineDistance, 0, 1000);
      const rangeMax = hawkeyeNumber(rawRange[1], lineDistance, rangeMin, 1000);
      const boundaryLabels = { far_baseline: "上方底线", near_baseline: "下方底线", left_sideline: "左侧边线", right_sideline: "右侧边线" };
      const boundaryLabel = boundaryLabels[String(candidate.boundary || "")] || "边线";
      const lineDistanceCopy = recoveredLanding
        ? `${boundaryLabel}外约 ${formatNumber(rangeMin, 0)}–${formatNumber(rangeMax, 0)}cm`
        : measuredLanding
          ? `距最近边线 ${formatNumber(lineDistance, 1)}cm`
          : "未生成可量化距离";
      const evidenceFrames = Math.max(0, Math.min(99, Number(candidate.evidence_frames) || 0));
      const landing = candidate.landing && typeof candidate.landing === "object" ? candidate.landing : {};
      const x = hawkeyeNumber(landing.x_m, 0, -3, 9.1);
      const y = hawkeyeNumber(landing.y_m, 0, -3, 16.4);
      const canSeek = Number.isFinite(seconds) && seconds >= 0 && /^\d{2,3}:\d{2}$/.test(timecode);
      const seek = canSeek ? `<button class="hawk-seek" type="button" data-hawkeye-seconds="${escapeHTML(String(seconds))}" title="在原始视频定位 ${escapeHTML(timecode)}">原视频 ${escapeHTML(timecode)} ↗</button>` : "";
      const locationCopy = recoveredLanding
        ? `<strong>${escapeHTML(boundaryLabel)}外</strong> · ${escapeHTML(lineDistanceCopy)}`
        : `落点 <strong>${formatNumber(x, 2)}m × ${formatNumber(y, 2)}m</strong> · ${escapeHTML(lineDistanceCopy)}`;
      const displayLabel = recoveredLanding && decision === "out" ? "OUT · 轨迹" : labels[decision];
      return `<article class="hawk-card ${decision} ${recoveredLanding ? "recovered" : ""}">${hawkeyeCourt(candidate, court)}<div class="hawk-card-copy"><header><div><span>RALLY ${escapeHTML(String(candidate.rally_id || 0).padStart(2, "0"))}</span><small>${escapeHTML(hitter)} · ${escapeHTML(method)}</small></div><b>${escapeHTML(displayLabel)}</b></header><p>${locationCopy}</p><footer><span>${formatNumber(evidenceFrames)} 帧实测证据 · ${formatNumber(confidence)}% 证据质量</span>${seek}</footer></div></article>`;
    }).join("");
    $$('[data-hawkeye-seconds]', dom.hawkeyeGrid).forEach((button) => button.addEventListener("click", () => {
      const seconds = Number(button.dataset.hawkeyeSeconds);
      if (!Number.isFinite(seconds)) return;
      // Hawk-Eye timing belongs to the original tracking video.  The FX
      // analysis render can stretch/freeze frames, so seeking it would show
      // the wrong evidence instant.
      if (!dom.previewVideo.src) { toast("原始视频不可用，无法定位复核时间点", "info"); return; }
      dom.previewVideo.currentTime = seconds;
      dom.previewVideo.play().catch(() => { /* Browser may require a click. */ });
      dom.mediaShell.scrollIntoView({ behavior: "smooth", block: "center" });
    }));
  }

  function renderResults(job) {
    const status = job && safeJobStatus(job.status);
    if (!job || !TERMINAL.has(status)) return;
    dom.resultsPanel.hidden = false;
    setCompletePill(status);
    dom.statusLed.classList.remove("active", "done");
    dom.previewCaption.textContent = status === "completed" ? "ANALYSIS / READY" : `ANALYSIS / ${status.toUpperCase()}`;
    if (status !== "completed") {
      dom.kpiGrid.hidden = true;
      dom.analysisOutput.hidden = true;
      dom.coachingSection.hidden = true;
      dom.coachingGrid.replaceChildren();
      dom.strokeTypesSection.hidden = true;
      dom.strokeTypesSummary.replaceChildren();
      dom.strokeTypesGrid.replaceChildren();
      dom.hawkeyeSection.hidden = true;
      dom.hawkeyeGrid.replaceChildren();
      dom.hawkeyeSummary.replaceChildren();
      dom.analysisVideo.removeAttribute("src");
      dom.analysisVideo.load();
      if (dom.analysisTitle) dom.analysisTitle.textContent = "专业分析视频";
      if (dom.analysisCaption) dom.analysisCaption.textContent = "含击球、姿态与移动视觉叠加";
      dom.rallySection.hidden = true;
      dom.rallyGrid.replaceChildren();
      dom.sidecarLink.hidden = true;
      dom.resultEmpty.hidden = false;
      dom.resultEmpty.classList.add("failure");
      dom.resultEmpty.textContent = status === "failed"
        ? `任务未完成：${job.error || "分析引擎返回了错误，请查看任务日志后重试。"}`
        : "任务已取消。已生成的临时文件不会作为正式结果展示。";
      return;
    }

    dom.kpiGrid.hidden = false;
    dom.resultEmpty.classList.remove("failure");
    const stats = job.stats && typeof job.stats === "object" ? job.stats : {};
    const config = job.config && typeof job.config === "object" ? job.config : {};
    const rallies = Array.isArray(job.rallies) ? job.rallies.filter((item) => item && typeof item === "object") : [];
    const duration = stats.duration_seconds || (state.video && state.video.metadata && state.video.metadata.duration);
    const exportedRallyCount = config.split_rallies ? rallies.length : stats.rally_count;
    dom.kpiGrid.innerHTML = [
      kpi("分析时长", formatDuration(duration), "", ""),
      kpi(config.split_rallies ? "导出回合" : "轨迹回合", formatNumber(exportedRallyCount), "RALLIES", "accent"),
      kpi("检测击球", formatNumber(stats.hit_count), "HITS", "violet"),
      kpi("峰值球速", stats.max_shot_speed_kmh == null ? "—" : formatNumber(stats.max_shot_speed_kmh, 1), "KM/H", "lime"),
      kpi("可见覆盖", stats.visible_coverage == null ? "—" : formatNumber(stats.visible_coverage * 100, 1), "%", ""),
    ].join("");
    const outputs = job.outputs && typeof job.outputs === "object" ? job.outputs : {};
    const analysisURL = outputs.analysis && safeMediaURL(outputs.analysis.url);
    const bulletTime = bulletConfigEnabled(config);
    if (dom.analysisTitle) dom.analysisTitle.textContent = bulletTime
      ? "子弹镜头专业分析视频"
      : "普通速度专业分析视频";
    if (dom.analysisCaption) dom.analysisCaption.textContent = bulletTime
      ? "含击球、姿态与移动视觉叠加 · 已加入冻结与慢动作"
      : "含击球、姿态与移动视觉叠加";
    if (analysisURL) {
      dom.analysisOutput.hidden = false;
      dom.analysisVideo.src = analysisURL;
      dom.analysisVideo.load();
    } else {
      dom.analysisOutput.hidden = true;
      dom.analysisVideo.removeAttribute("src");
      dom.analysisVideo.load();
    }
    renderHawkeye(job.hawkeye);
    renderCoaching(job.coaching);
    renderStrokeTypes(job.stroke_types);
    const sidecarURL = outputs.sidecar && safeMediaURL(outputs.sidecar.url);
    if (sidecarURL) {
      dom.sidecarLink.hidden = false;
      dom.sidecarLink.href = sidecarURL;
      const lightweight = Boolean(config.split_rallies) && !Boolean(config.professional_analysis);
      dom.sidecarLabel.textContent = lightweight ? "下载轻量回合 CSV" : "下载逐帧 CSV";
      dom.sidecarLink.title = lightweight
        ? "轻量模式用于回合索引；部分 Shot* 事件字段不会像专业模式一样完整。"
        : "下载逐帧分析数据";
    } else dom.sidecarLink.hidden = true;
    dom.rallySection.hidden = rallies.length === 0;
    if (rallies.length) {
      const detected = Number.isFinite(Number(stats.rally_count)) ? Number(stats.rally_count) : rallies.length;
      dom.rallySummary.textContent = `检测到 ${detected} 个回合 · 已导出 ${rallies.length} 段`;
    } else dom.rallySummary.textContent = "智能回合 · 可下载微调";
    dom.rallyGrid.innerHTML = rallies.map((rally) => {
      const quality = QUALITY_LABELS[rally.quality] || rally.quality || "回合";
      const fallback = rally.quality === "trajectory_fallback";
      const durationText = formatDuration(rally.duration_seconds);
      const url = safeMediaURL(rally.url);
      const player = url ? `<video src="${escapeHTML(url)}" preload="none" controls playsinline></video>` : "";
      const actions = url ? `<div class="rally-card-actions"><a href="${escapeHTML(url)}" download>下载短片 ↓</a><a href="${escapeHTML(url)}" target="_blank" rel="noopener">新窗口 ↗</a></div>` : "";
      return `<article class="rally-card">${player}<div class="rally-card-body"><div class="rally-card-title"><strong>RALLY ${escapeHTML(String(rally.index || 0).padStart(2, "0"))}</strong><span class="quality-tag ${fallback ? "fallback" : ""}">${escapeHTML(quality)}</span></div><div class="rally-meta"><span>${durationText}</span><span>${formatNumber(rally.hit_count)} HITS</span></div>${actions}</div></article>`;
    }).join("");
    const onlyPreview = !analysisURL && !sidecarURL && !rallies.length;
    dom.resultEmpty.hidden = !onlyPreview;
    if (onlyPreview) dom.resultEmpty.textContent = "本次只完成了视频导入与预览。打开任一分析开关后，可生成击球与回合报告。";
    dom.statusLed.classList.add("done");
  }

  function clearJobPoll() {
    if (state.pollTimer) window.clearTimeout(state.pollTimer);
    state.pollTimer = null;
  }

  function consumeJob(job, expectedJobId = null, expectedContextToken = null) {
    if (
      !job
      || !job.id
      || (expectedJobId && job.id !== expectedJobId)
      || (expectedContextToken !== null && expectedContextToken !== state.jobContextToken)
    ) return false;
    if (state.job && state.job.id && state.job.id !== job.id && hasQueuedOrRunningJob()) return false;
    if (state.job && state.job.id === job.id) {
      const currentStatus = safeJobStatus(state.job.status);
      const incomingStatus = safeJobStatus(job.status);
      // A delayed polling response must never turn a completed/failed/cancelled
      // task back into running after SSE has already delivered its terminal
      // snapshot.  ISO timestamps are sortable for non-terminal updates.
      if (TERMINAL.has(currentStatus) && !TERMINAL.has(incomingStatus)) return false;
      const currentUpdated = String(state.job.updated_at || "");
      const incomingUpdated = String(job.updated_at || "");
      if (currentUpdated && incomingUpdated && incomingUpdated < currentUpdated) return false;
    }
    state.job = job;
    renderProgress(job);
    updateCalibrationUI();
    const status = safeJobStatus(job.status);
    if (TERMINAL.has(status)) {
      if (state.eventSource) { state.eventSource.close(); state.eventSource = null; }
      state.eventSourceJobId = null;
      clearJobPoll();
      renderResults(job);
      loadHistory();
      const noticeKey = `${job.id}:${status}`;
      if (state.terminalNoticeFor !== noticeKey) {
        state.terminalNoticeFor = noticeKey;
        if (status === "completed") {
          const config = job.config && typeof job.config === "object" ? job.config : {};
          toast(
            config.split_rallies || config.professional_analysis
              ? "分析完成，比赛报告已生成"
              : "视频已就绪，仅完成导入与预览",
            "success",
          );
        }
        else if (status === "failed") toast(job.error || "处理失败", "error", 7000);
        else toast("任务已取消", "info");
      }
    }
    return true;
  }

  function pollJob(jobId, contextToken = state.jobContextToken, initialDelay = 2000) {
    clearJobPoll();
    const tick = async () => {
      if (
        contextToken !== state.jobContextToken
        || !state.job
        || state.job.id !== jobId
        || TERMINAL.has(safeJobStatus(state.job.status))
      ) return;
      try {
        const job = await requestJSON(`/api/jobs/${encodeURIComponent(jobId)}`);
        consumeJob(job, jobId, contextToken);
      } catch (_) { /* SSE will reconnect when possible; keep polling lightly. */ }
      if (
        contextToken === state.jobContextToken
        && state.job
        && state.job.id === jobId
        && !TERMINAL.has(safeJobStatus(state.job.status))
      ) {
        state.pollTimer = window.setTimeout(tick, 8000);
      }
    };
    state.pollTimer = window.setTimeout(tick, initialDelay);
  }

  function subscribeJob(jobId) {
    if (state.eventSource) state.eventSource.close();
    clearJobPoll();
    const source = new EventSource(`/api/jobs/${encodeURIComponent(jobId)}/events`);
    state.eventSource = source;
    state.eventSourceJobId = jobId;
    const contextToken = state.jobContextToken;
    source.onmessage = (event) => {
      try {
        const job = JSON.parse(event.data);
        if (job && job.id === jobId) consumeJob(job, jobId, contextToken);
      } catch (_) { /* ignore malformed event */ }
    };
    source.onerror = () => {
      if (
        contextToken !== state.jobContextToken
        || state.eventSource !== source
        || state.eventSourceJobId !== jobId
      ) return;
      if (state.job && state.job.id === jobId && TERMINAL.has(safeJobStatus(state.job.status))) {
        source.close();
      } else if (source.readyState === EventSource.CLOSED) {
        source.close();
        pollJob(jobId, contextToken, 2000);
      }
    };
    // Keep a sparse status watchdog even while SSE is healthy: proxies can
    // leave EventSource in reconnecting state without ever closing it.
    pollJob(jobId, contextToken, 10000);
  }

  async function startJob() {
    if (hasActiveJob()) { toast("已有任务正在处理", "error"); return; }
    if (!state.video) { toast("请先导入视频", "error"); return; }
    const split = dom.splitToggle.checked;
    const hawkeye = dom.hawkeyeToggle.checked;
    const bulletTime = bulletTimeSelected();
    const professional = professionalSelected();
    const continueAfterSceneCuts = (split || hawkeye || professional) && dom.sceneCutToggle.checked;
    const unavailable = missingCapabilities();
    if (unavailable.length) {
      toast(`分析引擎不可用：${unavailable.join("、")}`, "error");
      return;
    }
    if ((split || hawkeye || professional) && state.points.length !== 4) {
      toast("请先完成球场四点标定，或点击“使用建议点”", "error"); return;
    }
    const submissionContextToken = invalidateJobContext();
    state.submitting = true;
    updateCalibrationUI();
    dom.start.disabled = true;
    dom.start.querySelector(".cta-label").textContent = "正在排队…";
    dom.progressPanel.hidden = false;
    resetResultMedia({ hideResults: true });
    try {
      const points = state.points.reduce((all, point) => all.concat([Math.round(point.x), Math.round(point.y)]), []);
      const job = await requestJSON("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ video_id: state.video.id, split_rallies: split, hawkeye_review: hawkeye, professional_analysis: professional, bullet_time_enabled: bulletTime, continue_after_scene_cuts: continueAfterSceneCuts, court_points: points, device: dom.deviceSelect.value, coach_target: dom.coachTarget.value, near_dominant_hand: dom.nearDominantHand.value, far_dominant_hand: dom.farDominantHand.value }),
      });
      state.submitting = false;
      state.terminalNoticeFor = null;
      const accepted = consumeJob(job, job.id, submissionContextToken);
      if (accepted && !TERMINAL.has(safeJobStatus(job.status))) subscribeJob(job.id);
      else if (accepted) renderResults(job);
    } catch (error) {
      toast(error.message, "error", 7000);
      dom.progressPanel.hidden = true;
    } finally {
      state.submitting = false;
      dom.start.querySelector(".cta-label").textContent = "开始处理";
      updateCalibrationUI();
    }
  }

  async function cancelJob() {
    if (!state.job || !state.job.id || TERMINAL.has(safeJobStatus(state.job.status))) return;
    const requestedJobId = state.job.id;
    const contextToken = state.jobContextToken;
    dom.cancel.disabled = true;
    try {
      const job = await requestJSON(`/api/jobs/${encodeURIComponent(requestedJobId)}/cancel`, { method: "POST" });
      if (state.job && state.job.id === requestedJobId) {
        consumeJob(job, requestedJobId, contextToken);
      }
      toast("已发送取消请求", "info");
    }
    catch (error) { toast(error.message, "error"); }
    finally { dom.cancel.disabled = false; }
  }

  function pointListFromConfig(config) {
    const values = config && Array.isArray(config.court_points) ? config.court_points : [];
    if (values.length !== 8 || values.some((value) => !Number.isFinite(Number(value)))) return [];
    return ["TL", "TR", "BR", "BL"].map((label, index) => ({
      label,
      x: Number(values[index * 2]),
      y: Number(values[index * 2 + 1]),
    }));
  }

  function restoreJob(job) {
    if (!job || !job.id || hasActiveJob()) return;
    const config = job.config && typeof job.config === "object" ? job.config : {};
    const storedVideo = job.video && typeof job.video === "object" ? job.video : {};
    const video = state.history.find((item) => item.id === job.video_id) || (job.video_id ? {
      id: job.video_id,
      name: storedVideo.name || "原始视频",
      metadata: storedVideo.metadata || {},
      uploaded: true,
      content_url: `/api/videos/${encodeURIComponent(job.video_id)}/content`,
      poster_url: `/api/videos/${encodeURIComponent(job.video_id)}/poster`,
    } : null);
    if (video) {
      selectVideo(video, false);
      state.points = pointListFromConfig(config);
      drawCourt();
    } else clearVideo();
    dom.splitToggle.checked = Boolean(config.split_rallies);
    dom.hawkeyeToggle.checked = Boolean(config.hawkeye_review);
    dom.proToggle.checked = Boolean(config.professional_analysis) || bulletConfigEnabled(config);
    setControlChecked(dom.bulletTimeToggle, bulletConfigEnabled(config));
    if (["near", "far", "both"].includes(config.coach_target)) dom.coachTarget.value = config.coach_target;
    if (["auto", "left", "right"].includes(config.near_dominant_hand)) dom.nearDominantHand.value = config.near_dominant_hand;
    if (["auto", "left", "right"].includes(config.far_dominant_hand)) dom.farDominantHand.value = config.far_dominant_hand;
    dom.sceneCutToggle.checked = Boolean(config.continue_after_scene_cuts) && (
      Boolean(config.split_rallies) || Boolean(config.hawkeye_review) || Boolean(config.professional_analysis)
    );
    if (["auto", "cuda", "mps", "cpu"].includes(config.device)) dom.deviceSelect.value = config.device;
    updateModeExplainer();
    consumeJob(job, job.id);
    if (!TERMINAL.has(safeJobStatus(job.status))) subscribeJob(job.id);
  }

  function renderHistory(videos, jobs = []) {
    const safeVideos = (Array.isArray(videos) ? videos : []).filter(
      (video) => video && typeof video === "object",
    );
    const safeJobs = Array.isArray(jobs) ? jobs : [];
    const videosById = new Map(safeVideos.map((video) => [video.id, video]));
    const jobCards = safeJobs.slice(0, 6).map((job) => {
      if (!job || !job.id) return "";
      const video = videosById.get(job.video_id);
      const status = safeJobStatus(job.status);
      const config = job.config && typeof job.config === "object" ? job.config : {};
      const modeParts = [];
      if (config.professional_analysis) modeParts.push("专业分析");
      if (config.split_rallies) modeParts.push("智能回合切割");
      if (config.hawkeye_review) modeParts.push("鹰眼复核");
      if (bulletConfigEnabled(config)) modeParts.push("子弹镜头");
      const baseMode = modeParts.length ? modeParts.join(" + ") : "仅预览";
      const mode = config.continue_after_scene_cuts ? `${baseMode} · 跨镜头` : baseMode;
      const poster = video && safeMediaURL(video.poster_url);
      const thumb = poster
        ? `<img src="${escapeHTML(poster)}" alt="">`
        : `<span class="history-job-thumb">${status === "running" ? "RUN" : "JOB"}</span>`;
      return `<div class="history-item history-job" data-job-id="${escapeHTML(job.id)}">${thumb}<div class="history-item-main"><strong>${escapeHTML((job.video && job.video.name) || (video && video.name) || "分析任务")}</strong><span>${escapeHTML(mode)} · ${escapeHTML(String(job.stage || "等待任务"))}</span></div><em class="history-status ${status}">${escapeHTML(status.toUpperCase())}</em></div>`;
    }).filter(Boolean);
    const videoCards = safeVideos.slice(0, 7).map((video) => `<div class="history-item" data-video-id="${escapeHTML(video.id)}"><img src="${escapeHTML(safeMediaURL(video.poster_url))}" alt=""><div class="history-item-main"><strong>${escapeHTML(video.name || "未命名视频")}</strong><span>${escapeHTML(metadataOf(video))}</span></div><em>预览 ↗</em></div>`);
    const cards = [...jobCards, ...videoCards];
    if (!cards.length) { dom.historyList.innerHTML = '<div class="history-placeholder">还没有处理记录</div>'; return; }
    dom.historyList.innerHTML = cards.join("");
    $$("[data-video-id]", dom.historyList).forEach((item) => item.addEventListener("click", () => {
      const video = safeVideos.find((candidate) => candidate.id === item.dataset.videoId);
      if (video) { selectVideo(video); window.scrollTo({ top: 0, behavior: "smooth" }); }
    }));
    $$("[data-job-id]", dom.historyList).forEach((item) => item.addEventListener("click", () => {
      const job = safeJobs.find((candidate) => candidate.id === item.dataset.jobId);
      if (!job) return;
      if (hasActiveJob() && (!state.job || state.job.id !== job.id)) {
        toast("当前已有任务正在处理，请先等待完成或取消", "error");
        return;
      }
      restoreJob(job);
      window.scrollTo({ top: 0, behavior: "smooth" });
    }));
  }

  async function loadLibrary() {
    const videos = await requestJSON("/api/videos");
    state.history = Array.isArray(videos) ? videos : [];
    renderHistory(state.history, state.jobs);
    return state.history;
  }

  async function loadJobs() {
    const jobs = await requestJSON("/api/jobs");
    state.jobs = Array.isArray(jobs) ? jobs : [];
    renderHistory(state.history, state.jobs);
    return state.jobs;
  }

  async function loadHistory({ resumeActive = false } = {}) {
    try {
      await Promise.all([loadLibrary(), loadJobs()]);
      if (resumeActive && !hasActiveJob()) {
        const active = state.jobs.find((job) => job && safeJobStatus(job.status) === "running")
          || state.jobs.find((job) => job && !TERMINAL.has(safeJobStatus(job.status)));
        if (active) restoreJob(active);
      }
    } catch (error) {
      setSystemStatus("error", "LIBRARY OFFLINE");
    }
  }

  async function loadHealth() {
    try {
      const health = await requestJSON("/api/health");
      state.health = health && typeof health === "object" ? health : null;
      const weights = (state.health && state.health.weights) || {};
      if (weights.tracknet && state.health.ffmpeg) {
        const suffix = weights.pose ? "READY" : "TRACKNET READY · POSE MISSING";
        setSystemStatus("ready", `${state.health.accelerator || "ENGINE"} · ${suffix}`);
      } else {
        setSystemStatus("error", "ANALYSIS ENGINE CHECK REQUIRED");
      }
    } catch (error) {
      state.health = null;
      setSystemStatus("error", "ANALYSIS ENGINE OFFLINE");
    } finally {
      updateModeExplainer();
    }
  }

  function bind() {
    dom.dropZone.addEventListener("click", () => dom.fileInput.click());
    dom.dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); dom.fileInput.click(); } });
    ["dragenter", "dragover"].forEach((name) => dom.dropZone.addEventListener(name, (event) => { event.preventDefault(); dom.dropZone.classList.add("dragover"); }));
    ["dragleave", "drop"].forEach((name) => dom.dropZone.addEventListener(name, (event) => { event.preventDefault(); dom.dropZone.classList.remove("dragover"); }));
    dom.dropZone.addEventListener("drop", (event) => uploadFile(event.dataTransfer.files && event.dataTransfer.files[0]));
    dom.fileInput.addEventListener("change", () => { uploadFile(dom.fileInput.files && dom.fileInput.files[0]); dom.fileInput.value = ""; });
    dom.pathForm.addEventListener("submit", importPath);
    dom.clearSource.addEventListener("click", clearVideo);
    dom.splitToggle.addEventListener("change", updateModeExplainer);
    dom.hawkeyeToggle.addEventListener("change", updateModeExplainer);
    dom.proToggle.addEventListener("change", () => {
      if (!dom.proToggle.checked && bulletTimeSelected()) {
        setControlChecked(dom.bulletTimeToggle, false);
        toast("已关闭子弹镜头特效", "info");
      }
      updateModeExplainer();
    });
    if (dom.bulletTimeToggle) {
      dom.bulletTimeToggle.addEventListener("change", () => {
        if (bulletTimeSelected() && !dom.proToggle.checked) {
          dom.proToggle.checked = true;
          toast("子弹镜头需要专业分析，已自动开启专业模式", "info");
        }
        updateModeExplainer();
      });
    }
    dom.coachTarget.addEventListener("change", updateModeExplainer);
    dom.nearDominantHand.addEventListener("change", updateModeExplainer);
    dom.farDominantHand.addEventListener("change", updateModeExplainer);
    dom.sceneCutToggle.addEventListener("change", updateModeExplainer);
    dom.deviceSelect.addEventListener("change", updateModeExplainer);
    dom.defaultCourt.addEventListener("click", useDefaultCourt);
    dom.clearCourt.addEventListener("click", clearCourt);
    dom.courtCanvas.addEventListener("click", courtClick);
    dom.previewVideo.addEventListener("loadedmetadata", syncCanvas);
    dom.previewVideo.addEventListener("timeupdate", () => { dom.time.textContent = `${formatDuration(dom.previewVideo.currentTime)} / ${formatDuration(dom.previewVideo.duration)}`; });
    dom.previewVideo.addEventListener("error", () => { if (state.video) toast("浏览器无法直接播放此编码，可使用封面并继续分析", "error", 5500); });
    dom.start.addEventListener("click", startJob);
    dom.cancel.addEventListener("click", cancelJob);
    dom.refresh.addEventListener("click", () => { loadHealth(); loadHistory({ resumeActive: true }); });
    window.addEventListener("resize", () => { syncCanvas(); drawCourt(); });
  }

  bind();
  updateModeExplainer();
  loadHealth();
  loadHistory({ resumeActive: true });
}());
