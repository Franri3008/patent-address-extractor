// Patent Pipeline Dashboard — live updates via SSE, polling fallback

let lastComparison = null;

// ── Particle animation ───────────────────────────────────
const STAGE_ORDER = ['bq_fetch', 'pdf_worker', 'ocr_worker', 'llm_worker', 'output_writer'];
const STAGE_COLORS = {
  bq_fetch: '#7c3aed',
  pdf_worker: '#ea580c',
  ocr_worker: '#0891b2',
  llm_worker: '#1d6ef5',
  output_writer: '#16a34a',
};
const _stageStates = {};

function animateParticle(fromId, toId) {
  const fromEl = document.getElementById('stage-' + fromId);
  const toEl = document.getElementById('stage-' + toId);
  const particle = document.getElementById('particle');
  const pipeline = document.querySelector('.pipeline');
  if (!fromEl || !toEl || !particle || !pipeline) return;

  const pipelineRect = pipeline.getBoundingClientRect();
  const fromRect = fromEl.getBoundingClientRect();
  const toRect = toEl.getBoundingClientRect();

  const fromX = fromRect.left + fromRect.width / 2 - pipelineRect.left;
  const fromY = fromRect.top + fromRect.height / 2 - pipelineRect.top;
  const toX = toRect.left + toRect.width / 2 - pipelineRect.left;
  const toY = toRect.top + toRect.height / 2 - pipelineRect.top;

  const color = STAGE_COLORS[toId] || '#1d6ef5';
  particle.style.background = color;
  particle.style.boxShadow = `0 0 10px 3px ${color}88`;
  particle.style.transition = 'none';
  particle.style.transform = `translate(${fromX}px, ${fromY}px)`;
  particle.style.opacity = '1';

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      particle.style.transition = 'transform 0.55s cubic-bezier(0.4,0,0.2,1), opacity 0.25s ease 0.45s';
      particle.style.transform = `translate(${toX}px, ${toY}px)`;
      particle.style.opacity = '0';
    });
  });
}

function applyState(data) {
  updateHeader(data);
  updateProgress(data);
  updateStage('bq_fetch', data.stages.bq_fetch);
  updateStage('pdf_worker', data.stages.pdf_worker);
  updateStage('ocr_worker', data.stages.ocr_worker);
  updateStage('llm_worker', data.stages.llm_worker);
  updateStage('output_writer', data.stages.output_writer);
  updateMetrics(data);
  updateTimings(data.timings);
  updateLLMPanel(data.stages.llm_worker);

  if (data.latest_comparison && data.latest_comparison.patent_id) {
    lastComparison = data.latest_comparison;
    document.getElementById('btn-compare').disabled = false;
  }
}

// Polling path (file:// or SSE unavailable)
async function tick() {
  try {
    const res = await fetch('status.json?' + Date.now());
    if (!res.ok) return;
    applyState(await res.json());
  } catch {
    // file not ready yet
  }
}

// ── Header ──────────────────────────────────────────────

function updateHeader(data) {
  document.getElementById('run-mode').textContent = data.run_mode;
  document.getElementById('run-id').textContent = data.run_id;
  document.getElementById('elapsed').textContent = fmtTime(data.timings.total_elapsed_s);
}

// ── Progress bar ────────────────────────────────────────

function updateProgress(data) {
  const total = data.total_patents || 1;
  const done = data.stages.output_writer.completed || 0;
  const pct = Math.min(100, (done / total) * 100);
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-label').textContent = done + ' / ' + total;
}

// ── Stage cards ─────────────────────────────────────────

function updateStage(id, stage) {
  const el = document.getElementById('stage-' + id);
  const badge = document.getElementById('badge-' + id);

  const prev = _stageStates[id];
  _stageStates[id] = stage.status;

  // Animate particle when a stage becomes active (transitions to running)
  if (prev !== 'running' && stage.status === 'running') {
    const idx = STAGE_ORDER.indexOf(id);
    if (idx > 0) animateParticle(STAGE_ORDER[idx - 1], id);
  }

  el.classList.remove('active', 'done');
  if (stage.status === 'running') el.classList.add('active');
  else if (stage.status === 'done') el.classList.add('done');

  badge.textContent = stage.status;
}

// ── Per-stage metrics ───────────────────────────────────

function updateMetrics(data) {
  const s = data.stages;

  // BQ
  setText('bq-fetched', s.bq_fetch.patents_fetched);
  setText('bq-time', fmtSec(s.bq_fetch.elapsed_s));
  setText('fab-bq', s.bq_fetch.patents_fetched || '—');

  // PDF
  setText('pdf-completed', s.pdf_worker.completed);
  setText('pdf-errors', s.pdf_worker.errors);
  setText('pdf-type', s.pdf_worker.last_pdf_type || '—');
  setText('pdf-current', truncate(s.pdf_worker.current_patent, 22));
  setText('fab-pdf', s.pdf_worker.queue_size != null ? s.pdf_worker.queue_size : '—');

  // OCR
  setText('ocr-completed', s.ocr_worker.completed);
  setText('ocr-pages', s.ocr_worker.last_pages_used || '—');
  setText('ocr-sections', (s.ocr_worker.last_sections || []).join(' ') || '—');
  setText('ocr-current', truncate(s.ocr_worker.current_patent, 22));
  setText('fab-ocr', s.ocr_worker.queue_size != null ? s.ocr_worker.queue_size : '—');

  // LLM
  setText('llm-completed', s.llm_worker.completed);
  setText('llm-errors', s.llm_worker.errors);
  setText('llm-current', truncate(s.llm_worker.current_patent, 22));
  setText('fab-llm', s.llm_worker.queue_size != null ? s.llm_worker.queue_size : '—');

  const preview = s.llm_worker.last_result_preview || {};
  const found = preview.found;
  const foundStat = document.getElementById('stat-llm-found');
  if (found != null && foundStat) {
    setText('llm-found', found ? 'yes' : 'no');
    foundStat.className = 'stat ' + (found ? 'ok' : 'err');
  } else {
    setText('llm-found', '—');
  }

  // Output
  setText('out-successes', s.output_writer.successes);
  setText('out-failures', s.output_writer.failures);
  setText('fab-out', s.output_writer.successes != null ? s.output_writer.successes : '—');
}

// ── Timings ─────────────────────────────────────────────

function updateTimings(t) {
  setText('avg-pdf', fmtSec(t.avg_pdf_s));
  setText('avg-ocr', fmtSec(t.avg_ocr_s));
  setText('avg-llm', fmtSec(t.avg_llm_s));
  setText('total-time', fmtTime(t.total_elapsed_s));
}

// ── LLM response panel ─────────────────────────────────

function updateLLMPanel(llm) {
  const raw = llm.last_raw_response;
  const el = document.getElementById('llm-response');
  if (raw) {
    try {
      el.textContent = JSON.stringify(JSON.parse(raw), null, 2);
    } catch {
      el.textContent = raw;
    }
  }
}

// ── Comparison modal ────────────────────────────────────

document.getElementById('btn-compare').addEventListener('click', () => {
  if (lastComparison) openModal(lastComparison);
});
document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-overlay').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

function openModal(comp) {
  document.getElementById('modal-patent-id').textContent = comp.patent_id;

  const imgCol = document.getElementById('modal-images');
  imgCol.innerHTML = '<div class="col-label">PDF Pages</div>';
  (comp.page_images || []).forEach(src => {
    const img = document.createElement('img');
    img.src = src + '?' + Date.now();
    img.alt = 'Patent page';
    imgCol.appendChild(img);
  });

  document.getElementById('modal-json').textContent =
    JSON.stringify(comp.llm_result, null, 2);

  document.getElementById('modal-overlay').classList.add('open');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.remove('open');
}

// ── Helpers ─────────────────────────────────────────────

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val != null ? val : '—';
}

function fmtSec(s) {
  if (s == null || s === 0) return '—';
  return s.toFixed(2) + 's';
}

function fmtTime(s) {
  if (s == null || s === 0) return '0.0s';
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(0);
  return m + 'm ' + sec + 's';
}

function truncate(str, len) {
  if (!str) return '';
  return str.length > len ? str.slice(0, len) + '…' : str;
}

// ── Connect: SSE when served over HTTP, polling otherwise ──
if (location.protocol === 'http:' || location.protocol === 'https:') {
  const es = new EventSource('/api/status/stream');
  es.onmessage = (e) => {
    try { applyState(JSON.parse(e.data)); } catch {}
  };
  es.onerror = () => {
    // SSE failed (server not up yet?) — fall back to polling
    es.close();
    tick();
    setInterval(tick, 1000);
  };
} else {
  // Opened as a local file — use polling
  tick();
  setInterval(tick, 1000);
}
