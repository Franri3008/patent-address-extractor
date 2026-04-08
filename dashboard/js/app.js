// Patent Pipeline Dashboard — polls status.json every second

let lastComparison = null;

async function tick() {
  let data;
  try {
    const res = await fetch('status.json?' + Date.now());
    if (!res.ok) return;
    data = await res.json();
  } catch {
    return; // file not ready yet
  }

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

// ── Header ──────────────────────────────────────────────

function updateHeader(data) {
  document.getElementById('run-mode').textContent = data.run_mode;
  document.getElementById('pipeline-mode').textContent =
    'mode ' + data.pipeline_mode + (data.pipeline_mode === 0 ? ' (OCR+LLM)' : ' (Vision)');
  document.getElementById('run-id').textContent = data.run_id;
  document.getElementById('elapsed').textContent = fmtTime(data.timings.total_elapsed_s);
}

// ── Progress bar ────────────────────────────────────────

function updateProgress(data) {
  const total = data.total_patents || 1;
  const done = (data.stages.output_writer.completed || 0);
  const pct = Math.min(100, (done / total) * 100);
  document.getElementById('progress-bar').style.width = pct.toFixed(1) + '%';
  document.getElementById('progress-label').textContent = done + ' / ' + total;
}

// ── Stage cards ─────────────────────────────────────────

function updateStage(id, stage) {
  const el = document.getElementById('stage-' + id);
  const badge = document.getElementById('badge-' + id);

  el.classList.remove('active', 'done', 'error');
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

  // PDF
  setText('pdf-completed', s.pdf_worker.completed);
  setText('pdf-queue', s.pdf_worker.queue_size);
  setText('pdf-errors', s.pdf_worker.errors);
  setText('pdf-last-time', fmtSec(s.pdf_worker.last_elapsed_s));
  setText('pdf-type', s.pdf_worker.last_pdf_type || '--');
  setText('pdf-current', truncate(s.pdf_worker.current_patent, 20));

  // OCR
  setText('ocr-completed', s.ocr_worker.completed);
  setText('ocr-queue', s.ocr_worker.queue_size);
  setText('ocr-pages', s.ocr_worker.last_pages_used || '--');
  setText('ocr-reason', s.ocr_worker.last_page_reason || '--');
  setText('ocr-sections', (s.ocr_worker.last_sections || []).join(' ') || '--');
  setText('ocr-last-time', fmtSec(s.ocr_worker.last_elapsed_s));
  setText('ocr-current', truncate(s.ocr_worker.current_patent, 20));

  // LLM
  setText('llm-completed', s.llm_worker.completed);
  setText('llm-queue', s.llm_worker.queue_size);
  setText('llm-errors', s.llm_worker.errors);
  setText('llm-last-time', fmtSec(s.llm_worker.last_elapsed_s));
  const preview = s.llm_worker.last_result_preview || {};
  setText('llm-found', preview.found != null ? (preview.found ? 'Yes' : 'No') : '--');
  setText('llm-current', truncate(s.llm_worker.current_patent, 20));

  // Output
  setText('out-completed', s.output_writer.completed);
  setText('out-successes', s.output_writer.successes);
  setText('out-failures', s.output_writer.failures);
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
      const parsed = JSON.parse(raw);
      el.textContent = JSON.stringify(parsed, null, 2);
    } catch {
      el.textContent = raw;
    }
  }
}

// ── Comparison modal ────────────────────────────────────

document.getElementById('btn-compare').addEventListener('click', () => {
  if (!lastComparison) return;
  openModal(lastComparison);
});

document.getElementById('modal-close').addEventListener('click', closeModal);
document.getElementById('modal-overlay').addEventListener('click', (e) => {
  if (e.target === e.currentTarget) closeModal();
});

function openModal(comp) {
  document.getElementById('modal-patent-id').textContent = comp.patent_id;

  // Images
  const imgCol = document.getElementById('modal-images');
  imgCol.innerHTML = '<h3>PDF Pages</h3>';
  (comp.page_images || []).forEach(src => {
    const img = document.createElement('img');
    img.src = src + '?' + Date.now();
    img.alt = 'Patent page';
    imgCol.appendChild(img);
  });

  // JSON result
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
  if (el) el.textContent = val != null ? val : '--';
}

function fmtSec(s) {
  if (s == null || s === 0) return '--';
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
  return str.length > len ? str.slice(0, len) + '...' : str;
}

// ── Start polling ───────────────────────────────────────
tick();
setInterval(tick, 1000);
