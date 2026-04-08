// Patent Review Tab — js/review.js
// Loaded alongside app.js; activated when the Review tab is clicked.

let reviewPatents   = [];   // full list from /api/patents
let reviewMap       = {};   // patent_id → patent object
let currentPatentId = null;

// Zoom / pan state
let zoomLevel  = 1.0;
let panX       = 0;
let panY       = 0;
let isPanning  = false;
let panStartX  = 0;
let panStartY  = 0;
let panOriginX = 0;
let panOriginY = 0;

// ── Entry point (called by tab switcher in index.html) ───────────────────────

async function loadReviewTab() {
  try {
    const [patentsRes] = await Promise.all([
      fetch('/api/patents?' + Date.now()),
    ]);
    if (!patentsRes.ok) throw new Error('Could not load patents');
    reviewPatents = await patentsRes.json();
    reviewMap = {};
    reviewPatents.forEach(p => { reviewMap[p.patent_id] = p; });
    renderPatentList(reviewPatents);
  } catch (err) {
    document.getElementById('review-list').innerHTML =
      `<div class="review-empty review-error">Failed to load patents: ${err.message}</div>`;
  }
}

// ── Patent list ──────────────────────────────────────────────────────────────

function renderPatentList(patents) {
  const list  = document.getElementById('review-list');
  const count = document.getElementById('review-count');

  count.textContent = patents.length;

  if (patents.length === 0) {
    list.innerHTML = '<div class="review-empty">No processed patents found in output/.</div>';
    return;
  }

  list.innerHTML = '';
  patents.forEach(p => {
    const item = document.createElement('div');
    item.className = 'review-item' + (p.reviewed ? ' reviewed' : '');
    item.dataset.id = p.patent_id;

    const badge = p.reviewed
      ? `<span class="rv-badge ${p.is_correct ? 'badge-correct' : 'badge-wrong'}">
           ${p.is_correct ? '✓' : '✗'}
         </span>`
      : `<span class="rv-badge badge-pending">○</span>`;

    const date = p.publication_date
      ? `<span class="rv-date">${String(p.publication_date).slice(0, 8)}</span>`
      : '';

    const src  = p.source === 'batch' ? '<span class="rv-src">batch</span>' : '';
    const imgs = p.has_images ? '' : '<span class="rv-no-img" title="No images">⊘</span>';

    item.innerHTML = `
      <div class="rv-item-left">
        ${badge}
        <div>
          <div class="rv-id">${p.patent_id}</div>
          <div class="rv-meta-row">${date}${src}${imgs}</div>
        </div>
      </div>`;

    item.addEventListener('click', () => selectPatent(p.patent_id));
    list.appendChild(item);
  });
}

// ── Patent viewer ─────────────────────────────────────────────────────────────

function selectPatent(patentId) {
  // Highlight list item
  document.querySelectorAll('.review-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === patentId);
  });

  currentPatentId = patentId;
  const p = reviewMap[patentId];
  if (!p) return;

  // Reset zoom/pan
  zoomLevel = 1.0;
  panX = 0;
  panY = 0;
  applyZoom();

  // Header
  document.getElementById('rv-patent-id').textContent = p.patent_id;
  const langStr = p.language ? ` · ${p.language}` : '';
  document.getElementById('rv-meta').textContent =
    `${p.country_code}${langStr} · ${p.llm_provider || 'unknown model'}`;

  // Verdict badge
  const verdictEl = document.getElementById('rv-verdict');
  if (p.reviewed) {
    verdictEl.textContent  = p.is_correct ? '✓ Correct' : '✗ Wrong';
    verdictEl.className    = 'review-verdict ' + (p.is_correct ? 'verdict-correct' : 'verdict-wrong');
    verdictEl.hidden       = false;
  } else {
    verdictEl.textContent = '';
    verdictEl.hidden      = true;
  }

  // Images
  const zoomWrap   = document.getElementById('rv-zoom-wrap');
  const zoomInner  = document.getElementById('rv-zoom-inner');
  const noImages   = document.getElementById('rv-no-images');

  if (p.has_images && p.thumbnail_paths && p.thumbnail_paths.length > 0) {
    zoomWrap.hidden  = false;
    noImages.hidden  = true;
    zoomInner.innerHTML = '';
    p.thumbnail_paths.forEach((src, i) => {
      const img = document.createElement('img');
      img.src   = src + '?' + Date.now();
      img.alt   = `Page ${i + 1}`;
      img.className = 'rv-page-img';
      zoomInner.appendChild(img);
    });
  } else {
    zoomWrap.hidden  = true;
    noImages.hidden  = false;
  }

  // JSON output
  const jsonEl = document.getElementById('rv-json');
  try {
    jsonEl.textContent = JSON.stringify(p.llm_output, null, 2);
  } catch {
    jsonEl.textContent = String(p.llm_output);
  }

  // Vote buttons
  const btnCorrect = document.getElementById('btn-correct');
  const btnWrong   = document.getElementById('btn-wrong');
  const savedMsg   = document.getElementById('vote-saved-msg');

  if (p.reviewed) {
    btnCorrect.disabled = true;
    btnWrong.disabled   = true;
    savedMsg.textContent = 'Already reviewed';
    savedMsg.className   = 'vote-saved-msg vote-already';
  } else {
    btnCorrect.disabled = false;
    btnWrong.disabled   = false;
    savedMsg.textContent = '';
    savedMsg.className   = 'vote-saved-msg';
  }

  // Show viewer
  document.getElementById('review-viewer-empty').hidden   = true;
  document.getElementById('review-viewer-content').hidden = false;
}

// ── Vote buttons ─────────────────────────────────────────────────────────────

async function submitReview(isCorrect) {
  if (!currentPatentId) return;

  const btnCorrect = document.getElementById('btn-correct');
  const btnWrong   = document.getElementById('btn-wrong');
  const savedMsg   = document.getElementById('vote-saved-msg');

  btnCorrect.disabled = true;
  btnWrong.disabled   = true;
  savedMsg.textContent = 'Saving…';
  savedMsg.className   = 'vote-saved-msg';

  try {
    const res = await fetch('/api/review', {
      method:  'POST',
      headers: {'Content-Type': 'application/json'},
      body:    JSON.stringify({patent_id: currentPatentId, is_correct: isCorrect}),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({error: res.statusText}));
      throw new Error(err.error || res.statusText);
    }

    // Update local state
    const p = reviewMap[currentPatentId];
    p.reviewed   = true;
    p.is_correct = isCorrect;

    // Update list badge
    const listItem = document.querySelector(`.review-item[data-id="${currentPatentId}"]`);
    if (listItem) {
      listItem.classList.add('reviewed');
      const badge = listItem.querySelector('.rv-badge');
      if (badge) {
        badge.className  = `rv-badge ${isCorrect ? 'badge-correct' : 'badge-wrong'}`;
        badge.textContent = isCorrect ? '✓' : '✗';
      }
    }

    // Update viewer verdict
    const verdictEl = document.getElementById('rv-verdict');
    verdictEl.textContent = isCorrect ? '✓ Correct' : '✗ Wrong';
    verdictEl.className   = 'review-verdict ' + (isCorrect ? 'verdict-correct' : 'verdict-wrong');
    verdictEl.hidden      = false;

    savedMsg.textContent = 'Saved ✓';
    savedMsg.className   = 'vote-saved-msg vote-ok';

    // Auto-advance to next unreviewed patent after a short delay
    setTimeout(() => advanceToNext(), 800);

  } catch (err) {
    savedMsg.textContent = `Error: ${err.message}`;
    savedMsg.className   = 'vote-saved-msg vote-error';
    btnCorrect.disabled  = false;
    btnWrong.disabled    = false;
  }
}

function advanceToNext() {
  const unreviewed = reviewPatents.filter(p => !p.reviewed);
  if (unreviewed.length > 0) {
    selectPatent(unreviewed[0].patent_id);
    // Scroll the list item into view
    const el = document.querySelector(`.review-item[data-id="${unreviewed[0].patent_id}"]`);
    if (el) el.scrollIntoView({block: 'nearest', behavior: 'smooth'});
  }
}

document.getElementById('btn-correct').addEventListener('click', () => submitReview(true));
document.getElementById('btn-wrong').addEventListener('click',   () => submitReview(false));

// ── Zoom & pan ────────────────────────────────────────────────────────────────

function applyZoom() {
  const inner = document.getElementById('rv-zoom-inner');
  if (!inner) return;
  inner.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
  inner.style.transformOrigin = '0 0';
}

const zoomWrap = document.getElementById('rv-zoom-wrap');
if (zoomWrap) {
  zoomWrap.addEventListener('wheel', (e) => {
    e.preventDefault();
    const delta = e.deltaY < 0 ? 0.15 : -0.15;
    zoomLevel = Math.min(4.0, Math.max(0.5, zoomLevel + delta));
    applyZoom();
  }, {passive: false});

  zoomWrap.addEventListener('mousedown', (e) => {
    if (zoomLevel <= 1.0) return;
    isPanning  = true;
    panStartX  = e.clientX;
    panStartY  = e.clientY;
    panOriginX = panX;
    panOriginY = panY;
    zoomWrap.style.cursor = 'grabbing';
  });

  window.addEventListener('mousemove', (e) => {
    if (!isPanning) return;
    panX = panOriginX + (e.clientX - panStartX);
    panY = panOriginY + (e.clientY - panStartY);
    applyZoom();
  });

  window.addEventListener('mouseup', () => {
    if (!isPanning) return;
    isPanning = false;
    zoomWrap.style.cursor = '';
  });
}
