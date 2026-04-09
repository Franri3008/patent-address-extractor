"""Pipeline profiler — fine-grained per-patent and aggregate timing.

Always-on instrumentation with negligible overhead (perf_counter calls +
dict writes under a lock). Reports p50/p95 latencies, cold-start vs
steady-state analysis, concurrency gauges, and bottleneck identification.
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import get_logger

logger = get_logger("profiler")


@dataclass
class PatentProfile:
    patent_id: str
    pipeline_enter_t: float = 0.0
    pdf_download_s: float = 0.0
    pdf_render_s: float = 0.0
    pdf_total_s: float = 0.0
    ocr_queue_wait_s: float = 0.0
    ocr_per_page_s: list[float] = field(default_factory=list)
    ocr_total_s: float = 0.0
    ocr_pages: int = 0
    llm_queue_wait_s: float = 0.0
    llm_inference_s: float = 0.0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    output_write_s: float = 0.0
    pipeline_exit_t: float = 0.0
    total_wall_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "patent_id": self.patent_id,
            "pdf_download_s": self.pdf_download_s,
            "pdf_render_s": self.pdf_render_s,
            "pdf_total_s": self.pdf_total_s,
            "ocr_queue_wait_s": self.ocr_queue_wait_s,
            "ocr_per_page_s": self.ocr_per_page_s,
            "ocr_total_s": self.ocr_total_s,
            "ocr_pages": self.ocr_pages,
            "llm_queue_wait_s": self.llm_queue_wait_s,
            "llm_inference_s": self.llm_inference_s,
            "llm_tokens_in": self.llm_tokens_in,
            "llm_tokens_out": self.llm_tokens_out,
            "output_write_s": self.output_write_s,
            "total_wall_s": self.total_wall_s,
        }


class PipelineProfiler:
    """Thread-safe profiler that collects per-patent timing and concurrency data."""

    def __init__(self) -> None:
        self._profiles: dict[str, PatentProfile] = {}
        self._lock = threading.Lock()

        # Concurrency gauges
        self._ocr_inflight = 0
        self._ocr_inflight_max = 0
        self._ocr_inflight_samples: list[int] = []

        self._llm_inflight = 0
        self._llm_inflight_max = 0
        self._llm_inflight_samples: list[int] = []

    # ── Patent lifecycle ─────────────────────────────────────────────

    def start_patent(self, patent_id: str) -> None:
        with self._lock:
            self._profiles[patent_id] = PatentProfile(
                patent_id=patent_id,
                pipeline_enter_t=time.perf_counter(),
            )

    def _get(self, patent_id: str) -> PatentProfile | None:
        return self._profiles.get(patent_id)

    # ── PDF stage ────────────────────────────────────────────────────

    def record_pdf(self, patent_id: str, download_s: float, render_s: float, total_s: float) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.pdf_download_s = download_s
                p.pdf_render_s = render_s
                p.pdf_total_s = total_s

    # ── OCR stage ────────────────────────────────────────────────────

    def record_ocr_start(self, patent_id: str, queue_wait_s: float) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.ocr_queue_wait_s = queue_wait_s

    def record_ocr_page(self, patent_id: str, page_elapsed_s: float) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.ocr_per_page_s.append(page_elapsed_s)

    def record_ocr_done(self, patent_id: str, total_s: float, pages: int) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.ocr_total_s = total_s
                p.ocr_pages = pages

    def ocr_inflight_inc(self) -> None:
        with self._lock:
            self._ocr_inflight += 1
            self._ocr_inflight_max = max(self._ocr_inflight_max, self._ocr_inflight)
            self._ocr_inflight_samples.append(self._ocr_inflight)

    def ocr_inflight_dec(self) -> None:
        with self._lock:
            self._ocr_inflight = max(0, self._ocr_inflight - 1)
            self._ocr_inflight_samples.append(self._ocr_inflight)

    # ── LLM stage ────────────────────────────────────────────────────

    def record_llm_start(self, patent_id: str, queue_wait_s: float) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.llm_queue_wait_s = queue_wait_s

    def record_llm_done(
        self, patent_id: str, inference_s: float, tokens_in: int, tokens_out: int
    ) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.llm_inference_s = inference_s
                p.llm_tokens_in = tokens_in
                p.llm_tokens_out = tokens_out

    def llm_inflight_inc(self) -> None:
        with self._lock:
            self._llm_inflight += 1
            self._llm_inflight_max = max(self._llm_inflight_max, self._llm_inflight)
            self._llm_inflight_samples.append(self._llm_inflight)

    def llm_inflight_dec(self) -> None:
        with self._lock:
            self._llm_inflight = max(0, self._llm_inflight - 1)
            self._llm_inflight_samples.append(self._llm_inflight)

    # ── Output stage ─────────────────────────────────────────────────

    def record_output_done(self, patent_id: str, write_s: float) -> None:
        with self._lock:
            p = self._get(patent_id)
            if p:
                p.output_write_s = write_s
                p.pipeline_exit_t = time.perf_counter()
                p.total_wall_s = p.pipeline_exit_t - p.pipeline_enter_t

    # ── Query ────────────────────────────────────────────────────────

    def get_patent_profile(self, patent_id: str) -> dict | None:
        with self._lock:
            p = self._get(patent_id)
            return p.to_dict() if p else None

    # ── Reporting ────────────────────────────────────────────────────

    def _compute_aggregates(self, total_elapsed_s: float) -> dict:
        with self._lock:
            profiles = sorted(self._profiles.values(), key=lambda p: p.pipeline_enter_t)
            ocr_samples = list(self._ocr_inflight_samples)
            llm_samples = list(self._llm_inflight_samples)
            ocr_max = self._ocr_inflight_max
            llm_max = self._llm_inflight_max

        if not profiles:
            return {"error": "no patents profiled"}

        n = len(profiles)

        def _stats(values: list[float]) -> dict:
            if not values:
                return {"avg": 0, "p50": 0, "p95": 0, "min": 0, "max": 0}
            avg = sum(values) / len(values)
            if len(values) >= 2:
                quantiles = statistics.quantiles(values, n=20)
                p50 = quantiles[9]   # 50th percentile
                p95 = quantiles[18]  # 95th percentile
            else:
                p50 = values[0]
                p95 = values[0]
            return {
                "avg": round(avg, 4),
                "p50": round(p50, 4),
                "p95": round(p95, 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
            }

        total_walls = [p.total_wall_s for p in profiles if p.total_wall_s > 0]
        pdf_downloads = [p.pdf_download_s for p in profiles if p.pdf_download_s > 0]
        pdf_renders = [p.pdf_render_s for p in profiles if p.pdf_render_s > 0]
        pdf_totals = [p.pdf_total_s for p in profiles if p.pdf_total_s > 0]
        ocr_queues = [p.ocr_queue_wait_s for p in profiles if p.ocr_queue_wait_s > 0]
        ocr_totals = [p.ocr_total_s for p in profiles if p.ocr_total_s > 0]
        ocr_pages_list = [p.ocr_pages for p in profiles if p.ocr_pages > 0]
        all_ocr_page_times = [t for p in profiles for t in p.ocr_per_page_s]
        llm_queues = [p.llm_queue_wait_s for p in profiles if p.llm_queue_wait_s > 0]
        llm_inferences = [p.llm_inference_s for p in profiles if p.llm_inference_s > 0]
        llm_tok_in = [p.llm_tokens_in for p in profiles if p.llm_tokens_in > 0]
        llm_tok_out = [p.llm_tokens_out for p in profiles if p.llm_tokens_out > 0]
        output_writes = [p.output_write_s for p in profiles if p.output_write_s > 0]

        # Cold start vs steady state
        cold_n = min(5, n)
        cold = profiles[:cold_n]
        steady = profiles[cold_n:]
        cold_walls = [p.total_wall_s for p in cold if p.total_wall_s > 0]
        steady_walls = [p.total_wall_s for p in steady if p.total_wall_s > 0]

        # Concurrency gauge stats
        ocr_inflight_avg = round(sum(ocr_samples) / len(ocr_samples), 2) if ocr_samples else 0
        llm_inflight_avg = round(sum(llm_samples) / len(llm_samples), 2) if llm_samples else 0

        return {
            "total_patents": n,
            "total_elapsed_s": round(total_elapsed_s, 2),
            "patents_per_sec": round(n / total_elapsed_s, 3) if total_elapsed_s > 0 else 0,
            "avg_sec_per_patent": round(total_elapsed_s / n, 3) if n > 0 else 0,
            "total_wall": _stats(total_walls),
            "pdf_download": _stats(pdf_downloads),
            "pdf_render": _stats(pdf_renders),
            "pdf_total": _stats(pdf_totals),
            "ocr_queue_wait": _stats(ocr_queues),
            "ocr_per_page": _stats(all_ocr_page_times),
            "ocr_per_patent": _stats(ocr_totals),
            "ocr_pages_per_patent": round(sum(ocr_pages_list) / len(ocr_pages_list), 2) if ocr_pages_list else 0,
            "llm_queue_wait": _stats(llm_queues),
            "llm_inference": _stats(llm_inferences),
            "llm_tokens_in": _stats([float(t) for t in llm_tok_in]),
            "llm_tokens_out": _stats([float(t) for t in llm_tok_out]),
            "output_write": _stats(output_writes),
            "cold_start_avg_wall_s": round(sum(cold_walls) / len(cold_walls), 3) if cold_walls else 0,
            "steady_state_avg_wall_s": round(sum(steady_walls) / len(steady_walls), 3) if steady_walls else 0,
            "ocr_inflight_max": ocr_max,
            "ocr_inflight_avg": ocr_inflight_avg,
            "llm_inflight_max": llm_max,
            "llm_inflight_avg": llm_inflight_avg,
        }

    def print_profile_report(self, total_elapsed_s: float) -> None:
        agg = self._compute_aggregates(total_elapsed_s)
        if "error" in agg:
            logger.info(f"Profiler: {agg['error']}")
            return

        def _fmt(s: dict) -> str:
            return f"avg={s['avg']:.3f}  p50={s['p50']:.3f}  p95={s['p95']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}"

        lines = [
            "",
            "=" * 80,
            "  PIPELINE PROFILE REPORT",
            "=" * 80,
            f"  Total patents:        {agg['total_patents']}",
            f"  Total elapsed:        {agg['total_elapsed_s']:.1f}s",
            f"  Throughput:           {agg['patents_per_sec']:.3f} patents/sec",
            f"  Avg sec/patent:       {agg['avg_sec_per_patent']:.3f}s",
            "",
            "  ── Per-patent wall time (seconds) ──",
            f"    {_fmt(agg['total_wall'])}",
            "",
            "  ── PDF stage ──",
            f"    Download:   {_fmt(agg['pdf_download'])}",
            f"    Render:     {_fmt(agg['pdf_render'])}",
            f"    Total:      {_fmt(agg['pdf_total'])}",
            "",
            "  ── OCR stage ──",
            f"    Queue wait: {_fmt(agg['ocr_queue_wait'])}",
            f"    Per page:   {_fmt(agg['ocr_per_page'])}",
            f"    Per patent: {_fmt(agg['ocr_per_patent'])}",
            f"    Pages/pat:  {agg['ocr_pages_per_patent']}",
            f"    Inflight:   max={agg['ocr_inflight_max']}  avg={agg['ocr_inflight_avg']}",
            "",
            "  ── LLM stage ──",
            f"    Queue wait: {_fmt(agg['llm_queue_wait'])}",
            f"    Inference:  {_fmt(agg['llm_inference'])}",
            f"    Tokens in:  {_fmt(agg['llm_tokens_in'])}",
            f"    Tokens out: {_fmt(agg['llm_tokens_out'])}",
            f"    Inflight:   max={agg['llm_inflight_max']}  avg={agg['llm_inflight_avg']}",
            "",
            "  ── Output stage ──",
            f"    Write time: {_fmt(agg['output_write'])}",
            "",
            "  ── Cold start vs steady state ──",
            f"    Cold (first 5):  {agg['cold_start_avg_wall_s']:.3f}s avg wall",
            f"    Steady (rest):   {agg['steady_state_avg_wall_s']:.3f}s avg wall",
            "=" * 80,
            "",
        ]
        report = "\n".join(lines)
        # Print to stdout directly for visibility
        print(report)
        logger.info(f"Profile report generated for {agg['total_patents']} patents")

    def save_report(self, path: Path) -> None:
        total_elapsed = 0.0
        with self._lock:
            profiles = sorted(self._profiles.values(), key=lambda p: p.pipeline_enter_t)
            if profiles and profiles[-1].pipeline_exit_t > 0 and profiles[0].pipeline_enter_t > 0:
                total_elapsed = profiles[-1].pipeline_exit_t - profiles[0].pipeline_enter_t

        agg = self._compute_aggregates(total_elapsed)
        per_patent = [p.to_dict() for p in profiles]

        report = {
            "aggregates": agg,
            "per_patent": per_patent,
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Profile report saved to {path}")
