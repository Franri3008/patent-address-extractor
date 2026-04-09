from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from PIL import Image as PILImage

class StatusTracker:
    def __init__(self, dashboard_dir: Path, run_id: str, run_mode: str,
                 pipeline_mode: int, total_patents: int,
                 broadcast_fn=None):
        self._dir = dashboard_dir
        self._pages_dir = dashboard_dir / "pages"
        self._pages_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._broadcast_fn = broadcast_fn

        self.state: dict = {
            "run_id": run_id,
            "run_mode": run_mode,
            "pipeline_mode": pipeline_mode,
            "total_patents": total_patents,
            "started_at": time.time(),
            "updated_at": time.time(),
            "stages": {
                "bq_fetch": {
                    "status": "pending",
                    "patents_fetched": 0,
                    "elapsed_s": 0.0,
                },
                "pdf_worker": {
                    "status": "pending",
                    "active": 0,
                    "completed": 0,
                    "errors": 0,
                    "queue_size": 0,
                    "current_patent": "",
                    "last_elapsed_s": 0.0,
                    "last_pdf_type": "",
                },
                "ocr_worker": {
                    "status": "pending",
                    "active": 0,
                    "completed": 0,
                    "errors": 0,
                    "queue_size": 0,
                    "current_patent": "",
                    "last_pages_used": 0,
                    "last_page_reason": "",
                    "last_sections": [],
                    "last_elapsed_s": 0.0,
                },
                "llm_worker": {
                    "status": "pending",
                    "active": 0,
                    "completed": 0,
                    "errors": 0,
                    "queue_size": 0,
                    "current_patent": "",
                    "last_elapsed_s": 0.0,
                    "last_raw_response": "",
                    "last_result_preview": {},
                },
                "output_writer": {
                    "status": "pending",
                    "completed": 0,
                    "successes": 0,
                    "failures": 0,
                },
            },
            "latest_comparison": {
                "patent_id": "",
                "page_images": [],
                "llm_result": {"inventors": [], "applicants": [], "agents": []},
            },
            "timings": {
                "avg_pdf_s": 0.0,
                "avg_ocr_s": 0.0,
                "avg_llm_s": 0.0,
                "total_elapsed_s": 0.0,
            },
        }
        self._pdf_times: list[float] = []
        self._ocr_times: list[float] = []
        self._llm_times: list[float] = []
        self._flush()

    def update(self, stage: str, **kwargs) -> None:
        with self._lock:
            self.state["stages"][stage].update(kwargs)
            self.state["updated_at"] = time.time()
            self.state["timings"]["total_elapsed_s"] = round(
                time.time() - self.state["started_at"], 2
            )
            self._flush()

    def record_timing(self, stage: str, elapsed_s: float) -> None:
        with self._lock:
            if stage == "pdf":
                self._pdf_times.append(elapsed_s)
                self.state["timings"]["avg_pdf_s"] = round(
                    sum(self._pdf_times) / len(self._pdf_times), 3
                )
            elif stage == "ocr":
                self._ocr_times.append(elapsed_s)
                self.state["timings"]["avg_ocr_s"] = round(
                    sum(self._ocr_times) / len(self._ocr_times), 3
                )
            elif stage == "llm":
                self._llm_times.append(elapsed_s)
                self.state["timings"]["avg_llm_s"] = round(
                    sum(self._llm_times) / len(self._llm_times), 3
                )

    def set_comparison(self, patent_id: str, page_images: list[str], llm_result: dict) -> None:
        with self._lock:
            self.state["latest_comparison"] = {
                "patent_id": patent_id,
                "page_images": page_images,
                "llm_result": llm_result,
            }
            self._flush()

    def save_page_images(self, patent_id: str, images: list[PILImage.Image], pages_used: int) -> list[str]:
        paths: list[str] = []
        clean_id = patent_id.replace("-", "")
        for i, img in enumerate(images[:pages_used], start=1):
            fname = f"{clean_id}_p{i}.jpg"
            out = self._pages_dir / fname
            img.save(out, format="JPEG", quality=75)
            paths.append(f"pages/{fname}")
        return paths

    def finish(self) -> None:
        with self._lock:
            for stage in self.state["stages"].values():
                if stage["status"] == "running":
                    stage["status"] = "done"
            self.state["updated_at"] = time.time()
            self.state["timings"]["total_elapsed_s"] = round(
                time.time() - self.state["started_at"], 2
            )
            self._flush()

    def _flush(self) -> None:
        tmp = self._dir / "status.json.tmp"
        dst = self._dir / "status.json"
        tmp.write_text(json.dumps(self.state, ensure_ascii=False, default=str))
        tmp.rename(dst)
        if self._broadcast_fn is not None:
            self._broadcast_fn(self.state)
