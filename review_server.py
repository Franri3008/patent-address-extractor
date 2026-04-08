#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import mimetypes
import os
import shutil
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR        = Path(__file__).parent
DASHBOARD_DIR   = BASE_DIR / "dashboard"
OUTPUT_DIR      = BASE_DIR / "output"
INDIVIDUAL_DIR  = OUTPUT_DIR / "individual"
IMAGES_DIR      = OUTPUT_DIR / "images"   # batch-mode thumbnails
TEST_DIR        = BASE_DIR / "test"
GROUND_TRUTH    = TEST_DIR / "ground_truth.csv"
TEST_IMAGES_DIR = TEST_DIR / "images"

GT_COLUMNS = [
    "patent_id", "country_code", "language", "llm_output_json",
    "is_correct", "reviewed_at", "run_id", "llm_provider", "ocr_model",
]

PORT = 8765


def _load_reviews() -> dict[str, dict]:
    """Return {patent_id: row_dict} from ground_truth.csv."""
    if not GROUND_TRUTH.exists():
        return {}
    reviews: dict[str, dict] = {}
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            reviews[row["patent_id"]] = row
    return reviews


def _find_thumbnails(base_id: str) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Search both output/individual/ (individual mode) and output/images/ (batch mode)
    search_roots: list[tuple[Path, str]] = []
    if INDIVIDUAL_DIR.exists():
        search_roots.append((INDIVIDUAL_DIR, "individual"))
    if IMAGES_DIR.exists():
        search_roots.append((IMAGES_DIR, "images"))

    for root, prefix in search_roots:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if d.name != base_id and not d.name.startswith(base_id):
                continue
            for img in sorted(d.glob("page_*_thumb.jpg")):
                if img.name not in seen:
                    results.append((f"/thumbnails/{prefix}/{d.name}/{img.name}", d.name))
                    seen.add(img.name)
            # Also check non-thumb pattern (page_1.jpg etc)
            for img in sorted(d.glob("page_*.jpg")):
                if "_thumb" in img.name:
                    continue
                if img.name not in seen:
                    results.append((f"/thumbnails/{prefix}/{d.name}/{img.name}", d.name))
                    seen.add(img.name)
    return results


def _scan_individual() -> list[dict]:
    """Scan output/individual/* for processed patent results."""
    patents = []
    if not INDIVIDUAL_DIR.exists():
        return patents

    seen_base_ids: set[str] = set()

    for meta_path in sorted(INDIVIDUAL_DIR.glob("*/metadata.json")):
        try:
            with open(meta_path, encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    continue
                meta = json.loads(content)
        except Exception:
            continue

        patent_id = meta.get("patent_id", meta_path.parent.name).replace("-", "")
        folder_id = meta_path.parent.name
        seen_base_ids.add(patent_id)

        thumb_info = _find_thumbnails(patent_id)
        thumbs = [url for url, _ in thumb_info]

        llm_raw = meta.get("llm_raw_response") or ""
        llm_result = meta.get("result", {})
        try:
            llm_output = json.loads(llm_raw) if llm_raw else {}
        except Exception:
            llm_output = {"raw": llm_raw}

        if llm_result:
            llm_output.setdefault("found", llm_result.get("found"))
            llm_output.setdefault("inventors_count", llm_result.get("inventors_count"))
            llm_output.setdefault("applicants_count", llm_result.get("applicants_count"))
            llm_output.setdefault("sections_detected", llm_result.get("sections_detected"))

        llm_section = meta.get("llm", {})
        ocr_section = meta.get("ocr", {})

        patents.append({
            "patent_id":     patent_id,
            "folder_id":     folder_id,
            "country_code":  patent_id[:2] if len(patent_id) >= 2 else "WO",
            "language":      meta.get("title_language") or _guess_language(meta),
            "llm_output":    llm_output,
            "run_id":        meta.get("run_id", ""),
            "llm_provider":  f"{llm_section.get('provider','')}/{llm_section.get('model','')}".strip("/"),
            "ocr_model":     ocr_section.get("model", ""),
            "has_images":    len(thumbs) > 0,
            "thumbnail_paths": thumbs,
            "source":        "individual",
            "publication_date": meta.get("publication_date", ""),
        })

    for root, prefix in [(INDIVIDUAL_DIR, "individual"), (IMAGES_DIR, "images")]:
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            if any(d.name == sid or d.name.startswith(sid) for sid in seen_base_ids):
                continue
            imgs = sorted(d.glob("page_*_thumb.jpg")) or sorted(d.glob("page_*.jpg"))
            if not imgs:
                continue
            folder_id = d.name
            thumbs = [f"/thumbnails/{prefix}/{folder_id}/{img.name}" for img in imgs]
            patents.append({
                "patent_id":     folder_id,
                "folder_id":     folder_id,
                "country_code":  folder_id[:2] if len(folder_id) >= 2 else "WO",
                "language":      "",
                "llm_output":    {},
                "run_id":        "",
                "llm_provider":  "",
                "ocr_model":     "",
                "has_images":    True,
                "thumbnail_paths": thumbs,
                "source":        "individual",
                "publication_date": "",
            })

    return patents


def _guess_language(meta: dict) -> str:
    for key in ("title_language", "language"):
        v = meta.get(key)
        if v:
            return str(v)
    return ""


def _scan_batch_csv() -> list[dict]:
    patents = []
    existing_ids: set[str] = set()

    if INDIVIDUAL_DIR.exists():
        for d in INDIVIDUAL_DIR.iterdir():
            if d.is_dir():
                existing_ids.add(d.name.replace("-", ""))

    for csv_path in sorted(OUTPUT_DIR.glob("*.csv")):
        if csv_path.stem.startswith("raw_"):
            continue
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pid = row.get("publication_number", "").replace("-", "")
                    if not pid or pid in existing_ids:
                        continue
                    existing_ids.add(pid)

                    llm_output: dict = {}
                    inv_json = row.get("inventors_with_address", "")
                    app_json = row.get("applicants_with_address", "")
                    try:
                        llm_output["inventors"] = json.loads(inv_json) if inv_json else []
                    except Exception:
                        llm_output["inventors"] = []
                    try:
                        llm_output["applicants"] = json.loads(app_json) if app_json else []
                    except Exception:
                        llm_output["applicants"] = []
                    llm_output["found"] = row.get("addresses_found", "").lower() == "true"
                    llm_output["sections_detected"] = row.get("sections_found", "").split() if row.get("sections_found") else []

                    thumb_info = _find_thumbnails(pid)
                    patents.append({
                        "patent_id":       pid,
                        "folder_id":       pid,
                        "country_code":    row.get("country_code", pid[:2]),
                        "language":        row.get("title_language", ""),
                        "llm_output":      llm_output,
                        "run_id":          "",
                        "llm_provider":    row.get("llm_provider", ""),
                        "ocr_model":       row.get("ocr_model", ""),
                        "has_images":      len(thumb_info) > 0,
                        "thumbnail_paths": [url for url, _ in thumb_info],
                        "source":          "batch",
                        "publication_date": str(row.get("publication_date", "")),
                    })
        except Exception:
            continue

    return patents


def get_all_patents() -> list[dict]:
    """Return merged list of all reviewable patents, newest first."""
    patents = _scan_individual() + _scan_batch_csv()
    reviews = _load_reviews()

    for p in patents:
        pid = p["patent_id"]
        if pid in reviews:
            p["reviewed"]   = True
            p["is_correct"] = reviews[pid]["is_correct"].lower() == "true"
        else:
            p["reviewed"]   = False
            p["is_correct"] = None

    patents.sort(key=lambda x: (x["reviewed"], x.get("publication_date", "") or ""), reverse=False)
    return patents


def save_review(patent_id: str, is_correct: bool, patents_index: dict[str, dict]) -> dict:
    """Append a review to test/ground_truth.csv and copy images."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    p = patents_index.get(patent_id)
    if p is None:
        raise ValueError(f"Patent {patent_id} not found in index")

    dest_img_dir = TEST_IMAGES_DIR / p["folder_id"]
    if p.get("has_images"):
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        for thumb_url in p.get("thumbnail_paths", []):
            parts = thumb_url.lstrip("/").split("/")
            if len(parts) == 3:
                src = INDIVIDUAL_DIR / parts[1] / parts[2]
                if src.exists():
                    shutil.copy2(src, dest_img_dir / parts[2])

    row = {
        "patent_id":      patent_id,
        "country_code":   p.get("country_code", ""),
        "language":       p.get("language", ""),
        "llm_output_json": json.dumps(p.get("llm_output", {}), ensure_ascii=False),
        "is_correct":     str(is_correct).lower(),
        "reviewed_at":    datetime.now(timezone.utc).isoformat(),
        "run_id":         p.get("run_id", ""),
        "llm_provider":   p.get("llm_provider", ""),
        "ocr_model":      p.get("ocr_model", ""),
    }

    write_header = not GROUND_TRUTH.exists()
    with open(GROUND_TRUTH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


class ReviewHandler(BaseHTTPRequestHandler):
    _patents_cache: list[dict] | None = None
    _patents_index: dict[str, dict] = {}

    def log_message(self, fmt, *args): 
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/") or "/"

        if path == "/" or path == "/index.html":
            self._serve_file(DASHBOARD_DIR / "index.html")
        elif path == "/status.json":
            self._serve_file(DASHBOARD_DIR / "status.json")
        elif path.startswith("/css/"):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        elif path.startswith("/js/"):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        elif path.startswith("/ui/"):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        elif path.startswith("/pages/"):
            self._serve_file(DASHBOARD_DIR / path.lstrip("/"))
        elif path.startswith("/thumbnails/"):
            # /thumbnails/{prefix}/{folder_id}/{filename}
            # prefix is either 'individual' or 'images'
            rel = path[len("/thumbnails/"):]
            candidate = OUTPUT_DIR / rel
            self._serve_file(candidate)
        elif path == "/api/patents":
            self._api_patents()
        elif path == "/api/reviews":
            self._api_reviews()
        else:
            self._not_found()

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        if path == "/api/review":
            self._api_submit_review()
        else:
            self._not_found()


    def _api_patents(self):
        patents = get_all_patents()
        ReviewHandler._patents_cache = patents
        ReviewHandler._patents_index = {p["patent_id"]: p for p in patents}
        self._json_response(patents)

    def _api_reviews(self):
        reviews = list(_load_reviews().values())
        self._json_response(reviews)

    def _api_submit_review(self):
        length = int(self.headers.get("Content-Length", 0))
        body   = self.rfile.read(length)
        try:
            data       = json.loads(body)
            patent_id  = data["patent_id"]
            is_correct = bool(data["is_correct"])
        except Exception as exc:
            self._error(400, str(exc))
            return

        if not ReviewHandler._patents_index:
            patents = get_all_patents()
            ReviewHandler._patents_cache = patents
            ReviewHandler._patents_index = {p["patent_id"]: p for p in patents}

        try:
            row = save_review(patent_id, is_correct, ReviewHandler._patents_index)
            # Invalidate cache so next /api/patents reflects the review
            ReviewHandler._patents_cache = None
        except ValueError as exc:
            self._error(404, str(exc))
            return
        except Exception as exc:
            self._error(500, str(exc))
            return

        self._json_response({"ok": True, "row": row}, status=201)

    def _serve_file(self, path: Path):
        if not path.exists() or not path.is_file():
            self._not_found()
            return
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json_response(self, obj, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self):
        self.send_response(404)
        self.end_headers()

    def _error(self, code: int, msg: str):
        data = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start_review_server(config: dict | None = None, port: int = PORT) -> None:
    """Start the review server and open the browser."""
    url = f"http://localhost:{port}"
    print(f"\nPatent Review Server")
    print(f"  URL:           {url}")
    print(f"  Ground truth:  {GROUND_TRUTH}")
    print(f"  Output dir:    {OUTPUT_DIR}")
    print(f"  Press Ctrl+C to stop.\n")

    server = HTTPServer(("localhost", port), ReviewHandler)

    threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nReview server stopped.")


if __name__ == "__main__":
    start_review_server()
