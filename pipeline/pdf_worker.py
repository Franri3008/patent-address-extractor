"""PDF download and page extraction worker (async).

Responsibilities:
  - Build Google Patents URL from publication_number
  - Scrape the page to find the PDF link
  - Download the PDF to a temp file
  - Detect pdf_type (vector vs scanned via pypdf)
  - Convert all pages 1..max_pages to PIL Images (in-memory)
  - In individual mode: also save thumbnail images to disk
  - Emit result dict into the image queue
"""
from __future__ import annotations

import asyncio
import io
import tempfile
import time
from pathlib import Path

import aiohttp
import pypdf
from pdf2image import convert_from_bytes
from PIL import Image as PILImage

from utils.logger import get_logger

logger = get_logger("pdf_worker");

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
};
_SCANNED_TEXT_THRESHOLD = 300;
_REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=60);


def _pub_to_url(pub_number: str) -> str:
    return f"https://patents.google.com/patent/{pub_number.replace('-', '')}";


async def _scrape_pdf_url(session: aiohttp.ClientSession, page_url: str) -> str | None:
    """Scrape Google Patents page for the PDF download link."""
    try:
        async with session.get(page_url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT) as resp:
            resp.raise_for_status();
            html = await resp.text();
    except Exception as e:
        logger.warning(f"Failed to fetch page {page_url}: {e}");
        return None;

    import re
    for pattern in [
        r'<meta[^>]+itemprop="pdfLink"[^>]+content="([^"]+)"',
        r'<meta[^>]+name="citation_pdf_url"[^>]+content="([^"]+)"',
        r'href="(https://[^"]+\.pdf)"',
    ]:
        m = re.search(pattern, html, re.IGNORECASE);
        if m:
            return m.group(1);
    return None;


async def _download_pdf(session: aiohttp.ClientSession, pdf_url: str) -> bytes | None:
    """Download PDF bytes."""
    try:
        async with session.get(pdf_url, headers=_HEADERS, timeout=_REQUEST_TIMEOUT) as resp:
            resp.raise_for_status();
            return await resp.read();
    except Exception as e:
        logger.warning(f"Failed to download {pdf_url}: {e}");
        return None;


def _detect_pdf_type(pdf_bytes: bytes) -> str:
    """Return 'vector' if embedded text layer found, else 'scanned'."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes));
        text = reader.pages[0].extract_text() or "";
        return "vector" if len(text.strip()) >= _SCANNED_TEXT_THRESHOLD else "scanned";
    except Exception:
        return "scanned";


def _extract_pages_sync(pdf_bytes: bytes, max_pages: int, dpi: int) -> list[PILImage.Image]:
    """Convert PDF pages to PIL Images (blocking, run in thread)."""
    return convert_from_bytes(
        pdf_bytes,
        first_page=1,
        last_page=max_pages,
        dpi=dpi,
        fmt="jpeg",
    );


def _save_thumbnails(images: list[PILImage.Image], out_dir: Path, pub_number: str, thumb_dpi: int) -> list[str]:
    """Save thumbnail images to disk and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True);
    paths: list[str] = [];
    for i, img in enumerate(images, start=1):
        ratio = thumb_dpi / 150;
        w, h = img.size;
        thumb = img.resize((int(w * ratio), int(h * ratio)), PILImage.LANCZOS);
        p = out_dir / f"page_{i}_thumb.jpg";
        thumb.save(p, format="JPEG", quality=70);
        paths.append(str(p));
    return paths;


async def pdf_worker(
    patent_q: asyncio.Queue,
    image_q: asyncio.Queue,
    config: dict,
    n_sentinels_to_emit: int,
) -> None:
    """
    Single PDF worker coroutine. Consumes patent rows from patent_q,
    downloads + converts PDFs, pushes results to image_q.
    When it receives a None sentinel from patent_q, it forwards one
    sentinel downstream and exits.
    """
    pdf_cfg = config["pdf"];
    max_pages: int = pdf_cfg["max_pages"];
    dpi: int = pdf_cfg["dpi"];
    thumb_dpi: int = pdf_cfg["thumbnail_dpi"];
    individual_cfg = config.get("individual", {});
    is_individual = config["run_mode"] == "individual";
    out_dir = Path(config["output"]["dir"]);

    async with aiohttp.ClientSession() as session:
        while True:
            row = await patent_q.get();

            if row is None:
                await image_q.put(None);
                patent_q.task_done();
                return;

            pub_number: str = str(row.get("publication_number", ""));
            t0 = time.perf_counter();
            result = {
                "row": row,
                "images": None,
                "pdf_url": None,
                "pdf_type": "unknown",
                "thumbnail_paths": [],
                "error": None,
                "elapsed_s": 0.0,
            };

            try:
                page_url = _pub_to_url(pub_number);
                pdf_url = await _scrape_pdf_url(session, page_url);
                if not pdf_url:
                    raise RuntimeError(f"PDF URL not found for {pub_number}");

                result["pdf_url"] = pdf_url;
                pdf_bytes = await _download_pdf(session, pdf_url);
                if not pdf_bytes:
                    raise RuntimeError(f"PDF download failed for {pub_number}");

                result["pdf_type"] = _detect_pdf_type(pdf_bytes);

                images: list[PILImage.Image] = await asyncio.to_thread(
                    _extract_pages_sync, pdf_bytes, max_pages, dpi
                );
                result["images"] = images;

                if is_individual and individual_cfg.get("save_thumbnails"):
                    ind_out = out_dir / "individual" / pub_number.replace("-", "");
                    result["thumbnail_paths"] = _save_thumbnails(images, ind_out, pub_number, thumb_dpi);

            except Exception as e:
                result["error"] = str(e);
                logger.warning(f"[PDF] {pub_number}: {e}");
            finally:
                result["elapsed_s"] = time.perf_counter() - t0;

            await image_q.put(result);
            patent_q.task_done();
