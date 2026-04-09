"""Two-column layout detection and splitting for WIPO PCT patent first pages.

WIPO first pages have a consistent layout:
  - Top ~60-70%: two-column bibliographic data (left = sections 71/72/74,
    right = classification codes, abstract continuation, etc.)
  - Horizontal separator line
  - Bottom: full-width title, abstract, and drawings

This module detects the separator and column gap using projection profiles,
then crops just the left column above the separator for OCR.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image as PILImage

from utils.logger import get_logger

logger = get_logger("preprocessing.columns");


@dataclass
class ColumnLayout:
    is_two_column: bool;
    split_x: int | None = None;       # vertical column boundary (x-coordinate)
    separator_y: int | None = None;    # horizontal separator line (y-coordinate)
    confidence: float = 0.0;


def detect_columns(img: PILImage.Image) -> ColumnLayout:
    """Detect two-column layout and horizontal separator on a WIPO first page.

    Uses projection profiles (no OpenCV needed for detection itself):
      1. Find the horizontal separator line (a row of high ink density spanning
         most of the page width).
      2. In the region above the separator, find the vertical column gap
         (a column of low ink density near the center).
    """
    gray = img.convert("L");
    arr = np.array(gray);
    h, w = arr.shape;

    # Binarise: dark pixels = 1, light = 0 (inverted so ink = high values)
    threshold = 128;
    binary = (arr < threshold).astype(np.float32);

    # --- Step 1: Find horizontal separator line ---
    # Compute horizontal projection: sum of dark pixels per row.
    h_proj = binary.sum(axis=1);

    # The separator is a thin horizontal rule spanning most of the page width.
    # Look between 35% and 80% of page height.
    search_top = int(h * 0.35);
    search_bot = int(h * 0.80);
    min_line_density = w * 0.5;  # must span at least 50% of page width

    separator_y: int | None = None;
    best_density = 0.0;

    for y in range(search_top, search_bot):
        density = h_proj[y];
        if density > min_line_density and density > best_density:
            best_density = density;
            separator_y = y;

    if separator_y is None:
        logger.debug("No horizontal separator found");
        return ColumnLayout(is_two_column=False);

    # Refine: skip past any cluster of dense rows (the line may be >1px thick)
    while separator_y > 0 and h_proj[separator_y - 1] > min_line_density * 0.5:
        separator_y -= 1;

    logger.debug(f"Horizontal separator detected at y={separator_y} (page height={h})");

    # --- Step 2: Find vertical column gap above the separator ---
    # Use the region above the separator for vertical projection.
    top_region = binary[:separator_y, :];
    v_proj = top_region.sum(axis=0);

    # The column gap is near the center. Search between 30% and 70% of width.
    search_left = int(w * 0.30);
    search_right = int(w * 0.70);

    if search_left >= search_right:
        return ColumnLayout(is_two_column=False);

    center_v_proj = v_proj[search_left:search_right];
    gap_idx = int(np.argmin(center_v_proj));
    split_x = search_left + gap_idx;
    gap_density = center_v_proj[gap_idx];

    # Confidence: how "empty" is the gap compared to the columns?
    # Low gap density + high column density = high confidence.
    left_density = v_proj[:split_x].mean() if split_x > 0 else 0;
    right_density = v_proj[split_x:].mean() if split_x < w else 0;
    avg_col_density = (left_density + right_density) / 2;

    if avg_col_density == 0:
        return ColumnLayout(is_two_column=False);

    # Confidence: 1.0 when gap is completely empty, 0.0 when gap = avg column density
    confidence = max(0.0, 1.0 - (gap_density / avg_col_density));

    logger.debug(
        f"Column gap at x={split_x}, gap_density={gap_density:.1f}, "
        f"avg_col_density={avg_col_density:.1f}, confidence={confidence:.3f}"
    );

    return ColumnLayout(
        is_two_column=True,
        split_x=split_x,
        separator_y=separator_y,
        confidence=confidence,
    );


def split_top_left_column(img: PILImage.Image, layout: ColumnLayout) -> PILImage.Image:
    """Crop just the left column above the horizontal separator.

    This isolates the bibliographic data (sections 71, 72, 74) from the
    right column (classification codes, abstract) and the bottom section
    (title, drawings), preventing OCR from reading across columns.
    """
    if not layout.is_two_column or layout.split_x is None or layout.separator_y is None:
        return img;

    # Crop: (left, top, right, bottom)
    # Add a small margin to the right of the split to avoid cutting text
    margin = 10;
    right_edge = min(layout.split_x + margin, img.width);
    return img.crop((0, 0, right_edge, layout.separator_y));
