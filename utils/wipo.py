import re

# Matches any parenthesised 2-3 digit number anywhere in text (used for
# section detection / page heuristic).
_SECTION_RE = re.compile(r"(?<!\d)\((\d{2,3})\)(?!\d)");

# Matches WIPO section headers only when they appear at the start of a line
# (optionally preceded by whitespace). This avoids false positives like house
# numbers "Hauptstraße (72), Berlin" which are always mid-line.
_SECTION_HEADER_RE = re.compile(r"^\s*\((\d{2,3})\)", re.MULTILINE);

_TARGET_SECTION = 72;


def extract_section_text(text: str, section_num: int) -> str | None:
    """Extract the text block for a given WIPO section number.

    Finds the line-leading `(section_num)` header and returns everything up to
    the next line-leading section header with a higher number. Returns None if
    the section is not present.

    Using a line-anchored pattern avoids false positives from parenthesised
    numbers inside address strings (e.g. house numbers like "(72)").
    """
    # Collect all line-leading section headers with their positions.
    headers: list[tuple[int, int]] = [];  # (section_number, match_start)
    for m in _SECTION_HEADER_RE.finditer(text):
        num = int(m.group(1));
        if 10 <= num <= 899:
            headers.append((num, m.start()));

    # Find the target section.
    for i, (num, pos) in enumerate(headers):
        if num == section_num:
            # End is the next header with a strictly higher number.
            end_pos: int | None = None;
            for num2, pos2 in headers[i + 1:]:
                if num2 > section_num:
                    end_pos = pos2;
                    break;
            block = text[pos:end_pos].strip() if end_pos else text[pos:].strip();
            return block if block else None;

    return None;


def extract_sections(text: str) -> set[int]:
    """Return all WIPO section numbers found in OCR text."""
    return {int(m) for m in _SECTION_RE.findall(text) if 10 <= int(m) <= 899};


def page_decision(sections: set[int]) -> tuple[str, str]:
    """
    Decide whether to fetch another page based on WIPO section numbers found
    so far across all OCR'd pages.

    Returns
    -------
    (decision, reason)
    decision : 'done' | 'next_page'
    reason   : 'complete' | 'continues' | 'absent' | 'not_reached'

    Logic (TARGET = 72, Inventors section):
      (72) found AND higher section present  -> complete, stop
      (72) found AND no higher section       -> section continues, fetch next page
      (72) absent AND higher section present -> section absent from patent, stop
      (72) absent AND no higher section      -> not yet reached, fetch next page
    """
    has_target = _TARGET_SECTION in sections;
    has_higher = any(s > _TARGET_SECTION for s in sections);

    if has_target and has_higher:
        return "done", "complete";
    if has_target:
        return "next_page", "continues";
    if has_higher:
        return "done", "absent";
    return "next_page", "not_reached";
