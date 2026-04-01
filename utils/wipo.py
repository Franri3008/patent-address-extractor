import re

_SECTION_RE = re.compile(r"(?<!\d)\((\d{2,3})\)(?!\d)");
_TARGET_SECTION = 72;


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
