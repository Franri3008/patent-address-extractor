import re

# Matches any parenthesised 2-3 digit number anywhere in text (used for section detection / page heuristic)
_SECTION_RE = re.compile(r"(?<!\d)\((\d{2,3})\)(?!\d)");

# Matches WIPO section headers only when they appear at the start of a line (optionally preceded by whitespace) - avoids false positives like house numbers "Hauptstraße (72), Berlin" which are always mid-line
_SECTION_HEADER_RE = re.compile(r"^\s*\((\d{2,3})\)", re.MULTILINE);

# Valid WIPO PCT section numbers are in the range 10–90.
# Numbers like (100), (240), (251) are patent reference numerals from the
# abstract / figures and must be excluded to avoid false positives in the
# page-decision heuristic.
_MAX_WIPO_SECTION = 90;

_TARGET_SECTION = 72;

def extract_section_text(text: str, section_num: int) -> str | None:
    headers: list[tuple[int, int]] = [];  # (section_number, match_start)
    for m in _SECTION_HEADER_RE.finditer(text):
        num = int(m.group(1));
        if 10 <= num <= 899:
            headers.append((num, m.start()));

    for i, (num, pos) in enumerate(headers):
        if num == section_num:
            end_pos: int | None = None;
            for num2, pos2 in headers[i + 1:]:
                if num2 > section_num:
                    end_pos = pos2;
                    break;
            block = text[pos:end_pos].strip() if end_pos else text[pos:].strip();
            return block if block else None;

    return None;


def parse_known_names(row: dict) -> dict:
    """Extract known applicant/inventor names from BigQuery row data.

    Returns a dict suitable for passing as template_vars to the LLM prompt.
    Names come as pipe-separated strings from BigQuery (e.g. "Foo Corp | Bar Inc").
    """
    result: dict = {};
    assignees = row.get("assignee_names") or "";
    inventors = row.get("inventor_names") or "";
    if assignees:
        result["known_applicants"] = [n.strip() for n in assignees.split("|") if n.strip()];
    if inventors:
        result["known_inventors"] = [n.strip() for n in inventors.split("|") if n.strip()];
    return result;


def extract_sections(text: str) -> set[int]:
    return {int(m) for m in _SECTION_RE.findall(text) if 10 <= int(m) <= _MAX_WIPO_SECTION};


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
      (72) found AND no higher section -> section continues, fetch next page
      (72) absent AND higher section present -> section absent from patent, stop
      (72) absent AND no higher section -> not yet reached, fetch next page
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
