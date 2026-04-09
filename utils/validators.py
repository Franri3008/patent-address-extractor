"""Deterministic post-validation for LLM extraction results.

These checks run after LLM extraction with zero extra model calls.
They produce warning strings stored in metadata for debugging and
used as triggers for the vision verification fallback (4D).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

from models.llm.base import LLMResult

# ISO 3166-1 alpha-2 country codes (complete set).
_COUNTRY_CODES = {
    "AD", "AE", "AF", "AG", "AI", "AL", "AM", "AO", "AQ", "AR", "AS", "AT",
    "AU", "AW", "AX", "AZ", "BA", "BB", "BD", "BE", "BF", "BG", "BH", "BI",
    "BJ", "BL", "BM", "BN", "BO", "BR", "BS", "BT", "BV", "BW", "BY", "BZ",
    "CA", "CC", "CD", "CF", "CG", "CH", "CI", "CK", "CL", "CM", "CN", "CO",
    "CR", "CU", "CV", "CW", "CX", "CY", "CZ", "DE", "DJ", "DK", "DM", "DO",
    "DZ", "EC", "EE", "EG", "EH", "ER", "ES", "ET", "FI", "FJ", "FK", "FM",
    "FO", "FR", "GA", "GB", "GD", "GE", "GF", "GG", "GH", "GI", "GL", "GM",
    "GN", "GP", "GQ", "GR", "GS", "GT", "GU", "GW", "GY", "HK", "HM", "HN",
    "HR", "HT", "HU", "ID", "IE", "IL", "IM", "IN", "IO", "IQ", "IR", "IS",
    "IT", "JE", "JM", "JO", "JP", "KE", "KG", "KH", "KI", "KM", "KN", "KP",
    "KR", "KW", "KY", "KZ", "LA", "LB", "LC", "LI", "LK", "LR", "LS", "LT",
    "LU", "LV", "LY", "MA", "MC", "MD", "ME", "MF", "MG", "MH", "MK", "ML",
    "MM", "MN", "MO", "MP", "MQ", "MR", "MS", "MT", "MU", "MV", "MW", "MX",
    "MY", "MZ", "NA", "NC", "NE", "NF", "NG", "NI", "NL", "NO", "NP", "NR",
    "NU", "NZ", "OM", "PA", "PE", "PF", "PG", "PH", "PK", "PL", "PM", "PN",
    "PR", "PS", "PT", "PW", "PY", "QA", "RE", "RO", "RS", "RU", "RW", "SA",
    "SB", "SC", "SD", "SE", "SG", "SH", "SI", "SJ", "SK", "SL", "SM", "SN",
    "SO", "SR", "SS", "ST", "SV", "SX", "SY", "SZ", "TC", "TD", "TF", "TG",
    "TH", "TJ", "TK", "TL", "TM", "TN", "TO", "TR", "TT", "TV", "TW", "TZ",
    "UA", "UG", "UM", "US", "UY", "UZ", "VA", "VC", "VE", "VG", "VI", "VN",
    "VU", "WF", "WS", "YE", "YT", "ZA", "ZM", "ZW",
    # Common WIPO designations (not ISO countries but valid in patent context)
    "EP", "WO", "EA", "OA", "AP", "GC",
};

_COUNTRY_RE = re.compile(r"\(([A-Z]{2})\)\s*\.?\s*$");


def _normalise_name(name: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for fuzzy comparison."""
    name = re.sub(r"[^\w\s]", " ", name.lower());
    return re.sub(r"\s+", " ", name).strip();


def _names_match(a: str, b: str, threshold: float = 0.6) -> bool:
    """Check if two names are similar enough to be considered the same entity."""
    na, nb = _normalise_name(a), _normalise_name(b);
    if na == nb:
        return True;
    # One contained in the other (e.g. "KSM COMPONENT CO LTD" vs "KSM COMPONENT CO., LTD.")
    if na in nb or nb in na:
        return True;
    return SequenceMatcher(None, na, nb).ratio() >= threshold;


def validate_country_codes(entities: list[dict]) -> list[str]:
    """Check country codes in addresses against ISO 3166-1 alpha-2."""
    warnings: list[str] = [];
    for ent in entities:
        addr = ent.get("address") or "";
        m = _COUNTRY_RE.search(addr);
        if m:
            code = m.group(1);
            if code not in _COUNTRY_CODES:
                warnings.append(
                    f"Unknown country code '({code})' in address of '{ent.get('name', '?')}'"
                );
    return warnings;


def validate_entity_completeness(
    result: LLMResult,
    known_applicants: list[str] | None = None,
    known_inventors: list[str] | None = None,
) -> list[str]:
    """Flag known entities from BigQuery that are missing from the extraction."""
    warnings: list[str] = [];

    if known_applicants:
        extracted_names = [a.get("name", "") for a in result.applicants];
        for known in known_applicants:
            if not any(_names_match(known, ext) for ext in extracted_names):
                warnings.append(f"Missing known applicant: '{known}'");

    if known_inventors:
        extracted_names = [i.get("name", "") for i in result.inventors];
        for known in known_inventors:
            if not any(_names_match(known, ext) for ext in extracted_names):
                warnings.append(f"Missing known inventor: '{known}'");

    return warnings;


def validate_sections_consistency(
    result: LLMResult,
    ocr_sections: set[int] | None = None,
) -> list[str]:
    """Check that LLM-reported sections are consistent with OCR-detected sections."""
    if not ocr_sections:
        return [];

    warnings: list[str] = [];
    llm_sections = set();
    for s in result.sections_detected:
        # Parse "(72)" -> 72
        m = re.match(r"\((\d+)\)", s);
        if m:
            llm_sections.add(int(m.group(1)));

    # LLM claims sections OCR never found
    for s in llm_sections - ocr_sections:
        if s in (71, 72, 74):  # Only flag address-relevant sections
            warnings.append(f"LLM reports section ({s}) but OCR did not detect it");

    # OCR found address sections that LLM didn't report
    for s in ocr_sections - llm_sections:
        if s in (71, 72, 74):
            warnings.append(f"OCR detected section ({s}) but LLM did not report it");

    return warnings;


def run_all_validations(
    result: LLMResult,
    known_applicants: list[str] | None = None,
    known_inventors: list[str] | None = None,
    ocr_sections: set[int] | None = None,
) -> list[str]:
    """Run all validation checks and return combined warnings."""
    warnings: list[str] = [];
    all_entities = result.inventors + result.applicants + result.agents;
    warnings.extend(validate_country_codes(all_entities));
    warnings.extend(validate_entity_completeness(result, known_applicants, known_inventors));
    warnings.extend(validate_sections_consistency(result, ocr_sections));
    return warnings;
