"""Typed answer envelopes and deterministic citation provenance checks.

An exact quote proves only that text occurs in a retrieved document. It does not
prove that an answer follows from the quote, that the document is true, or that
the model actually used the evidence. Those require separate evaluations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


MAX_ANSWER_CHARS = 24_000
MAX_CITATIONS = 64
MAX_QUOTE_CHARS = 8_000


@dataclass(frozen=True)
class EvidenceDocument:
    id: str
    title: str
    text: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    document_id: str
    quote: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    status: str
    answer: str
    citations: list[Citation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_answer(payload: dict[str, Any]) -> Answer:
    """Parse the strict wire schema; never coerce types or discard extra fields.

    Shape validation is separate from provenance validation. A structurally valid
    answer can still contain a fabricated citation and fail validate_answer().
    Error messages are fixed codes, so untrusted answer text is not leaked.
    """
    if not isinstance(payload, dict) or set(payload) != {"status", "answer", "citations"}:
        raise ValueError("answer_schema_fields")
    if not isinstance(payload["status"], str) or payload["status"] not in {"answer", "abstain"}:
        raise ValueError("answer_status")
    if not isinstance(payload["answer"], str):
        raise ValueError("answer_text_type")
    if len(payload["answer"]) > MAX_ANSWER_CHARS:
        raise ValueError("answer_text_too_large")
    if not isinstance(payload["citations"], list) or len(payload["citations"]) > MAX_CITATIONS:
        raise ValueError("answer_citations_shape")
    citations = []
    for raw in payload["citations"]:
        if not isinstance(raw, dict) or set(raw) != {"document_id", "quote"}:
            raise ValueError("citation_schema_fields")
        if not isinstance(raw["document_id"], str) or not isinstance(raw["quote"], str):
            raise ValueError("citation_field_type")
        if len(raw["quote"]) > MAX_QUOTE_CHARS:
            raise ValueError("citation_quote_too_large")
        citations.append(Citation(raw["document_id"], raw["quote"]))
    return Answer(payload["status"], payload["answer"], citations)


def document_errors(documents: list[EvidenceDocument]) -> list[str]:
    """Validate the trusted retrieval envelope, including unambiguous IDs."""
    if not isinstance(documents, list):
        return ["documents_must_be_list"]
    errors: list[str] = []
    seen: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, EvidenceDocument):
            errors.append(f"document[{index}]:invalid_type")
            continue
        if not all(isinstance(value, str) for value in (document.id, document.title, document.text)):
            errors.append(f"document[{index}]:invalid_fields")
            continue
        if not document.id.strip():
            errors.append(f"document[{index}]:empty_id")
        elif document.id in seen:
            errors.append(f"document[{index}]:duplicate_id")
        seen.add(document.id)
    return errors


def validate_answer(answer: Answer, documents: list[EvidenceDocument]) -> dict[str, Any]:
    """Check the answer schema and citation membership/exact substrings.

    `valid` means this mechanical contract passed, not that the prose is correct.
    Abstention is a distinct envelope: empty answer text and zero citations. The
    validator does not infer whether an abstention was justified by the query.
    """
    checks: dict[str, Any] = {
        "schema": False,
        "documents": False,
        "answer_text": False,
        "citation_presence": False,
        "citation_documents": False,
        "quote_match": False,
        "abstention_shape": False,
        "entailment": "not_checked",
        "source_veracity": "not_checked",
        "causal_grounding": "not_checked",
    }
    errors = document_errors(documents)
    checks["documents"] = not errors
    try:
        if not isinstance(answer, Answer):
            raise ValueError("answer_must_be_typed")
        if not isinstance(answer.citations, list) or any(not isinstance(c, Citation) for c in answer.citations):
            raise ValueError("citations_must_be_typed")
        parsed = parse_answer(answer.to_dict())
    except (TypeError, ValueError) as exc:
        errors.append(str(exc))
        return {"valid": False, "errors": errors, "checks": checks}
    checks["schema"] = True
    if parsed.status == "abstain":
        shape_ok = parsed.answer == "" and parsed.citations == []
        checks["abstention_shape"] = shape_ok
        checks["answer_text"] = parsed.answer == ""
        checks["citation_presence"] = not parsed.citations
        checks["citation_documents"] = not parsed.citations
        checks["quote_match"] = not parsed.citations
        if not shape_ok:
            errors.append("abstention_must_have_empty_answer_and_citations")
        return {"valid": not errors, "errors": errors, "checks": checks}

    checks["abstention_shape"] = True  # Not applicable to an answer envelope.
    checks["answer_text"] = bool(parsed.answer.strip())
    checks["citation_presence"] = bool(parsed.citations)
    if not checks["answer_text"]:
        errors.append("answer_text_required")
    if not checks["citation_presence"]:
        errors.append("answer_citation_required")
    if not checks["documents"]:
        # Duplicate IDs must not be resolved by choosing an arbitrary document.
        return {"valid": False, "errors": errors, "checks": checks}

    by_id = {document.id: document for document in documents}
    checks["citation_documents"] = bool(parsed.citations)
    checks["quote_match"] = bool(parsed.citations)
    for index, citation in enumerate(parsed.citations):
        document = by_id.get(citation.document_id)
        if document is None:
            errors.append(f"citation[{index}]:document_not_retrieved")
            checks["citation_documents"] = False
            checks["quote_match"] = False
        if not citation.quote.strip():
            errors.append(f"citation[{index}]:empty_quote")
            checks["quote_match"] = False
        elif document is not None and citation.quote not in document.text:
            errors.append(f"citation[{index}]:quote_not_exact_substring")
            checks["quote_match"] = False
    return {"valid": not errors, "errors": errors, "checks": checks}
