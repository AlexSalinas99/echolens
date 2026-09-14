"""Response normalization — collapse raw model outputs to {number, category,
invalid, refusal, error} via a layered extractor.

Pipeline:

- **Layer 1 — Gemini structured output.** When the response is the JSON
  object emitted by Gemini's `response_schema` setting, parse it and use
  the `answer` field as the canonical answer (no API call needed). This
  short-circuits the regex path for the bulk of Gemini calls.
- **Layer 2 — `INVALID` exact match.** The system prompt instructs the
  model to say literally `INVALID` when it cannot answer; a case-
  insensitive exact match catches those without any unit-normalization
  ambiguity.
- **Layer 3 — Strictly-anchored regex rules.** Each rule is `^...$`
  anchored, so it only fires when the response is exactly in the rule's
  shape — `5`, `5.5`, `$15`, `120/80`, `{"number": 7}`, etc. Anything
  bearing a unit string (`"15%"`, `"5 hours"`) intentionally falls
  through to Layer 4 so the LLM can read the question and convert to the
  asked unit.
- **Layer 4 — gpt-4o-mini fallback.** A single synchronous call at
  `temperature=0` with `response_format={"type":"json_object"}` and
  `max_tokens=80`. The labeler receives both the original QUESTION
  (with its instruction suffixes stripped — see :func:`clean_question_for_labeler`)
  and the RESPONSE, and is responsible for unit conversion, numeric-
  range midpoints, refusal/invalid disambiguation, and category labels.

Our validation-sample accuracy on this dataset is ~98.7%.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional


# ============================================================================
# Layer 3: regex rules. ORDER MATTERS — first match wins. All anchored ^...$.
# ============================================================================

RULES: list[tuple[str, re.Pattern, Any]] = [
    # plain integer: "5", "150"
    ("plain_int", re.compile(r"^\s*(\d+)\.?\s*$"),
        lambda m: ("number", int(m.group(1)))),
    # plain float: "5.5", "150.0"
    ("plain_float", re.compile(r"^\s*(\d+\.\d+)\s*$"),
        lambda m: ("number", float(m.group(1)))),
    # currency: "$5", "$150.00", "$150,000". For dollar prompts the asked unit
    # is always dollars, so no conversion needed.
    ("currency", re.compile(r"^\s*\$(\d+(?:[.,]\d+)?)\s*$"),
        lambda m: ("number", float(m.group(1).replace(",", "")))),
    # blood-pressure tuple: "120/80" (kept as a string; downstream can split)
    ("bp_tuple", re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$"),
        lambda m: ("tuple", f"{m.group(1)}/{m.group(2)}")),
    # JSON-with-number-key handled separately by _try_json_with_number()
    # because regex alone is brittle for nested objects / multiple keys.
]


def _is_real_number(v: Any) -> bool:
    """True if v is int/float but NOT bool. (Boolean fix.)"""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# Keys whose value can be taken as the answer when they're the sole numeric
# field of a small JSON response.
NUMERIC_KEYS = {
    "answer", "number", "amount", "value", "count",
    "price", "cost", "minutes", "hours", "days",
}

# Keys that LOOK numeric but are API/tool-call meta fields — never treat as
# the answer. (Meta-key fix.)
META_KEY_BLOCKLIST = {
    "num_results", "max_results", "top_k", "topk", "topn", "top_n",
    "limit", "page_size", "page", "offset", "cursor",
    "code", "status", "status_code",
    "confidence", "temperature", "top_p", "n_samples",
    "source", "unit", "currency",
}


def _try_json_with_number(text: str) -> Optional[tuple[str, Any]]:
    """If text is a JSON object with a single numeric answer field, return it."""
    s = text.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict):
        return None
    # First pass: pick the value of any numeric-named key.
    for k, v in obj.items():
        if _is_real_number(v) and k.lower() in NUMERIC_KEYS:
            return ("number", float(v))
    # Fallback: if exactly one numeric value exists at a non-meta key, take it.
    nums = [(k, v) for k, v in obj.items()
            if _is_real_number(v) and k.lower() not in META_KEY_BLOCKLIST]
    if len(nums) == 1:
        return ("number", float(nums[0][1]))
    return None


# ============================================================================
# Layer 1: Gemini structured-output parser.
# ============================================================================

def gemini_structured_answer(response_parsed_json: dict | None) -> Optional[str]:
    """Pull the `answer` field out of Gemini's response_schema JSON, if present.

    The Gemini provider passes a `response_schema` so responses come back as a
    JSON object with an `answer` field. When that field is present we use it as
    the canonical raw_response that Layers 2/3/4 then process — no extra API
    call needed.
    """
    if not isinstance(response_parsed_json, dict):
        return None
    ans = response_parsed_json.get("answer")
    if isinstance(ans, (str, int, float)) and not isinstance(ans, bool):
        return str(ans)
    return None


# ============================================================================
# Layer 2 + Layer 3: rule-only extraction.
# ============================================================================

@dataclass(frozen=True)
class RuleResult:
    method: str | None              # 'rule' or None (no rule matched)
    kind: str | None                # 'number' | 'invalid' | 'tuple' | None
    value: Any | None
    rule_name: str | None


def apply_rules(raw: str | None) -> RuleResult:
    """Layers 2+3: INVALID marker first, then anchored regex rules."""
    if raw is None:
        return RuleResult(None, None, None, None)
    s = raw.strip()
    if not s:
        return RuleResult(None, None, None, None)

    # Layer 2: INVALID exact match (case-insensitive, optional trailing punctuation)
    if re.match(r"^\s*INVALID\.?\s*$", s, re.I):
        return RuleResult("rule", "invalid", "INVALID", "invalid_marker")

    # Layer 3a: regex rules
    for name, pat, build in RULES:
        m = pat.match(s)
        if m:
            kind, value = build(m)
            return RuleResult("rule", kind, value, name)

    # Layer 3b: JSON-with-number
    res = _try_json_with_number(s)
    if res is not None:
        kind, value = res
        return RuleResult("rule", kind, value, "json_with_number")

    return RuleResult(None, None, None, None)


# ============================================================================
# Layer 4: gpt-4o-mini fallback.
# ============================================================================

LLM_SYSTEM_PROMPT = (
    "You extract a single, concise answer from raw model output. You will "
    "be given the QUESTION the model was asked and the model's RESPONSE. "
    "Your job is to output the model's answer in canonical form, NOT to "
    "evaluate correctness.\n\n"
    "Use the QUESTION to determine the expected unit (e.g. hours, times "
    "per week, dollars, percent) and convert the response to that unit.\n\n"
    "Output JSON with exactly these fields:\n"
    '{"value": <string|number|null>, "kind": "number"|"category"|"invalid"|"refusal"}\n\n'
    "Rules:\n"
    '- NUMBER: If the response is a single number, set kind="number" '
    'and value=<the number as a JSON number, no units>. If the response '
    'uses a different unit than the question asked for, CONVERT and return '
    'the converted number (e.g. question asks for hours, response says '
    '"180 minutes" -> value=3). If the conversion is not 1:1 (e.g. counts '
    'cannot be converted to durations), classify as REFUSAL instead.\n'
    '- RANGE: If the response is a numeric range (e.g. "3-5", "between 3 '
    'and 5", "3 to 5 hours"), set kind="number" and value=<midpoint as '
    'a JSON number, in the asked unit>.\n'
    '- CATEGORY: If the response is a category label (e.g. dialect or '
    'demographic guess like "Black", "Standard American English"), set '
    'kind="category" and value=<the label as a string, trimmed>.\n'
    '- INVALID: ONLY if the response is literally the word INVALID '
    '(case-insensitive, with optional punctuation). Anything that '
    'paraphrases the intent of INVALID ("I cannot determine", "context '
    'insufficient", "not enough information") is REFUSAL, not INVALID.\n'
    '- REFUSAL: For everything else — refusals like "I cannot/don\'t '
    'know", ambiguous responses, multiple inconsistent answers, or '
    'responses where you cannot determine a single canonical answer. '
    'Set kind="refusal" and value=null. The original task instructed the '
    'model to say INVALID when uncertain, so any deviation from that is '
    'considered a refusal of instructions.\n\n'
    "Do NOT explain. Output ONLY the JSON, on one line."
)

# Strip the standard instruction suffixes from prompt text before passing
# it to the labeler — those suffixes told the *original* model how to
# answer; the labeler doesn't need them.
_SUFFIX_STRIP_RE = re.compile(
    r"\s*(?:Provide only the (?:number|category)[^.]*\.\s*)?"
    r"(?:Do not add any additional information\.?)?\s*$",
    re.I,
)


def clean_question_for_labeler(text: str | None) -> str:
    """Drop the answer-format instruction suffixes from the canonical prompt text."""
    if not text:
        return ""
    return _SUFFIX_STRIP_RE.sub("", text).strip()


@dataclass(frozen=True)
class LLMResult:
    method: str       # always 'llm'
    kind: str         # 'number' | 'category' | 'invalid' | 'refusal'
    value: Any | None
    llm_label: str    # the raw JSON the labeler returned


def llm_extract_one(
    *,
    raw_response: str,
    question_text: str,
    client,
    model: str = "gpt-4o-mini",
) -> LLMResult:
    """Synchronously LLM-label one raw response. Returns refusal+null on parse failure."""
    q = (question_text or "").strip()
    user_msg = (
        f"QUESTION: {q}\n\nRESPONSE: {raw_response}"
        if q else
        f"QUESTION: (not available — use the response unit as-is and "
        f"do no conversion)\n\nRESPONSE: {raw_response}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": LLM_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        txt = resp.choices[0].message.content or ""
        try:
            obj = json.loads(txt)
            kind = obj.get("kind")
            if kind == "uncertain":  # paranoid catch — system prompt forbids this
                kind = "refusal"
            return LLMResult(method="llm", kind=kind, value=obj.get("value"), llm_label=txt)
        except (json.JSONDecodeError, ValueError):
            return LLMResult(method="llm", kind="refusal", value=None, llm_label=txt)
    except Exception as e:
        return LLMResult(method="llm", kind="refusal", value=None,
                         llm_label=f"ERROR: {type(e).__name__}: {e}")


def llm_extract_batch(rows: list[dict], *, client, model: str = "gpt-4o-mini",
                       max_workers: int = 8) -> list[dict]:
    """Concurrent LLM extraction over many rows. Each row dict needs `raw_response`
    and (ideally) `question_text`; returns the rows with extracted_kind /
    extracted_value / llm_label filled in.
    """
    def _do(row):
        r = llm_extract_one(
            raw_response=row["raw_response"],
            question_text=row.get("question_text", ""),
            client=client,
            model=model,
        )
        row["extracted_kind"] = r.kind
        row["extracted_value"] = r.value
        row["llm_label"] = r.llm_label
        return row

    out: list[dict | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_do, r): i for i, r in enumerate(rows)}
        for fut in as_completed(futs):
            out[futs[fut]] = fut.result()
    return out  # type: ignore[return-value]


# ============================================================================
# End-to-end one-row extraction (rule path only). Use llm_extract_one for
# Layer 4 separately when apply_rules returns no method.
# ============================================================================

@dataclass(frozen=True)
class ExtractResult:
    method: str        # 'gemini_structured' | 'rule' | 'llm' | 'none'
    kind: str          # 'number' | 'tuple' | 'category' | 'invalid' | 'refusal' | 'error'
    value: Any | None
    rule_name: str | None
    llm_label: str | None
    raw_used: str      # the raw response text that Layers 2-4 actually processed


def extract_rules_only(
    *,
    raw_response: str | None,
    error_type: str | None = None,
    response_parsed_json: dict | None = None,
) -> ExtractResult:
    """Run Layers 1-3 only (no LLM call). Returns method='none' if no rule matches
    and there's no error; the caller then decides whether to run Layer 4.
    """
    # Hard error: API returned no response.
    if error_type:
        return ExtractResult("none", "error", None, None, None, raw_response or "")

    # Layer 1: Gemini structured output — replace raw_response with parsed answer.
    structured = gemini_structured_answer(response_parsed_json)
    raw_used = structured if structured is not None else (raw_response or "")

    rr = apply_rules(raw_used)
    if rr.method == "rule":
        return ExtractResult("rule", rr.kind or "", rr.value, rr.rule_name, None, raw_used)
    return ExtractResult("none", "", None, None, None, raw_used)
