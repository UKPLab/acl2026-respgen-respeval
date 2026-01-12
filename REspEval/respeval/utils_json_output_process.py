import json
import logging
import regex as re
import json5

def _escape_invalid_backslashes(s: str) -> str:
    # Replace a single backslash that is not part of a valid JSON escape
    # (?<!\\)   : not preceded by a backslash (so we don't touch \\)
    # \\        : the backslash itself
    # (?!["\\/bfnrtu]) : next char is NOT one of the valid escape starters
    return re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', s)

# ---------- comma-related fixes ----------

def _fix_missing_comma_text_labels(s: str) -> str:
    """
    Insert a missing comma between a completed `"text": "..."` value and a following `"labels":`
    key, which is a very common LLM output error.
    """
    # Match a JSON string value for "text" (handles escaped quotes) followed by optional whitespace
    # and immediately "labels": without a comma.
    return re.sub(
        r'("text"\s*:\s*"(?:\\.|[^"\\])*")\s*(?="labels"\s*:)',
        r'\1, ',
        s,
        flags=re.DOTALL
    )

def _remove_trailing_commas(s: str) -> str:
    """
    Remove trailing commas before a closing } or ].
    Examples:
      {"a": 1,}      -> {"a": 1}
      [1, 2, ]       -> [1, 2]
    """
    # Collapse multiple commas before a closer, then drop the last one.
    s = re.sub(r',\s*,\s*([}\]])', r',\1', s)     # , ,} -> ,}
    s = re.sub(r',\s*([}\]])', r'\1', s)          # ,}   -> }
    return s

def _fix_missing_commas_common_pairs(s: str) -> str:
    """
    Insert missing commas for a few frequent key sequences besides text->labels.
    Safe, targeted lookaheads (we only add a comma *before* the listed keys).
    Add keys if you see more cases downstream.
    """
    keys = (
        "labels",
        "response",
        "requests",
        "criticisms",
        "review_text",
        "response_conv_score",
        "response_spec_score",
        "questions",
        "other_responses",
    )
    for k in keys:
        # Insert comma if any value (string, number, object, array, true/false/null) is
        # immediately followed by `"k":` without a comma in between.
        # We keep this conservative by only adding before the specific keys above.
        s = re.sub(
            rf'((?:"(?:\\.|[^"\\])*")|\d+(?:\.\d+)?|true|false|null|\}}|\]])\s*(?="{k}"\s*:)',
            r'\1, ',
            s,
            flags=re.IGNORECASE | re.DOTALL
        )
    return s

def _comma_repairs(s: str) -> str:
    """Apply a sequence of conservative comma repairs."""
    s = _fix_missing_comma_text_labels(s)
    s = _fix_missing_commas_common_pairs(s)
    s = _remove_trailing_commas(s)
    return s

def _strip_json_comments(s: str) -> str:
    """Remove //, /*...*/, and # comments outside strings."""
    out = []
    i, n = 0, len(s)
    in_str = False
    esc = False
    while i < n:
        ch = s[i]

        if in_str:
            out.append(ch)
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue

        if ch == '"':
            in_str = True
            out.append(ch); i += 1; continue

        # // line comment
        if ch == '/' and i+1 < n and s[i+1] == '/':
            i += 2
            while i < n and s[i] not in '\r\n':
                i += 1
            continue

        # /* block comment */
        if ch == '/' and i+1 < n and s[i+1] == '*':
            i += 2
            while i+1 < n and not (s[i] == '*' and s[i+1] == '/'):
                i += 1
            i += 2 if i+1 < n else 1
            continue

        # # line comment
        if ch == '#':
            i += 1
            while i < n and s[i] not in '\r\n':
                i += 1
            continue

        out.append(ch)
        i += 1
    return ''.join(out)

# ---------- main loader ----------

def robust_json_loads(s: str):
    t = s.strip()

    # Strip common Markdown fences if present
    if t.startswith("```"):
        t = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", t, flags=re.S)

    # Sanitize backslashes first (so regexes don't get confused by broken escapes)
    t = _escape_invalid_backslashes(t)

    # First pass: try as-is
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(t)
    except Exception:
        pass

    # Second pass:  Remove comments and try again
    t = _strip_json_comments(t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(t)
    except Exception:
        pass

    # Third pass: comma repairs on the full string
    t2 = _comma_repairs(t)
    try:
        return json.loads(t2)
    except json.JSONDecodeError:
        pass
    try:
        return json5.loads(t)
    except Exception:
        pass

    # Fourth pass: extract the largest top-level {...} block and repair that
    m = re.search(r"\{(?:[^{}]|(?R))*\}", t, flags=re.S)  # recursive-ish via PCRE-style; Python re ignores (?R)
    if not m:
        # Fallback: non-recursive greedy as in your original code
        m = re.search(r"\{.*\}", t, flags=re.S)

    if m:
        obj = m.group(0)
        obj = _escape_invalid_backslashes(obj)
        obj = _comma_repairs(obj)
        try:
            return json.loads(obj)
        except json.JSONDecodeError:
            pass
            
        try:
            return json5.loads(t)
        except Exception:
            pass

    logging.error("Failed to locate a JSON object in input.")
    raise json.JSONDecodeError("No JSON object could be decoded", t, 0)

def load_json_robust(text: str):
    # 1) Try strict JSON first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) Try JSON5 if available
    try:
        return json5.loads(text)
    except Exception:
        pass

    # 3) Last resort: your repair loader
    return robust_json_loads(text)