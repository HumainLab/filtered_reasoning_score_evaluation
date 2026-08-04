#!/usr/bin/env python3
"""
Extract a HumanEval function body from a model response.

This is a standalone copy of the HumanEval branch in evaluation/parser.py
(extract_answer(..., data_name="humaneval")). Send this file to collaborators
who only need answer extraction, not the full eval stack.

HumanEval grading expects:
  - prompt: function signature + docstring (from the dataset)
  - generated_code: function BODY only (indented, no ``def`` line)
  - test_code + entry_point: from the dataset

Usage (library):

    from humaneval_extract_answer import extract_humaneval_answer

    body = extract_humaneval_answer(model_response)
    # body is ready to concatenate with prompt + tests in humaneval_check()

Usage (CLI):

    python humaneval_extract_answer.py --text "Here is my solution:\\n```python\\n..."
    python humaneval_extract_answer.py --file response.txt
    echo "$RESPONSE" | python humaneval_extract_answer.py

Source of truth in this repo: evaluation/parser.py (humaneval branch, ~661–1139).
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import List, Optional


def _fix_malformed_code(text: str) -> str:
    """Fix common malformed code issues (missing spaces around keywords/operators)."""
    if not text or len(text) < 5:
        return text

    text = re.sub(r"\breturn([A-Za-z_])", r"return \1", text)
    text = re.sub(r"\breturn(True|False|None)", r"return \1", text)
    text = re.sub(
        r"\b(if|for|while|def|class|elif|else|try|except|finally|with|async)([A-Za-z_])",
        r"\1 \2",
        text,
    )
    text = re.sub(
        r"([a-zA-Z0-9_\)\]\}])([<>=!+\-*/%]=+)([a-zA-Z0-9_\(\[\{])",
        r"\1 \2 \3",
        text,
    )
    text = re.sub(
        r"([a-zA-Z0-9_\)\]\}])([<>=!+\-*/%])([a-zA-Z0-9_\(\[\{])",
        r"\1 \2 \3",
        text,
    )
    text = re.sub(r"\b(for|while)\s+(\w+)(in)", r"\1 \2 \3", text)
    text = re.sub(r"\b(if|for|while|def|class|elif|return|with|assert)(\(+)", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z_][a-zA-Z0-9_]*)=([^=])", r"\1 = \2", text)
    return text


def _strip_assertions(text: str) -> str:
    return re.sub(
        r"assert\s+\w+\s*\([^)]*\)\s*==?[^=].*?(?=\n|$)",
        "",
        text,
        flags=re.MULTILINE,
    )


_SECTION_PATTERNS = [
    r"###?\s*Solution\s*(?:Code)?[:\s]*\n",
    r"###?\s*Code[:\s]*\n",
    r"###?\s*Implementation[:\s]*\n",
    r"###?\s*Answer[:\s]*\n",
    r"\*\*Solution[:\s]*\*\*\s*\n",
    r"^Solution:\s*\n",
    r"\nSolution:\s*\n",
]

_CODE_BLOCK_PATTERNS = [
    r"```python\s*\n(.*?)```",
    r"```\s*python\s*\n(.*?)```",
    r"```\n(.*?)```",
    r"```(.*?)```",
]


def _body_from_def_block(block: str) -> str:
    """Drop ``def`` line; HumanEval wants the body only."""
    lines = block.split("\n")
    if lines and lines[0].strip().startswith("def "):
        return "\n".join(lines[1:]).rstrip()
    return block.rstrip()


def _extract_from_section_headers(pred_str: str) -> Optional[str]:
    for section_pattern in _SECTION_PATTERNS:
        section_match = re.search(section_pattern, pred_str, re.IGNORECASE)
        if not section_match:
            continue
        after_header = pred_str[section_match.end() :]
        code_block_match = re.search(r"```(?:python)?\s*\n(.*?)```", after_header, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).rstrip()
        def_match = re.search(
            r"(def\s+\w+[^P]*)(?=Problem:|$)",
            after_header,
            re.DOTALL,
        )
        if def_match:
            return def_match.group(1).rstrip()
    return None


def _extract_from_code_blocks(pred_str: str) -> Optional[str]:
    for pattern in _CODE_BLOCK_PATTERNS:
        code_blocks = re.findall(pattern, pred_str, re.DOTALL)
        if not code_blocks:
            continue

        for block in code_blocks:
            block_clean = _strip_assertions(block)
            if re.search(r"\bdef\s+\w+\s*\(", block_clean):
                pred = block_clean.rstrip()
                return _body_from_def_block(pred)

        for block in code_blocks:
            block_clean = _strip_assertions(block)
            if re.search(r"\b(return|if|for|while)\b", block_clean) and not re.search(
                r"\bassert\b", block_clean
            ):
                return block_clean.rstrip()

        if code_blocks:
            last_block = code_blocks[-1]
            return _strip_assertions(last_block).rstrip()
    return None


def _extract_def_from_text(pred_str: str) -> Optional[str]:
    def_matches = list(
        re.finditer(r"^def\s+\w+\s*\([^)]*\)\s*(?:->.*?)?:", pred_str, re.MULTILINE)
    )
    if not def_matches:
        return None

    start_pos = def_matches[0].start()
    remaining = pred_str[start_pos:]
    next_def = re.search(r"\n(?=def\s+\w+)", remaining[10:])
    if next_def:
        remaining = remaining[: next_def.start() + 10]
    section_end = re.search(r"\n###|\n\*\*[A-Z]|\n## ", remaining)
    if section_end:
        remaining = remaining[: section_end.start()]
    if "\n\n\n" in remaining:
        remaining = remaining.split("\n\n\n")[0]
    return remaining.rstrip()


def _score_code_block(block: str) -> int:
    score = 0
    block_lower = block.lower()
    if "return" in block_lower:
        score += 10
    if any(kw in block_lower for kw in ["for ", "while ", "if "]):
        score += 5
    if "=" in block and "==" not in block[: block.index("=") + 10]:
        score += 3
    if any(phrase in block_lower for phrase in ["step", "note:", "example"]):
        score -= 5
    return score


def _extract_from_reasoning_blocks(pred_str: str) -> Optional[str]:
    lines = pred_str.split("\n")
    code_blocks: List[str] = []
    current_block: List[str] = []
    in_block = False
    base_indent: Optional[int] = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_block:
                current_block.append(line)
            continue

        indent = len(line) - len(line.lstrip())
        is_code_keyword = any(
            stripped.startswith(kw)
            for kw in [
                "return ",
                "if ",
                "for ",
                "while ",
                "elif ",
                "else:",
                "try:",
                "except",
                "with ",
                "break",
                "continue",
                "pass",
                "yield",
                "raise",
                "import ",
                "from ",
            ]
        )
        has_code_operators = any(
            op in stripped
            for op in [
                " = ",
                "==",
                "!=",
                "<=",
                ">=",
                "<",
                ">",
                " += ",
                " -= ",
                " *= ",
                " /= ",
                " in ",
                " not in ",
                " and ",
                " or ",
                " is ",
            ]
        )
        is_assignment = bool(re.match(r"^\s*\w+\s*=\s*[^=]", line))
        is_function_call = bool(re.search(r"\w+\s*\([^)]*\)", stripped))
        is_pseudocode = any(
            pattern in stripped.lower()
            for pattern in [
                "if x is empty",
                "if numbers is empty",
                "if list is empty",
                "for each",
                "for every",
                "if x is not empty",
            ]
        ) and not re.search(r"if\s+(not\s+)?\w+\s*(:|\bor\b|\band\b)", stripped.lower())
        is_description = any(
            phrase in stripped.lower()
            for phrase in [
                "step",
                "steps:",
                "we can",
                "we should",
                "we need",
                "note:",
                "however",
                "for example",
                "e.g.",
                "this means",
                "we are given",
                "we want",
                "the problem",
                "important:",
                "alternative",
                "but note",
                "example:",
                "->",
            ]
        )
        is_code = (
            is_code_keyword
            or is_assignment
            or (indent >= 4 and (has_code_operators or is_function_call))
            or (indent >= 4 and stripped.startswith("#"))
        )

        if is_code and not is_description and not is_pseudocode:
            if not in_block:
                in_block = True
                base_indent = indent
                current_block = [line]
            elif base_indent is not None and indent >= base_indent - 2:
                current_block.append(line)
                if indent > base_indent:
                    base_indent = indent
            else:
                if len(current_block) >= 3:
                    code_blocks.append("\n".join(current_block))
                current_block = [line]
                base_indent = indent
        elif in_block:
            if stripped.startswith("#"):
                current_block.append(line)
            elif is_description:
                if len(current_block) >= 3:
                    code_blocks.append("\n".join(current_block))
                current_block = []
                in_block = False
            elif base_indent is not None and indent < base_indent - 4:
                if len(current_block) >= 3:
                    code_blocks.append("\n".join(current_block))
                current_block = []
                in_block = False
            elif not stripped:
                current_block.append(line)
            elif base_indent is not None and indent >= base_indent - 2:
                current_block.append(line)
            else:
                if len(current_block) >= 3:
                    code_blocks.append("\n".join(current_block))
                current_block = []
                in_block = False

    if in_block and len(current_block) >= 3:
        code_blocks.append("\n".join(current_block))

    if not code_blocks:
        return None

    best_block = max(code_blocks, key=_score_code_block)
    lines_clean = [
        ln
        for ln in best_block.split("\n")
        if ln.strip()
        or (ln and any(l2.strip() for l2 in best_block.split("\n")[:5]))
    ]
    lines_filtered = []
    for line in lines_clean:
        stripped = line.strip()
        if not stripped:
            lines_filtered.append(line)
        elif not any(
            phrase in stripped.lower()
            for phrase in ["step", "note:", "however", "for example"]
        ):
            lines_filtered.append(line)
        elif any(kw in stripped for kw in ["return", "if ", "for ", "="]):
            lines_filtered.append(line)

    return "\n".join(lines_filtered).rstrip() if lines_filtered else None


def _extract_indented_fallback(pred_str: str) -> str:
    work_str = pred_str
    if "<|endoftext|>" in work_str:
        work_str = work_str.split("<|endoftext|>")[0]

    lines = work_str.split("\n")
    code_lines: List[str] = []
    in_code = False

    for line in lines:
        stripped = line.strip()
        if not in_code:
            if (line.startswith("    ") or line.startswith("\t")) and stripped:
                if any(
                    stripped.startswith(kw)
                    for kw in ["return ", "if ", "for ", "while ", "try:", "with "]
                ):
                    in_code = True
                    code_lines.append(line)
        else:
            if stripped == "" or line.startswith("    ") or line.startswith("\t"):
                code_lines.append(line)
            elif stripped.startswith("#"):
                code_lines.append(line)
            else:
                break

    return "\n".join(code_lines).rstrip() if code_lines else ""


def _clean_extracted_body(pred: str) -> str:
    lines = pred.split("\n")
    clean_lines: List[str] = []
    skip_until_colon = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("def "):
            if ":" in stripped:
                continue
            skip_until_colon = True
            continue
        if skip_until_colon:
            if ":" in stripped:
                skip_until_colon = False
            continue
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith(("List[", "Dict[", "Tuple[")):
            continue
        clean_lines.append(line)

    pred = "\n".join(clean_lines)
    pred = "\n".join(
        ln
        for ln in pred.split("\n")
        if not ln.strip().startswith(("import ", "from "))
    )

    lines = pred.split("\n")
    clean_lines = []
    in_docstring = False
    docstring_quotes: Optional[str] = None

    for line in lines:
        stripped = line.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                in_docstring = True
                docstring_quotes = stripped[:3]
                if stripped.count(docstring_quotes) >= 2:
                    in_docstring = False
                continue
            clean_lines.append(line)
        else:
            if docstring_quotes and docstring_quotes in stripped:
                in_docstring = False

    pred = "\n".join(clean_lines)
    pred = _strip_assertions(pred)

    lines = pred.split("\n")
    if lines and lines[0].strip().startswith("def "):
        pred = "\n".join(lines[1:]).strip()

    pred = _fix_malformed_code(pred)

    lines = pred.split("\n")
    non_empty = [ln for ln in lines if ln.strip()]
    if non_empty:
        indents = [len(ln) - len(ln.lstrip()) for ln in non_empty]
        min_indent = min(indents)
        normalized: List[str] = []
        for line in lines:
            if line.strip():
                current_indent = len(line) - len(line.lstrip())
                relative_indent = current_indent - min_indent
                normalized.append(" " * (4 + relative_indent) + line.lstrip())
            else:
                normalized.append(line)
        pred = "\n".join(normalized).rstrip()

    return pred.strip() if pred else ""


def extract_humaneval_answer(response: str) -> str:
    """
    Extract the HumanEval function body from a raw model response.

    Args:
        response: Full model output (may include reasoning, markdown fences, etc.)

    Returns:
        Function body as a string (indented Python, no ``def`` line).
        Empty string if no plausible code was found.

    The evaluator in this repo concatenates:
        prompt.rstrip() + "\\n" + body + "\\n" + test_code + "\\ncheck(entry_point)"
    See evaluation/grader.py ``humaneval_check``.
    """
    pred_str = (response or "").replace("\u043a\u0438", "")
    if not pred_str.strip():
        return ""

    if len(pred_str) < 500 and "\n" not in pred_str[:100]:
        pred_str = _fix_malformed_code(pred_str)

    pred_str_no_asserts = _strip_assertions(pred_str)
    if len(pred_str_no_asserts.strip()) > len(pred_str.strip()) * 0.3:
        pred_str = pred_str_no_asserts

    pred: Optional[str] = None
    for extractor in (
        _extract_from_section_headers,
        _extract_from_code_blocks,
        _extract_def_from_text,
        _extract_from_reasoning_blocks,
    ):
        pred = extractor(pred_str)
        if pred:
            break

    if pred is None:
        pred = _extract_indented_fallback(pred_str)

    if not pred:
        return ""

    return _clean_extracted_body(pred)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract HumanEval function body from a model response.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--text", "-t", help="Raw model response string")
    group.add_argument("--file", "-f", help="Path to a file containing the response")
    parser.add_argument(
        "--json-field",
        help="If --file is JSONL, read this field from each line (e.g. pred, code)",
    )
    args = parser.parse_args()

    if args.text is not None:
        print(extract_humaneval_answer(args.text))
        return 0

    if args.file is not None:
        with open(args.file, encoding="utf-8") as fh:
            if args.json_field:
                import json

                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    raw = row.get(args.json_field, "")
                    if isinstance(raw, list):
                        raw = raw[0] if raw else ""
                    print(extract_humaneval_answer(str(raw)))
            else:
                print(extract_humaneval_answer(fh.read()))
        return 0

    if not sys.stdin.isatty():
        print(extract_humaneval_answer(sys.stdin.read()))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
