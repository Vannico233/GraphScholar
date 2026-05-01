import json
from typing import Any, Optional

from llm_api import call_llm


SELF_CHECK_SYSTEM_PROMPT = """
You are a strict reviewer for a graph-learning literature research assistant.

Check whether the draft answer:
1. answers the user's original question,
2. uses enough evidence,
3. mentions representative papers,
4. includes strengths, limitations, and insights,
5. clearly states uncertainty when evidence is weak.

Return valid JSON only with keys:
- passed: bool
- score: int from 0 to 100
- issues: list of short strings
- improved_answer: string

If the draft answer is already acceptable, keep improved_answer close to the draft.
If it is weak, revise it once. Do not invent papers or evidence.
"""


def run_rule_based_checks(
    question: str,
    rewritten_query: str,
    retrieved_evidence: dict[str, Any],
    draft_answer: str,
) -> dict[str, Any]:
    issues = []
    lowered = draft_answer.lower()

    if len(draft_answer.strip()) < 120:
        issues.append("answer_too_short")

    if "代表工作" not in draft_answer:
        issues.append("missing_representative_work")
    if "优点" not in draft_answer:
        issues.append("missing_strengths")
    if "局限" not in draft_answer:
        issues.append("missing_limitations")
    if "启发" not in draft_answer:
        issues.append("missing_insights")

    evidence_count = len(retrieved_evidence.get("retrieved_papers", [])) + len(
        retrieved_evidence.get("evidence_chunks", [])
    )
    if evidence_count < 2:
        issues.append("insufficient_evidence")

    query_tokens = [token for token in rewritten_query.lower().split() if len(token) >= 5]
    if query_tokens and not any(token in lowered for token in query_tokens[:4]):
        issues.append("weak_query_coverage")

    passed = len(issues) == 0
    score = max(0, 100 - len(issues) * 15)
    return {
        "passed": passed,
        "score": score,
        "issues": issues,
        "improved_answer": draft_answer,
    }


def self_check_answer(
    question: str,
    rewritten_query: str,
    retrieved_evidence: dict[str, Any],
    draft_answer: str,
    use_llm: bool = False,
    client: Optional[Any] = None,
) -> dict[str, Any]:
    rule_result = run_rule_based_checks(question, rewritten_query, retrieved_evidence, draft_answer)
    if rule_result["passed"] or not use_llm:
        return rule_result

    prompt = (
        f"Original question:\n{question}\n\n"
        f"Rewritten query:\n{rewritten_query}\n\n"
        f"Evidence:\n{json.dumps(retrieved_evidence, ensure_ascii=False, indent=2)}\n\n"
        f"Draft answer:\n{draft_answer}\n\n"
        "Review the draft and return corrected JSON."
    )
    raw = call_llm(
        prompt=prompt,
        system_prompt=SELF_CHECK_SYSTEM_PROMPT,
        temperature=0.1,
        max_tokens=1600,
        client=client,
    )

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "passed" in parsed:
            return parsed
    except Exception:
        pass

    fallback = dict(rule_result)
    fallback["issues"] = rule_result["issues"] + ["llm_self_check_parse_failed"]
    return fallback
