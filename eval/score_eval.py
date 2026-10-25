"""
eval/score_eval.py

Automatic pass/fail scoring of eval_results.json. It checks behaviour, not
factual accuracy (that still needs the hand-graded scorecard.md):
  - should_decline: the answer is one of the assistant's decline messages
  - should_answer:  the answer is not a decline, and the expected source
                    document was among the retrieved sources

Run with: python eval/score_eval.py   (exits 1 if below the thresholds)
"""

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate import OUT_OF_SCOPE_MESSAGE

RESULTS_FILE = Path(__file__).parent / "eval_results.json"

INSUFFICIENT_CONTEXT_PHRASE = "I don't have enough information in my documents"

# Declining correctly is the core guarantee, so it must be perfect; answering is
# allowed a little slack because LLM output varies slightly between runs.
MIN_DECLINE_PASS_RATE = 1.0
MIN_ANSWER_PASS_RATE = 0.9


def is_decline(answer):
    return answer.strip() == OUT_OF_SCOPE_MESSAGE or INSUFFICIENT_CONTEXT_PHRASE in answer


def score_case(result):
    declined = is_decline(result["actual_answer"])
    if result["type"] == "should_decline":
        return declined, "declined" if declined else "answered instead of declining"

    if declined:
        return False, "declined a question it should answer"
    if result["expected_source"] not in result["actual_sources"]:
        return False, f"expected source {result['expected_source']} not retrieved"
    return True, "answered from expected source"


def score(results):
    """Returns (passed_thresholds, report_lines)."""
    lines = []
    totals = {"should_answer": [0, 0], "should_decline": [0, 0]}  # type -> [passed, total]

    for result in results:
        ok, reason = score_case(result)
        totals[result["type"]][0] += ok
        totals[result["type"]][1] += 1
        lines.append(f"{'PASS' if ok else 'FAIL'}  #{result['id']:<3} {result['type']:<15} {reason}")

    def rate(kind):
        passed, total = totals[kind]
        return passed / total if total else 1.0

    lines.append("")
    for kind, minimum in (("should_answer", MIN_ANSWER_PASS_RATE), ("should_decline", MIN_DECLINE_PASS_RATE)):
        passed, total = totals[kind]
        lines.append(f"{kind}: {passed}/{total} ({rate(kind):.0%}), required >= {minimum:.0%}")

    ok = rate("should_answer") >= MIN_ANSWER_PASS_RATE and rate("should_decline") >= MIN_DECLINE_PASS_RATE
    return ok, lines


def main():
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    ok, lines = score(results)
    print("\n".join(lines))
    print("\nEVAL PASSED" if ok else "\nEVAL FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
