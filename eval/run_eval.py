import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from scripts.generate import generate_answer

EVAL_FILE = Path(__file__).parent / "eval_questions.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"

with open(EVAL_FILE, "r") as f:
    eval_cases = json.load(f)

print(f"Loaded {len(eval_cases)} eval questions.\n")

results = []

for case in eval_cases:
    print(f"[{case['id']}/{len(eval_cases)}] {case['question']}")

    try:
        answer, chunks = generate_answer(case["question"])
        sources_used = sorted(set(c.metadata.get("source", "unknown") for c in chunks))
    except Exception as e:
        print(f"  -> FAILED: {e}")
        answer = f"[ERROR: {e}]"
        sources_used = []

    results.append({
        "id": case["id"],
        "question": case["question"],
        "type": case["type"],
        "expected_answer_summary": case["expected_answer_summary"],
        "expected_source": case["expected_source"],
        "actual_answer": answer,
        "actual_sources": sources_used,
    })

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  -> done, saved progress\n")

print(f"\nFinished. {len(results)} results saved to {RESULTS_FILE}")