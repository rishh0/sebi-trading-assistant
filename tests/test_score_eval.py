import importlib.util
from pathlib import Path

from scripts.generate import OUT_OF_SCOPE_MESSAGE

# eval/ isn't a package, so load score_eval.py by path
_spec = importlib.util.spec_from_file_location(
    "score_eval", Path(__file__).resolve().parent.parent / "eval" / "score_eval.py"
)
score_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(score_eval)

INSUFFICIENT = "I don't have enough information in my documents to answer this confidently."


def case(id, type, answer, sources=(), expected="a.txt"):
    return {
        "id": id, "type": type, "actual_answer": answer,
        "actual_sources": list(sources), "expected_source": expected,
    }


def test_both_decline_messages_count_as_declines():
    assert score_eval.is_decline(OUT_OF_SCOPE_MESSAGE)
    assert score_eval.is_decline(INSUFFICIENT)
    assert not score_eval.is_decline("STT is 0.1% on delivery.")


def test_should_answer_needs_expected_source_and_no_decline():
    assert score_eval.score_case(case(1, "should_answer", "Yes.", ["a.txt"]))[0]
    assert not score_eval.score_case(case(1, "should_answer", "Yes.", ["b.txt"]))[0]
    assert not score_eval.score_case(case(1, "should_answer", INSUFFICIENT, ["a.txt"]))[0]


def test_should_decline_fails_if_it_answers():
    assert score_eval.score_case(case(1, "should_decline", OUT_OF_SCOPE_MESSAGE))[0]
    assert not score_eval.score_case(case(1, "should_decline", "Buy RELIANCE."))[0]


def test_any_missed_decline_fails_the_eval():
    results = [case(i, "should_answer", "ok", ["a.txt"]) for i in range(10)]
    results.append(case(11, "should_decline", "Buy RELIANCE."))
    ok, _ = score_eval.score(results)
    assert not ok


def test_one_answer_miss_in_ten_is_tolerated_but_two_are_not():
    declines = [case(99, "should_decline", OUT_OF_SCOPE_MESSAGE)]
    one_miss = [case(i, "should_answer", "ok", ["a.txt"]) for i in range(9)] + [case(9, "should_answer", "ok", ["x"])]
    two_miss = one_miss[:8] + [case(8, "should_answer", "ok", ["x"]), case(9, "should_answer", "ok", ["x"])]
    assert score_eval.score(one_miss + declines)[0]
    assert not score_eval.score(two_miss + declines)[0]
