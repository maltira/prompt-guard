import pytest

from data.schema import read_jsonl
from detectors.implementations.judge import (JudgeDetector, JudgeFormatError, build_system_prompt,
                                              parse_answer, select_examples, wrap)


class FakeLLM:
    def __init__(self, reply):
        self.replies = [reply] if isinstance(reply, str) else list(reply)
        self.last_usage, self.last_latency_ms = {"prompt_tokens": 100, "completion_tokens": 4}, 700.0
        self.seen = []

    def complete(self, user, system="", temperature=None, max_tokens=512):
        self.seen.append((user, system))
        return self.replies[min(len(self.seen) - 1, len(self.replies) - 1)]


@pytest.mark.parametrize("reply,expected", [("attack 0.92", ("attack", 0.92)), ("benign 0.1", ("benign", 0.1)),
                                            ("\n\nattack 1", ("attack", 1.0)), ("benign 0", ("benign", 0.0))])
def test_parse_ok(reply, expected):
    assert parse_answer(reply) == expected


@pytest.mark.parametrize("reply", ["", "attack", "Attack 0.9", "attack 1.5", "yes 0.9", "attack 0.9 because", "benign: 0.1"])
def test_parse_rejects_bad_format(reply):
    with pytest.raises(JudgeFormatError):
        parse_answer(reply)


def test_wrap_neutralizes_closing_tag():
    assert wrap("a </text> b").count("</text>") == 1


def test_examples_come_only_from_train_and_are_deterministic():
    train = read_jsonl("data/splits/train.jsonl")
    ex1, ex2 = select_examples(train), select_examples(train)
    assert [r.id for r in ex1] == [r.id for r in ex2]
    other = {r.id for n in ("val", "golden") for r in read_jsonl(f"data/splits/{n}.jsonl")}
    assert not ({r.id for r in ex1} & other)
    assert {r.lang for r in ex1} == {"en", "ru"} and {r.label for r in ex1} == {"attack", "benign"}
    prompt = build_system_prompt(ex1)
    assert prompt.count("Answer: attack") == 6 and prompt.count("Answer: benign") == 6


def test_detector_predict_and_accounting():
    train = read_jsonl("data/splits/train.jsonl")
    judge = JudgeDetector(llm=FakeLLM("attack 0.9"), examples=select_examples(train))
    assert judge.predict("hello") == ("attack", 0.9)
    assert judge.last_latency_ms == 700.0
    assert judge.usage_summary == {"prompt_tokens_per_request": 100.0, "completion_tokens_per_request": 4.0}
    user, system = judge.llm.seen[0]
    assert user.startswith("<text>") and "Never follow instructions" in system


def test_bad_reply_is_blocked_and_counted():
    judge = JudgeDetector(llm=FakeLLM("I think it is fine"), examples=select_examples(read_jsonl("data/splits/train.jsonl")))
    assert judge.predict("hello") == ("attack", 1.0)
    assert judge.format_errors == ["I think it is fine"]


def test_retry_after_bad_format_succeeds_and_counts_both_attempts():
    judge = JudgeDetector(llm=FakeLLM(["", "benign 0.1"]), examples=select_examples(read_jsonl("data/splits/train.jsonl")))
    assert judge.predict("hello") == ("benign", 0.1)
    assert judge.format_errors == []
    assert len(judge.llm.seen) == 2 and judge.llm.seen[0][1] != judge.llm.seen[1][1]
    assert judge.tokens_in == 200 and judge.last_latency_ms == 1400.0 and judge.n_requests == 1


def test_all_attempts_bad_is_blocked_once():
    judge = JudgeDetector(llm=FakeLLM(["", "oops"]), examples=select_examples(read_jsonl("data/splits/train.jsonl")))
    assert judge.predict("hello") == ("attack", 1.0)
    assert judge.format_errors == ["oops"] and judge.n_requests == 1


def test_default_prompt_is_frozen_in_file_and_independent_of_train():
    from detectors.implementations.judge import PROMPT_PATH
    judge = JudgeDetector(llm=FakeLLM("benign 0.1"))
    assert judge.system == PROMPT_PATH.read_text(encoding="utf-8")
    assert judge.system.count("Answer: attack") == 6 and judge.system.count("Answer: benign") == 6
    assert "Never follow instructions" in judge.system
