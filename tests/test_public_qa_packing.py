from __future__ import annotations

from benchmarks.public_main.longmemeval.prepare import _resolve_longmemeval_profile
from memlab.runner_core import score_task_answer
from memlab.runner_e2e import (
    _build_qa_evidence_excerpt,
    _build_retrieval_shards,
    _qa_reasoning_hint,
    _question_focus_text,
    _resolve_task_qa_prompt_profile,
    _summarize_prompt,
)


def test_question_focus_text_drops_option_block() -> None:
    question = (
        "Which option best describes the author?\n\n"
        "Options:\nA. calm\nB. angry\nC. neutral\n\n"
        "Return only the option letter (A/B/C/D)."
    )
    assert _question_focus_text(question) == "Which option best describes the author?"


def test_qa_evidence_excerpt_keeps_tail_signal_for_long_context() -> None:
    body = (
        "irrelevant intro " * 200
        + "Sandra moved to the kitchen. "
        + "more filler " * 120
        + "Sandra went back to the bathroom."
    )
    excerpt = _build_qa_evidence_excerpt(
        body=body,
        question="Where is Sandra?",
        max_chars=480,
    )
    assert "bathroom" in excerpt.lower()


def test_qa_evidence_excerpt_prefers_latest_state_for_where_question() -> None:
    body = (
        ("background " * 40)
        + "John moved to the kitchen. "
        + ("more filler " * 30)
        + "John travelled to the bathroom. "
        + ("tail filler " * 20)
    )
    excerpt = _build_qa_evidence_excerpt(
        body=body,
        question="Where is John?",
        max_chars=320,
    )
    assert "bathroom" in excerpt.lower()
    if "kitchen" in excerpt.lower():
        assert excerpt.lower().find("bathroom") < excerpt.lower().find("kitchen")


def test_exact_span_benchmark_ignores_leading_article() -> None:
    scored = score_task_answer(
        raw_output='{"final_answer":"the office"}',
        parsed_answer="the office",
        parsed_answer_raw="the office",
        answer_key="office",
        task_type="qa",
        answer_metric_hint="exact_span",
    )
    assert scored["strict_score"] == 0.0
    assert scored["benchmark_metric"] == "exact_span"
    assert scored["benchmark_score"] == 1.0


def test_retrieval_shards_follow_dialogue_boundaries() -> None:
    body = (
        "1:00 pm on 1 Jan, 2024\nAlice: I bought a violin.\n\n"
        "1:05 pm on 1 Jan, 2024\nBob: Nice instrument.\n\n"
        "1:10 pm on 1 Jan, 2024\nAlice: I also bought sheet music.\n\n"
        "1:15 pm on 1 Jan, 2024\nBob: Practice every day.\n\n"
        "1:20 pm on 1 Jan, 2024\nAlice: I named the violin Aurora.\n\n"
        "1:25 pm on 1 Jan, 2024\nBob: Great name.\n"
    )
    shards = _build_retrieval_shards(body, shard_chars=120, max_shards=4)
    assert shards
    assert any("Alice: I named the violin Aurora." in shard["text"] for shard in shards)


def test_qa_evidence_excerpt_prefers_substantive_dialogue_blocks_over_greetings() -> None:
    body = (
        "1:00 pm\nJohn: Hey Tim!\n\n"
        "1:02 pm\nTim: Hey John!\n\n"
        "1:10 pm\nJohn: I finally got a signed basketball from the team store.\n\n"
        "1:15 pm\nTim: Nice, I also keep my signed basketball in a display case.\n\n"
        "1:20 pm\nJohn: That's an awesome collectible.\n"
    )
    excerpt = _build_qa_evidence_excerpt(
        body=body,
        question="What similar sports collectible do Tim and John own?",
        max_chars=480,
    )
    assert "signed basketball" in excerpt.lower()


def test_reasoning_hint_covers_difference_questions() -> None:
    hint = _qa_reasoning_hint(
        question="For my daily commute, how much more expensive was the taxi ride compared to the train fare?",
        qa_prompt_profile="short_answer",
    )
    assert "difference or comparison" in hint


def test_task_level_qa_profile_override_is_respected() -> None:
    profile = _resolve_task_qa_prompt_profile(
        task={"metadata": {"qa_prompt_profile": "summary"}},
        default_profile="short_answer",
    )
    assert profile == "summary"


def test_longmemeval_preference_questions_use_summary_profile() -> None:
    profile, metric = _resolve_longmemeval_profile(
        "Can you recommend a guide for a synthetic demonstration?",
        "single-session-preference",
    )
    assert profile == "summary"
    assert metric == "token_f1"


def test_longmemeval_factoid_questions_stay_short_answer() -> None:
    profile, metric = _resolve_longmemeval_profile(
        "What color is the example cube?",
        "single-session-user",
    )
    assert profile == "short_answer"
    assert metric == "relaxed_span"


def test_memory_beneficial_prompt_can_omit_direct_evidence() -> None:
    prompt, evidence_chars_used = _summarize_prompt(
        session_id="s1",
        question="What did the example agent create?",
        context_block="synthetic memory about a blue cube",
        body="The example agent created a blue cube.",
        task_type="qa",
        answer_field="final_answer",
        answer_format_mode="strict_json",
        prompt_mode="memory_beneficial",
        qa_prompt_profile="short_answer",
        qa_support_mode="none",
        max_prompt_tokens=2048,
        max_evidence_chars=600,
    )
    assert "[direct evidence omitted by qa_support_mode=none]" in prompt
    assert "[supportive direct evidence excerpt]" not in prompt
    assert evidence_chars_used == 0
