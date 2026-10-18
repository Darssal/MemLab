from __future__ import annotations

from memlab.metrics.summary_builder import build_acceptance_result
from memlab.runner_core import compute_run_summary


def test_compute_run_summary_emits_total_denominator_accuracy() -> None:
    tasks = [
        {
            "status": "success",
            "strict_score": 1.0,
            "relaxed_score": 1.0,
            "accuracy": 1.0,
            "parse_failure": False,
            "no_answer": False,
            "queue_wait_ms": 1.0,
            "processing_ms": 10.0,
            "proc_whole_ms": 10.0,
            "proc_llm_ms": 7.0,
            "proc_non_llm_ms": 3.0,
            "end_to_end_ms": 11.0,
            "latency_ms": 11.0,
            "task_enqueued_ts": 1000,
            "task_end_ts": 1011,
            "llm_latency_ms": 7.0,
            "context_injected_tokens": 0,
            "context_injected_items": 0,
        },
        {
            "status": "error",
            "strict_score": 0.0,
            "relaxed_score": 0.0,
            "accuracy": 0.0,
            "parse_failure": False,
            "no_answer": False,
            "queue_wait_ms": 2.0,
            "processing_ms": 12.0,
            "proc_whole_ms": 12.0,
            "proc_llm_ms": 8.0,
            "proc_non_llm_ms": 4.0,
            "end_to_end_ms": 13.0,
            "latency_ms": 13.0,
            "task_enqueued_ts": 1000,
            "task_end_ts": 1013,
            "llm_latency_ms": 8.0,
            "context_injected_tokens": 0,
            "context_injected_items": 0,
        },
    ]
    summary = compute_run_summary(
        run_id="r-test",
        variant_id="F1",
        workload_id="E2E-UNIT",
        n_sessions=2,
        tasks=tasks,
        mem_ops=[],
    )
    assert summary["strict_accuracy"] == 1.0
    assert summary["strict_accuracy_on_total"] == 0.5
    assert summary["strict_correct_total"] == 1


def test_acceptance_prefers_strict_accuracy_on_total() -> None:
    rows = [
        {
            "variant_id": "B1",
            "workload_id": "E2E-UNIT",
            "prompt_mode": "memory_neutral",
            "tasks_path": "workloads/e2e/tasks.jsonl",
            "processing_p95_ms": 1000.0,
            "usage_unavailable": True,
            "strict_accuracy": 0.20,
            "strict_accuracy_on_total": 0.08,
        },
        {
            "variant_id": "F1",
            "workload_id": "E2E-UNIT",
            "prompt_mode": "memory_neutral",
            "tasks_path": "workloads/e2e/tasks.jsonl",
            "processing_p95_ms": 1000.0,
            "usage_unavailable": True,
            "strict_accuracy": 0.10,
            "strict_accuracy_on_total": 0.08,
        },
    ]
    result = build_acceptance_result(rows, min_p95_improve=0.01, min_prompt_drop=0.01, max_accuracy_drop=0.03)
    assert result["accuracy_metric"] == "strict_accuracy_on_total"
    assert abs(float(result["accuracy_drop"])) < 1e-9
