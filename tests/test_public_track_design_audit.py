from __future__ import annotations

import json
from pathlib import Path

from memlab.analysis.public_track_design_audit import resolve_design_audit


def _write_prepared_dir(base_dir: Path, *, dataset: str, tasks: list[dict], docs: list[dict]) -> None:
    base_dir.mkdir(parents=True)
    (base_dir / "manifest.json").write_text(
        json.dumps({"dataset": dataset, "sample_count": len(tasks)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (base_dir / "tasks_internal.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in tasks) + "\n",
        encoding="utf-8",
    )
    (base_dir / "docs_internal.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in docs) + "\n",
        encoding="utf-8",
    )


def test_design_audit_blocks_non_memory_dependent_track_m(tmp_path: Path) -> None:
    base_dir = tmp_path / "prepared" / "longbench_v2"
    tasks = [
        {
            "task_id": "t1",
            "doc_id": "d1",
            "gold_answer": "A",
            "metadata": {"memory_needed": False, "answer_metric_hint": "exact_option"},
        },
        {
            "task_id": "t2",
            "doc_id": "d2",
            "gold_answer": "B",
            "metadata": {"memory_needed": False, "answer_metric_hint": "exact_option"},
        },
    ]
    docs = [{"doc_id": "d1", "body": "x"}, {"doc_id": "d2", "body": "y"}]
    _write_prepared_dir(base_dir, dataset="longbench_v2", tasks=tasks, docs=docs)

    report, anomalies = resolve_design_audit(
        base_dir=base_dir,
        dataset_id="longbench_v2",
        track_id="track-m:memory_beneficial",
        experiment_track="memory_exposure",
        prompt_mode="memory_beneficial",
        qa_support_mode="none",
        qa_prompt_profile="multiple_choice_letter",
        track_goal="memory_sensitive",
        min_memory_needed_ratio=0.5,
    )
    assert report["memory_needed_ratio"] == 0.0
    assert "NON_MEMORY_DEPENDENT_DATASET" in anomalies


def test_design_audit_blocks_direct_evidence_leak_on_memory_track(tmp_path: Path) -> None:
    base_dir = tmp_path / "prepared" / "locomo"
    tasks = [
        {
            "task_id": "t1",
            "doc_id": "d1",
            "gold_answer": "a bookmark",
            "metadata": {"memory_needed": True, "answer_metric_hint": "token_f1"},
        }
    ]
    docs = [{"doc_id": "d1", "body": "dialogue"}]
    _write_prepared_dir(base_dir, dataset="locomo", tasks=tasks, docs=docs)

    report, anomalies = resolve_design_audit(
        base_dir=base_dir,
        dataset_id="locomo",
        track_id="track-m:memory_beneficial",
        experiment_track="memory_exposure",
        prompt_mode="memory_beneficial",
        qa_support_mode="standard",
        qa_prompt_profile="short_answer",
        track_goal="memory_sensitive",
        min_memory_needed_ratio=0.5,
    )
    assert report["memory_needed_ratio"] == 1.0
    assert "DIRECT_EVIDENCE_LEAKAGE" in anomalies


def test_design_audit_allows_standard_track_with_mixed_memory_dataset(tmp_path: Path) -> None:
    base_dir = tmp_path / "prepared" / "squad_v11"
    tasks = [
        {
            "task_id": "t1",
            "doc_id": "d1",
            "gold_answer": "alpha",
            "metadata": {"memory_needed": False, "answer_metric_hint": "token_f1"},
        },
        {
            "task_id": "t2",
            "doc_id": "d2",
            "gold_answer": "beta",
            "metadata": {"memory_needed": True, "answer_metric_hint": "token_f1"},
        },
    ]
    docs = [{"doc_id": "d1", "body": "x"}, {"doc_id": "d2", "body": "y"}]
    _write_prepared_dir(base_dir, dataset="squad_v11", tasks=tasks, docs=docs)

    report, anomalies = resolve_design_audit(
        base_dir=base_dir,
        dataset_id="squad_v11",
        track_id="track-s:memory_neutral",
        experiment_track="mainline",
        prompt_mode="memory_neutral",
        qa_support_mode="standard",
        qa_prompt_profile="short_answer",
        track_goal="standard",
        min_memory_needed_ratio=0.5,
    )
    assert report["memory_needed_ratio"] == 0.5
    assert anomalies == []
