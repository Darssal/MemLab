from __future__ import annotations

from memlab.analysis.public_track_gate import _build_result


def test_public_track_gate_accepts_complete_consistent_rows() -> None:
    rows = []
    for variant, score in {
        "B0": 0.30,
        "B1": 0.32,
        "V1": 0.33,
        "F1": 0.35,
        "F2": 0.36,
    }.items():
        rows.append(
            {
                "variant": variant,
                "repeat_idx": 0,
                "prompt_mode": "memory_beneficial",
                "benchmark_score_on_total": score,
                "strict_accuracy_on_total": score,
                "partial_accuracy_on_total": score,
                "parse_failure_rate_on_total": 0.0,
                "no_answer_rate_on_total": 0.0,
                "success_rate": 1.0,
            }
        )
    result = _build_result(
        rows,
        dataset="longbench_v2",
        track_id="mainline:memory_beneficial",
        workload_id="PUB-LONGBENCH_V2-SMOKE-MAINLINE-MEMORY-BENEFICIAL",
        stage="smoke",
        expected_variants=["B0", "B1", "V1", "F1", "F2"],
        expected_repeats=[0],
        invalid_rate_threshold=0.15,
        std_threshold=0.10,
    )
    assert result["verdict"] == "accepted"
    assert result["anomaly_codes"] == []


def test_public_track_gate_invalidates_missing_variant() -> None:
    rows = [
        {
            "variant": "B0",
            "repeat_idx": 0,
            "prompt_mode": "memory_neutral",
            "benchmark_score_on_total": 0.30,
            "strict_accuracy_on_total": 0.30,
            "partial_accuracy_on_total": 0.30,
            "parse_failure_rate_on_total": 0.0,
            "no_answer_rate_on_total": 0.0,
            "success_rate": 1.0,
        },
        {
            "variant": "B1",
            "repeat_idx": 0,
            "prompt_mode": "memory_neutral",
            "benchmark_score_on_total": 0.31,
            "strict_accuracy_on_total": 0.31,
            "partial_accuracy_on_total": 0.31,
            "parse_failure_rate_on_total": 0.0,
            "no_answer_rate_on_total": 0.0,
            "success_rate": 1.0,
        },
    ]
    result = _build_result(
        rows,
        dataset="longbench_v2",
        track_id="mainline:memory_neutral",
        workload_id="PUB-LONGBENCH_V2-SMOKE-MAINLINE-MEMORY-NEUTRAL",
        stage="smoke",
        expected_variants=["B0", "B1", "V1", "F1", "F2"],
        expected_repeats=[0],
        invalid_rate_threshold=0.15,
        std_threshold=0.10,
    )
    assert result["verdict"] == "invalid"
    assert "RAW_RUN_TABLE_MISMATCH" in result["anomaly_codes"]


def test_public_track_gate_detects_protocol_drift() -> None:
    rows = []
    for variant in ("B0", "B1", "V1", "F1", "F2"):
        rows.append(
            {
                "variant": variant,
                "repeat_idx": 0,
                "prompt_mode": "memory_neutral",
                "qa_prompt_profile": "multiple_choice_letter",
                "max_evidence_chars": 1600 if variant in {"B0", "B1"} else 18000,
                "context_top_k": 10 if variant in {"B0", "B1"} else 12,
                "context_item_max_chars": 600 if variant in {"B0", "B1"} else 1800,
                "answer_format_mode": "strict_json",
                "benchmark_score_on_total": 0.30,
                "strict_accuracy_on_total": 0.30,
                "partial_accuracy_on_total": 0.30,
                "parse_failure_rate_on_total": 0.0,
                "no_answer_rate_on_total": 0.0,
                "success_rate": 1.0,
            }
        )
    result = _build_result(
        rows,
        dataset="longbench_v2",
        track_id="track-s:memory_neutral",
        workload_id="PUB-LONGBENCH_V2-SMOKE",
        stage="smoke",
        expected_variants=["B0", "B1", "V1", "F1", "F2"],
        expected_repeats=[0],
        invalid_rate_threshold=0.15,
        std_threshold=0.10,
    )
    assert result["verdict"] == "diagnostic_only"
    assert "PROVIDER_PARAM_DRIFT" in result["anomaly_codes"]


def test_public_track_gate_invalidates_ordering_reversal_against_baseline() -> None:
    rows = []
    for repeat_idx, values in enumerate(((0.35, 0.37, 0.34), (0.36, 0.38, 0.33))):
        b1_score, v1_score, f1_score = values
        rows.extend(
            [
                {
                    "variant": "B1",
                    "repeat_idx": repeat_idx,
                    "prompt_mode": "memory_beneficial",
                    "benchmark_score_on_total": b1_score,
                    "strict_accuracy_on_total": b1_score,
                    "partial_accuracy_on_total": b1_score,
                    "parse_failure_rate_on_total": 0.0,
                    "no_answer_rate_on_total": 0.0,
                    "success_rate": 1.0,
                },
                {
                    "variant": "V1",
                    "repeat_idx": repeat_idx,
                    "prompt_mode": "memory_beneficial",
                    "benchmark_score_on_total": v1_score,
                    "strict_accuracy_on_total": v1_score,
                    "partial_accuracy_on_total": v1_score,
                    "parse_failure_rate_on_total": 0.0,
                    "no_answer_rate_on_total": 0.0,
                    "success_rate": 1.0,
                },
                {
                    "variant": "F1",
                    "repeat_idx": repeat_idx,
                    "prompt_mode": "memory_beneficial",
                    "benchmark_score_on_total": f1_score,
                    "strict_accuracy_on_total": f1_score,
                    "partial_accuracy_on_total": f1_score,
                    "parse_failure_rate_on_total": 0.0,
                    "no_answer_rate_on_total": 0.0,
                    "success_rate": 1.0,
                },
            ]
        )
    result = _build_result(
        rows,
        dataset="longbench_v2",
        track_id="mainline:memory_beneficial",
        workload_id="PUB-LONGBENCH_V2-RIGOR-M",
        stage="full",
        expected_variants=["B1", "V1", "F1"],
        expected_repeats=[0, 1],
        invalid_rate_threshold=0.15,
        std_threshold=0.10,
        baseline_variant="B1",
        must_beat_baseline_variants=["V1", "F1"],
    )
    assert result["verdict"] == "invalid"
    assert "UNEXPLAINED_ORDERING_REVERSAL" in result["anomaly_codes"]
    assert "F1" in result["ordering_reversals"]
    assert "V1" not in result["ordering_reversals"]


def test_public_track_gate_invalidates_baseline_out_of_band() -> None:
    rows = [
        {
            "variant": "B0",
            "repeat_idx": 0,
            "prompt_mode": "memory_neutral",
            "benchmark_score_on_total": 0.10,
            "strict_accuracy_on_total": 0.10,
            "partial_accuracy_on_total": 0.10,
            "parse_failure_rate_on_total": 0.0,
            "no_answer_rate_on_total": 0.0,
            "success_rate": 1.0,
        }
    ]
    result = _build_result(
        rows,
        dataset="longbench_v2",
        track_id="mainline:memory_neutral",
        workload_id="PUB-LONGBENCH_V2-FULL-MAINLINE-N",
        stage="full",
        expected_variants=["B0"],
        expected_repeats=[0],
        invalid_rate_threshold=0.15,
        std_threshold=0.10,
        baseline_variant="B0",
        baseline_lower_bound=0.20,
    )
    assert result["verdict"] == "invalid"
    assert "BASELINE_OUT_OF_BAND" in result["anomaly_codes"]
