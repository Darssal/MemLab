from __future__ import annotations

from memlab.pipeline.model_profile import (
    build_profile,
    classify_model_scale,
    estimate_model_size_b,
)


def test_estimate_model_size_b() -> None:
    assert estimate_model_size_b("Qwen/Qwen3-8B") == 8.0
    assert estimate_model_size_b("SomeModel-70B-Instruct") == 70.0
    assert estimate_model_size_b("no-size-token") is None


def test_classify_model_scale() -> None:
    assert classify_model_scale(7.0) == "small"
    assert classify_model_scale(13.0) == "medium"
    assert classify_model_scale(72.0) == "large"
    assert classify_model_scale(None) == "unknown"


def test_build_profile_runtime_recommendation_uses_scale() -> None:
    profile = build_profile(
        model_id="Qwen/Qwen3-72B",
        base_url="https://api.example.com/v1",
        registry_entries=[],
    )
    assert profile["model"]["scale_class"] == "large"
    rec = profile["recommended_runtime"]
    assert rec["e2e_inflight"] == 1
    assert rec["context_top_k"] == 8
    assert "metric_contract" in profile
