"""Tests for model-aware pricing and usage logging."""
import csv
from unittest.mock import patch

import pytest

from cpt_categorizer.config.openai import get_model_costs
from cpt_categorizer.utils.logging import log_agent_usage


class TestGetModelCosts:
    """Unit tests for get_model_costs."""

    def test_empty_model_returns_zero_rates(self):
        assert get_model_costs("") == (0.0, 0.0)
        assert get_model_costs("   ") == (0.0, 0.0)

    def test_gpt4o_returns_expected_rates(self):
        inp, out = get_model_costs("gpt-4o")
        assert inp == 0.005 / 1000
        assert out == 0.015 / 1000

    def test_gpt4o_mini_returns_expected_rates(self):
        inp, out = get_model_costs("gpt-4o-mini")
        assert inp == 0.00015 / 1000
        assert out == 0.0006 / 1000

    def test_unknown_model_falls_back_to_gpt4o(self):
        inp, out = get_model_costs("unknown-model")
        assert inp == 0.005 / 1000
        assert out == 0.015 / 1000


class TestLogAgentUsageModelAwareCost:
    """Test that log_agent_usage computes cost_usd from model."""

    def test_cost_zero_when_model_empty(self, tmp_path):
        with patch("cpt_categorizer.utils.logging.LOG_DIR", tmp_path):
            log_agent_usage(
                timestamp="2025-01-01T00:00:00",
                raw_text="test",
                description="test",
                parsed_output="[]",
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
                model="",
                runtime_ms=0,
                success=True,
                is_error=False,
                error_message="",
            )
            log_path = tmp_path / "usage.csv"
            assert log_path.exists()
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            assert len(rows) == 1
            assert float(rows[0]["cost_usd"]) == 0.0
            assert rows[0]["prompt_tokens"] == "100"
            assert rows[0]["model"] == ""

    def test_cost_gpt4o(self, tmp_path):
        with patch("cpt_categorizer.utils.logging.LOG_DIR", tmp_path):
            log_agent_usage(
                timestamp="2025-01-01T00:00:00",
                raw_text="test",
                description="test",
                parsed_output="[]",
                prompt_tokens=1000,
                completion_tokens=500,
                total_tokens=1500,
                model="gpt-4o",
                runtime_ms=0,
                success=True,
                is_error=False,
                error_message="",
            )
            log_path = tmp_path / "usage.csv"
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            cost = float(rows[0]["cost_usd"])
            # 1000 * 0.005/1000 + 500 * 0.015/1000 = 0.005 + 0.0075 = 0.0125
            assert cost == pytest.approx(0.0125, rel=1e-6)

    def test_cost_gpt4o_mini(self, tmp_path):
        with patch("cpt_categorizer.utils.logging.LOG_DIR", tmp_path):
            log_agent_usage(
                timestamp="2025-01-01T00:00:00",
                raw_text="test",
                description="test",
                parsed_output="[]",
                prompt_tokens=1000,
                completion_tokens=1000,
                total_tokens=2000,
                model="gpt-4o-mini",
                runtime_ms=0,
                success=True,
                is_error=False,
                error_message="",
            )
            log_path = tmp_path / "usage.csv"
            with open(log_path) as f:
                rows = list(csv.DictReader(f))
            cost = float(rows[0]["cost_usd"])
            # 1000 * 0.00015/1000 + 1000 * 0.0006/1000 = 0.00015 + 0.0006 = 0.00075
            assert cost == pytest.approx(0.00075, rel=1e-6)
