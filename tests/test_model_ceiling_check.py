from src.model_ceiling_check import compute_ceiling_comparison


def test_ceiling_comparison_keys_and_ranges():
    result = compute_ceiling_comparison()

    for key in (
        "oracle_auc",
        "logistic_regression_auc",
        "gradient_boosting_auc",
        "logistic_ceiling_capture_pct",
        "gbm_ceiling_capture_pct",
    ):
        assert key in result

    assert 0.5 <= result["oracle_auc"] <= 1.0
    assert 0.5 <= result["logistic_regression_auc"] <= 1.0
    assert 0.5 <= result["gradient_boosting_auc"] <= 1.0


def test_no_model_beats_the_oracle_ceiling():
    """The oracle uses the true data-generating probability -- by construction,
    nothing fit on a noisy Bernoulli draw of that same probability can score higher."""
    result = compute_ceiling_comparison()

    assert result["logistic_regression_auc"] <= result["oracle_auc"]
    assert result["gradient_boosting_auc"] <= result["oracle_auc"]


def test_logistic_regression_captures_most_of_the_available_signal():
    """Regression guard for the README's claim: the weak AUC isn't a fixable
    modeling bug, the logistic model already gets most of the way to the ceiling."""
    result = compute_ceiling_comparison()

    assert result["logistic_ceiling_capture_pct"] > 85.0
