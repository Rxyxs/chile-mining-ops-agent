from src.tools.credit_risk_tool import score_credit_risk


def test_score_credit_risk_low_risk_profile():
    result = score_credit_risk(
        age=45,
        monthly_income_clp=2_000_000,
        debt_to_income=0.10,
        months_employed=120,
        n_late_payments=0,
        requested_amount_clp=1_000_000,
    )
    assert 0.0 <= result["probability_default"] <= 1.0
    assert result["risk_tier"] in {"low", "medium", "high", "critical"}


def test_score_credit_risk_high_risk_profile_scores_higher():
    low_risk = score_credit_risk(
        age=45,
        monthly_income_clp=2_500_000,
        debt_to_income=0.05,
        months_employed=180,
        n_late_payments=0,
        requested_amount_clp=500_000,
    )
    high_risk = score_credit_risk(
        age=22,
        monthly_income_clp=300_000,
        debt_to_income=1.2,
        months_employed=1,
        n_late_payments=8,
        requested_amount_clp=8_000_000,
    )
    assert high_risk["probability_default"] > low_risk["probability_default"]


def test_score_credit_risk_returns_input_profile():
    result = score_credit_risk(
        age=30,
        monthly_income_clp=900_000,
        debt_to_income=0.3,
        months_employed=24,
        n_late_payments=1,
        requested_amount_clp=2_000_000,
    )
    assert result["input_profile"]["age"] == 30
