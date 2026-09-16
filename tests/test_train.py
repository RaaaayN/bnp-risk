import pandas as pd

from riskops.train import _new_xgboost, _period_days


def test_xgboost_uses_validation_early_stopping():
    model = _new_xgboost(pos_weight=10.0)
    assert model.get_params()["early_stopping_rounds"] == 30
    assert model.get_params()["n_jobs"] == 1


def test_period_days_uses_ceiling_for_partial_operational_days():
    frame = pd.DataFrame(
        {"Timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-02 12:00"])}
    )
    assert _period_days(frame) == 2
