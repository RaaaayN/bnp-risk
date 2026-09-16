import pandas as pd

from riskops.data_gen import generate
from riskops.features import build_features


def test_labels_are_account_episodes_not_iid_rows():
    raw = generate(n_rows=20_000, laundering_rate=0.01, seed=7)
    positives_per_account = raw.loc[raw["Is Laundering"] == 1].groupby("Account").size()
    assert len(raw) == 20_000
    assert raw["Is Laundering"].sum() == 200
    assert positives_per_account.min() >= 5


def test_injected_cross_bank_and_currency_signals_are_effective():
    raw = generate(n_rows=20_000, laundering_rate=0.01, seed=8)
    positive = raw["Is Laundering"] == 1
    cross_bank = raw["From Bank"] != raw["To Bank"]
    cross_currency = raw["Payment Currency"] != raw["Receiving Currency"]
    assert cross_bank[positive].mean() > cross_bank[~positive].mean() + 0.20
    assert cross_currency[positive].mean() > cross_currency[~positive].mean() + 0.15


def test_account_history_has_predictive_signal_and_exists_in_every_time_split():
    raw = generate(n_rows=20_000, laundering_rate=0.01, seed=9)
    feats = build_features(raw)
    positive = feats["Is Laundering"] == 1
    assert feats.loc[positive, "acct_txn_count_24h"].median() > feats.loc[~positive, "acct_txn_count_24h"].median()
    assert feats.loc[positive, "acct_distinct_counterparties_7d"].median() > feats.loc[~positive, "acct_distinct_counterparties_7d"].median()
    time_bin = pd.qcut(feats["Timestamp"], 3, labels=False)
    assert all(feats.loc[time_bin == part, "Is Laundering"].sum() > 0 for part in range(3))
