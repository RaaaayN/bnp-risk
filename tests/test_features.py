import pandas as pd

from riskops.features import build_features, FEATURE_COLUMNS


def _toy_raw():
    return pd.DataFrame({
        "Timestamp": pd.to_datetime([
            "2024-01-01 10:00", "2024-01-01 11:00", "2024-01-02 09:00", "2024-01-05 09:00",
        ]),
        "From Bank": [1, 1, 1, 2],
        "Account": [100, 100, 100, 200],
        "To Bank": [2, 2, 3, 1],
        "Account.1": [500, 500, 600, 100],
        "Amount Received": [1000.0, 2000.0, 500.0, 300.0],
        "Receiving Currency": ["US Dollar", "US Dollar", "Euro", "US Dollar"],
        "Amount Paid": [1000.0, 2000.0, 500.0, 300.0],
        "Payment Currency": ["US Dollar", "US Dollar", "US Dollar", "US Dollar"],
        "Payment Format": ["Wire", "Wire", "Cash", "ACH"],
        "Is Laundering": [0, 0, 0, 0],
    })


def test_build_features_has_all_columns():
    feats = build_features(_toy_raw())
    for col in FEATURE_COLUMNS:
        assert col in feats.columns


def test_no_leakage_first_transaction_has_no_prior_history():
    feats = build_features(_toy_raw())
    first_txn = feats.sort_values("Timestamp").iloc[0]
    assert first_txn["acct_txn_count_prior"] == 0
    assert first_txn["acct_avg_amount_prior"] == 0.0


def test_prior_avg_only_uses_past_transactions():
    feats = build_features(_toy_raw())
    acct_100 = feats[feats["Account"] == 100].sort_values("Timestamp").reset_index(drop=True)
    # 3eme transaction du compte 100: la moyenne prior doit etre (1000+2000)/2, pas la moyenne globale
    assert acct_100.loc[2, "acct_txn_count_prior"] == 2
    assert acct_100.loc[2, "acct_avg_amount_prior"] == 1500.0


def test_cross_bank_and_currency_flags():
    feats = build_features(_toy_raw())
    row = feats.iloc[0]
    assert row["is_cross_bank"] == 1  # From Bank 1 -> To Bank 2
    assert row["is_cross_currency"] == 0
