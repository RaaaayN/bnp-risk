"""Feature engineering metier pour la detection AML.

Toutes les features "historique de compte" sont calculees en ne regardant que
le passe de chaque compte (shift avant agregation) pour eviter toute fuite
temporelle. Le split train/val/test se fait ensuite chronologiquement dans
train.py.
"""
import pathlib
from collections import Counter

import numpy as np
import pandas as pd

RAW_COLUMNS_REQUIRED = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

TRANSACTION_FEATURE_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_cross_bank",
    "is_cross_currency",
    "payment_format_wire",
    "payment_format_ach",
    "payment_format_credit_card",
    "payment_format_cheque",
    "payment_format_cash",
    "payment_format_reinvestment",
]

HISTORY_FEATURE_COLUMNS = [
    "acct_txn_count_prior",
    "acct_avg_amount_prior",
    "amount_vs_acct_avg_ratio",
    "acct_distinct_counterparties_7d",
    "acct_txn_count_24h",
    "acct_amount_sum_24h",
    "hours_since_prev_txn",
    "counterparty_seen_before",
]

FEATURE_COLUMNS = TRANSACTION_FEATURE_COLUMNS + HISTORY_FEATURE_COLUMNS

PAYMENT_FORMAT_MAP = {
    "Wire": 0, "ACH": 1, "Credit Card": 2, "Cheque": 3, "Cash": 4, "Reinvestment": 5,
}


def load_raw(path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    missing = set(RAW_COLUMNS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    return df.sort_values("Timestamp").reset_index(drop=True)


def _causal_rolling_history(group: pd.DataFrame) -> pd.DataFrame:
    """Agrégats strictement antérieurs, calculés en O(n) pour un compte."""
    times = group["Timestamp"].tolist()
    amounts = group["amount"].to_numpy(dtype=float)
    counterparties = group["Account.1"].tolist()
    n = len(group)

    count_24h = np.zeros(n, dtype=int)
    amount_24h = np.zeros(n, dtype=float)
    distinct_7d = np.zeros(n, dtype=int)
    hours_since_prev = np.full(n, 24.0 * 90)
    cp_seen_before = np.zeros(n, dtype=int)

    left_24h = 0
    left_7d = 0
    sum_24h = 0.0
    cp_window: Counter = Counter()
    seen: set[int] = set()

    for i, timestamp in enumerate(times):
        while left_24h < i and times[left_24h] < timestamp - pd.Timedelta(hours=24):
            sum_24h -= amounts[left_24h]
            left_24h += 1
        while left_7d < i and times[left_7d] < timestamp - pd.Timedelta(days=7):
            old_cp = counterparties[left_7d]
            cp_window[old_cp] -= 1
            if cp_window[old_cp] == 0:
                del cp_window[old_cp]
            left_7d += 1

        count_24h[i] = i - left_24h
        amount_24h[i] = max(sum_24h, 0.0)
        distinct_7d[i] = len(cp_window)
        cp_seen_before[i] = int(counterparties[i] in seen)
        if i:
            hours_since_prev[i] = max((timestamp - times[i - 1]).total_seconds() / 3600, 0.0)

        sum_24h += amounts[i]
        cp_window[counterparties[i]] += 1
        seen.add(counterparties[i])

    return pd.DataFrame(
        {
            "acct_txn_count_24h": count_24h,
            "acct_amount_sum_24h": amount_24h,
            "acct_distinct_counterparties_7d": distinct_7d,
            "hours_since_prev_txn": hours_since_prev,
            "counterparty_seen_before": cp_seen_before,
        },
        index=group.index,
    )


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = df["Amount Paid"].astype(float)
    df["amount_log"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)
    df["is_cross_currency"] = (df["Payment Currency"] != df["Receiving Currency"]).astype(int)
    for payment_format in PAYMENT_FORMAT_MAP:
        column = "payment_format_" + payment_format.lower().replace(" ", "_")
        df[column] = (df["Payment Format"] == payment_format).astype(int)

    df = df.sort_values(["Account", "Timestamp"]).reset_index(drop=True)
    grouped = df.groupby("Account", sort=False)

    df["acct_txn_count_prior"] = grouped.cumcount()
    prior_amount_sum = grouped["amount"].cumsum() - df["amount"]
    df["acct_avg_amount_prior"] = (prior_amount_sum / df["acct_txn_count_prior"].replace(0, np.nan)).fillna(0.0)
    df["amount_vs_acct_avg_ratio"] = np.where(
        df["acct_avg_amount_prior"] > 0, df["amount"] / df["acct_avg_amount_prior"], 1.0
    )

    history_parts = []
    for _, g in df.groupby("Account", sort=False):
        history_parts.append(_causal_rolling_history(g))

    history = pd.concat(history_parts).sort_index()
    for column in history.columns:
        df[column] = history[column]

    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def make_dataset(raw_path, out_path=None) -> pd.DataFrame:
    raw = load_raw(raw_path)
    feats = build_features(raw)
    if out_path:
        feats.to_parquet(out_path, index=False)
    return feats


if __name__ == "__main__":
    root = pathlib.Path(__file__).resolve().parents[2]
    raw_path = root / "data" / "raw_transactions.csv"
    out_path = root / "data" / "features.parquet"
    feats = make_dataset(raw_path, out_path)
    print(f"{len(feats)} lignes, {len(FEATURE_COLUMNS)} features -> {out_path}")
