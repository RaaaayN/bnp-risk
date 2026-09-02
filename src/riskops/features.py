"""Feature engineering metier pour la detection AML.

Toutes les features "historique de compte" sont calculees en ne regardant que
le passe de chaque compte (shift avant agregation) pour eviter toute fuite
temporelle. Le split train/val/test se fait ensuite chronologiquement dans
train.py.
"""
import pathlib

import numpy as np
import pandas as pd

RAW_COLUMNS_REQUIRED = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering",
]

FEATURE_COLUMNS = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_cross_bank",
    "is_cross_currency",
    "payment_format_code",
    "acct_txn_count_prior",
    "acct_avg_amount_prior",
    "amount_vs_acct_avg_ratio",
    "acct_distinct_counterparties_7d",
    "acct_txn_count_24h",
]

PAYMENT_FORMAT_MAP = {
    "Wire": 0, "ACH": 1, "Credit Card": 2, "Cheque": 3, "Cash": 4, "Reinvestment": 5,
}


def load_raw(path: str | pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Timestamp"])
    missing = set(RAW_COLUMNS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {missing}")
    return df.sort_values("Timestamp").reset_index(drop=True)


def _rolling_counterparties_7d(group: pd.DataFrame) -> pd.Series:
    group = group.set_index("Timestamp")
    out = []
    seen_times = group.index.to_list()
    seen_cp = group["Account.1"].to_list()
    for i, t in enumerate(seen_times):
        window_start = t - pd.Timedelta(days=7)
        past = [seen_cp[j] for j in range(i) if seen_times[j] >= window_start]
        out.append(len(set(past)))
    return pd.Series(out, index=group.index)


def _rolling_txn_count_24h_prior(times: list) -> list:
    """Nombre de transactions du meme compte dans les 24h precedentes
    (bornes exclues la transaction courante)."""
    out = []
    for i, t in enumerate(times):
        window_start = t - pd.Timedelta(hours=24)
        count = sum(1 for j in range(i) if times[j] >= window_start)
        out.append(count)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount"] = df["Amount Paid"].astype(float)
    df["amount_log"] = np.log1p(df["amount"])
    df["hour_of_day"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["is_cross_bank"] = (df["From Bank"] != df["To Bank"]).astype(int)
    df["is_cross_currency"] = (df["Payment Currency"] != df["Receiving Currency"]).astype(int)
    df["payment_format_code"] = df["Payment Format"].map(PAYMENT_FORMAT_MAP).fillna(-1).astype(int)

    df = df.sort_values(["Account", "Timestamp"]).reset_index(drop=True)
    grouped = df.groupby("Account", sort=False)

    df["acct_txn_count_prior"] = grouped.cumcount()
    prior_amount_sum = grouped["amount"].cumsum() - df["amount"]
    df["acct_avg_amount_prior"] = (prior_amount_sum / df["acct_txn_count_prior"].replace(0, np.nan)).fillna(0.0)
    df["amount_vs_acct_avg_ratio"] = np.where(
        df["acct_avg_amount_prior"] > 0, df["amount"] / df["acct_avg_amount_prior"], 1.0
    )

    count_24h_parts = []
    cp_parts = []
    for _, g in df.groupby("Account", sort=False):
        times = g["Timestamp"].to_list()
        count_24h_parts.append(pd.Series(_rolling_txn_count_24h_prior(times), index=g.index))
        cp_parts.append(_rolling_counterparties_7d(g[["Timestamp", "Account.1"]]).set_axis(g.index))

    df["acct_txn_count_24h"] = pd.concat(count_24h_parts).sort_index().values
    df["acct_distinct_counterparties_7d"] = pd.concat(cp_parts).sort_index().values

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
