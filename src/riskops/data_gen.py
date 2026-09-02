"""
Generateur de transactions synthetiques au format du dataset IBM AML
(colonnes: Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering).

Le dataset reel (Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
pese plusieurs Go et necessite des credentials Kaggle. Pour un projet demo de
quelques jours, on genere ici un jeu synthetique avec le meme schema et le meme
type de desequilibre de classe (~0.1-0.5% de blanchiment), ce qui permet de
faire tourner tout le pipeline sans dependance externe. Le loader (features.py)
accepte aussi bien ce fichier que le vrai CSV IBM si l'utilisateur le place dans
data/raw_transactions.csv.
"""
import numpy as np
import pandas as pd

CURRENCIES = ["US Dollar", "Euro", "UK Pound", "Yen", "Swiss Franc"]
PAYMENT_FORMATS = ["Wire", "ACH", "Credit Card", "Cheque", "Cash", "Reinvestment"]
N_BANKS = 60
N_ACCOUNTS = 4000


def generate(n_rows: int = 120_000, laundering_rate: float = 0.003, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    start = pd.Timestamp("2024-01-01")
    minutes = rng.integers(0, 60 * 24 * 90, size=n_rows)  # 90 jours
    timestamps = start + pd.to_timedelta(minutes, unit="m")

    from_bank = rng.integers(1, N_BANKS, size=n_rows)
    to_bank = rng.integers(1, N_BANKS, size=n_rows)
    from_account = rng.integers(1, N_ACCOUNTS, size=n_rows)
    to_account = rng.integers(1, N_ACCOUNTS, size=n_rows)

    base_amount = rng.lognormal(mean=6.0, sigma=1.3, size=n_rows)
    payment_format = rng.choice(PAYMENT_FORMATS, size=n_rows, p=[0.25, 0.25, 0.2, 0.1, 0.1, 0.1])
    pay_ccy = rng.choice(CURRENCIES, size=n_rows)
    recv_ccy = rng.choice(CURRENCIES, size=n_rows)

    is_laundering = rng.random(n_rows) < laundering_rate

    # Les transactions de blanchiment ont des caracteristiques distinctes mais
    # bruitees (pas de separation parfaite, sinon le probleme est trivial) :
    # montants ronds/eleves, virements cross-bank/cross-currency, cash/wire.
    amount = base_amount.copy()
    laundering_idx = np.where(is_laundering)[0]
    amount[laundering_idx] *= rng.uniform(3, 15, size=len(laundering_idx))
    round_mask = rng.random(len(laundering_idx)) < 0.5
    amount[laundering_idx[round_mask]] = np.round(amount[laundering_idx[round_mask]], -3)

    payment_format = payment_format.astype(object)
    fmt_mask = laundering_idx[rng.random(len(laundering_idx)) < 0.6]
    payment_format[fmt_mask] = rng.choice(["Wire", "Cash"], size=len(fmt_mask))

    bank_mask = laundering_idx[rng.random(len(laundering_idx)) < 0.7]
    to_bank[bank_mask] = rng.integers(1, N_BANKS, size=len(bank_mask))

    recv_ccy = recv_ccy.astype(object)
    ccy_mask = laundering_idx[rng.random(len(laundering_idx)) < 0.4]
    recv_ccy[ccy_mask] = rng.choice(CURRENCIES, size=len(ccy_mask))

    df = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "From Bank": from_bank,
            "Account": from_account,
            "To Bank": to_bank,
            "Account.1": to_account,
            "Amount Received": np.round(amount, 2),
            "Receiving Currency": recv_ccy,
            "Amount Paid": np.round(amount, 2),
            "Payment Currency": pay_ccy,
            "Payment Format": payment_format,
            "Is Laundering": is_laundering.astype(int),
        }
    )
    df = df.sort_values("Timestamp").reset_index(drop=True)
    df.insert(0, "Transaction Id", [f"TXN{100000 + i}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate()
    out_path = out_dir / "raw_transactions.csv"
    df.to_csv(out_path, index=False)
    print(f"{len(df)} transactions ecrites dans {out_path} "
          f"({df['Is Laundering'].mean() * 100:.3f}% positives)")
