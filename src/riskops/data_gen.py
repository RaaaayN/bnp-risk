"""Générateur synthétique de transactions au format IBM AML.

Le label n'est pas tiré indépendamment ligne par ligne. Les positifs sont des
épisodes multi-transactions portés par un même compte (structuration, fan-out
ou transit rapide). Le comportement observable du compte est donc réellement
lié au label, tout en restant bruité par des épisodes bénins similaires.
"""
import numpy as np
import pandas as pd

CURRENCIES = ["US Dollar", "Euro", "UK Pound", "Yen", "Swiss Franc"]
PAYMENT_FORMATS = ["Wire", "ACH", "Credit Card", "Cheque", "Cash", "Reinvestment"]
N_BANKS = 60
N_ACCOUNTS = 4000
N_DAYS = 90


def _episode_sizes(total: int, rng: np.random.Generator, low: int = 5, high: int = 10) -> list[int]:
    """Découpe ``total`` en épisodes, sans créer de positif isolé."""
    sizes: list[int] = []
    remaining = total
    while remaining:
        if remaining <= high:
            if remaining < low and sizes:
                sizes[-1] += remaining
            else:
                sizes.append(remaining)
            break
        size = int(rng.integers(low, high + 1))
        if 0 < remaining - size < low:
            size = remaining - low
        sizes.append(size)
        remaining -= size
    return sizes


def generate(n_rows: int = 120_000, laundering_rate: float = 0.003, seed: int = 42) -> pd.DataFrame:
    if n_rows <= 0:
        raise ValueError("n_rows doit être strictement positif")
    if not 0 <= laundering_rate < 1:
        raise ValueError("laundering_rate doit être compris entre 0 et 1")

    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")

    # Profils persistants : banque, devise et montant usuel propres au compte.
    home_bank = rng.integers(1, N_BANKS + 1, size=N_ACCOUNTS + 1)
    home_currency = rng.choice(CURRENCIES, size=N_ACCOUNTS + 1)
    typical_amount = rng.lognormal(mean=6.1, sigma=0.7, size=N_ACCOUNTS + 1)

    # Une distribution d'activité hétérogène évite que tous les comptes aient
    # artificiellement le même historique.
    account_weights = rng.lognormal(mean=0.0, sigma=0.9, size=N_ACCOUNTS)
    account_weights /= account_weights.sum()
    from_account = rng.choice(np.arange(1, N_ACCOUNTS + 1), size=n_rows, p=account_weights)
    from_bank = home_bank[from_account]

    minutes = rng.integers(0, 60 * 24 * N_DAYS, size=n_rows)
    timestamps = (start + pd.to_timedelta(minutes, unit="m")).to_numpy(copy=True)

    # Les paiements ordinaires réutilisent souvent une petite liste de
    # bénéficiaires. C'est ce qui rend une soudaine dispersion mesurable.
    regular_counterparties = rng.integers(1, N_ACCOUNTS + 1, size=(N_ACCOUNTS + 1, 6))
    pool_slot = rng.integers(0, regular_counterparties.shape[1], size=n_rows)
    to_account = regular_counterparties[from_account, pool_slot]
    to_bank = home_bank[to_account].copy()

    # Environ 25 % de flux cross-bank et 8 % cross-currency dans l'activité
    # normale. Les valeurs sont imposées, pas simplement retirées au hasard.
    normal_cross_bank = rng.random(n_rows) < 0.25
    to_bank[~normal_cross_bank] = from_bank[~normal_cross_bank]
    same_bank_collision = normal_cross_bank & (to_bank == from_bank)
    to_bank[same_bank_collision] = (from_bank[same_bank_collision] % N_BANKS) + 1

    pay_ccy = home_currency[from_account].astype(object)
    recv_ccy = pay_ccy.copy()
    normal_cross_ccy = rng.random(n_rows) < 0.08
    currency_offset = rng.integers(1, len(CURRENCIES), size=normal_cross_ccy.sum())
    pay_idx = np.array([CURRENCIES.index(c) for c in pay_ccy[normal_cross_ccy]])
    recv_ccy[normal_cross_ccy] = np.asarray(CURRENCIES)[(pay_idx + currency_offset) % len(CURRENCIES)]

    amount = typical_amount[from_account] * rng.lognormal(mean=0.0, sigma=0.65, size=n_rows)
    payment_format = rng.choice(
        PAYMENT_FORMATS, size=n_rows, p=[0.18, 0.30, 0.25, 0.10, 0.08, 0.09]
    ).astype(object)
    is_laundering = np.zeros(n_rows, dtype=int)

    shuffled_positions = rng.permutation(n_rows)
    n_positive = int(round(n_rows * laundering_rate))
    positive_positions = shuffled_positions[:n_positive]
    cursor = 0

    # On choisit des comptes déjà actifs afin que le contraste à leur propre
    # historique soit calculable dès le début de l'épisode.
    active_accounts = np.flatnonzero(np.bincount(from_account, minlength=N_ACCOUNTS + 1) >= 8)
    active_accounts = active_accounts[active_accounts != 0]
    if len(active_accounts) == 0:
        active_accounts = np.arange(1, N_ACCOUNTS + 1)

    for episode_no, size in enumerate(_episode_sizes(n_positive, rng)):
        positions = positive_positions[cursor: cursor + size]
        cursor += size
        account = int(rng.choice(active_accounts))
        scenario = episode_no % 3

        # Les épisodes sont répartis sur toute la période : le split
        # chronologique rencontre les mêmes mécanismes dans le futur sans voir
        # les transactions futures pendant l'entraînement.
        episode_start = start + pd.to_timedelta(
            int(rng.integers(2 * 24 * 60, (N_DAYS - 1) * 24 * 60)), unit="m"
        )
        timestamps[positions] = episode_start + pd.to_timedelta(
            np.sort(rng.integers(0, 10 * 60, size=size)), unit="m"
        )
        from_account[positions] = account
        from_bank[positions] = home_bank[account]
        pay_ccy[positions] = home_currency[account]
        is_laundering[positions] = 1

        # Chaque scénario porte une combinaison différente de signaux. Aucun
        # indicateur isolé ne suffit pour reconnaître tous les épisodes.
        cp_pool_size = (3, size, 2)[scenario]
        counterparties = rng.choice(
            np.arange(1, N_ACCOUNTS + 1), size=cp_pool_size, replace=False
        )
        to_account[positions] = rng.choice(counterparties, size=size, replace=cp_pool_size < size)
        cross_bank_rate = (0.40, 0.80, 0.60)[scenario]
        cross_bank = rng.random(size) < cross_bank_rate
        to_bank[positions] = from_bank[positions]
        cross_positions = positions[cross_bank]
        to_bank[cross_positions] = (
            from_bank[cross_positions] + rng.integers(1, N_BANKS, size=cross_bank.sum()) - 1
        ) % N_BANKS + 1

        cross_ccy_rate = (0.12, 0.20, 0.75)[scenario]
        cross_ccy = rng.random(size) < cross_ccy_rate
        recv_ccy[positions] = pay_ccy[positions]
        base_ccy_idx = CURRENCIES.index(home_currency[account])
        recv_ccy[positions[cross_ccy]] = np.asarray(CURRENCIES)[
            (base_ccy_idx + rng.integers(1, len(CURRENCIES), size=cross_ccy.sum())) % len(CURRENCIES)
        ]

        if scenario == 0:  # structuration sous un seuil fictif de 10 000
            amount[positions] = np.round(rng.uniform(6_500, 9_900, size=size) / 100) * 100
            payment_format[positions] = rng.choice(["Cash", "Wire"], size=size, p=[0.55, 0.45])
        elif scenario == 1:  # fan-out : montants plausibles, dispersion anormale
            amount[positions] = typical_amount[account] * rng.uniform(0.8, 2.0, size=size)
            payment_format[positions] = rng.choice(["Wire", "ACH"], size=size, p=[0.75, 0.25])
        else:  # transit rapide cross-currency
            amount[positions] = typical_amount[account] * rng.uniform(1.5, 4.0, size=size)
            payment_format[positions] = "Wire"

    # Contrôles négatifs difficiles : des rafales légitimes existent aussi,
    # mais vers des bénéficiaires récurrents, en ACH/carte et dans la devise du
    # compte. La fréquence seule ne permet donc pas de résoudre le problème.
    remaining_positions = shuffled_positions[n_positive:]
    benign_total = min(len(remaining_positions), max(n_positive * 5, n_rows // 200))
    benign_positions = remaining_positions[:benign_total]
    cursor = 0
    for episode_no, size in enumerate(_episode_sizes(benign_total, rng, low=5, high=12)):
        positions = benign_positions[cursor: cursor + size]
        cursor += size
        account = int(rng.choice(active_accounts))
        episode_start = start + pd.to_timedelta(int(rng.integers(0, N_DAYS * 24 * 60)), unit="m")
        timestamps[positions] = episode_start + pd.to_timedelta(
            np.sort(rng.integers(0, 10 * 60, size=size)), unit="m"
        )
        from_account[positions] = account
        from_bank[positions] = home_bank[account]
        pay_ccy[positions] = home_currency[account]
        scenario = episode_no % 3
        recv_ccy[positions] = home_currency[account]

        if scenario == 0:  # paie/fournisseurs : fan-out légitime
            cp = rng.choice(np.arange(1, N_ACCOUNTS + 1), size=size, replace=False)
            to_account[positions] = cp
            cross_bank = rng.random(size) < 0.75
            to_bank[positions] = from_bank[positions]
            cross_positions = positions[cross_bank]
            to_bank[cross_positions] = (
                from_bank[cross_positions] + rng.integers(1, N_BANKS, size=cross_bank.sum()) - 1
            ) % N_BANKS + 1
            amount[positions] = typical_amount[account] * rng.uniform(0.7, 2.2, size=size)
            payment_format[positions] = rng.choice(["ACH", "Wire"], size=size, p=[0.8, 0.2])
        elif scenario == 1:  # commerçant : dépôts cash élevés et rapprochés
            cp = regular_counterparties[account, :2]
            to_account[positions] = rng.choice(cp, size=size)
            to_bank[positions] = from_bank[positions]
            amount[positions] = np.round(rng.uniform(5_500, 11_000, size=size) / 100) * 100
            payment_format[positions] = rng.choice(["Cash", "ACH"], size=size, p=[0.65, 0.35])
        else:  # activité internationale légitime
            cp = regular_counterparties[account, :3]
            to_account[positions] = rng.choice(cp, size=size)
            cross_bank = rng.random(size) < 0.65
            to_bank[positions] = from_bank[positions]
            cross_positions = positions[cross_bank]
            to_bank[cross_positions] = (
                from_bank[cross_positions] + rng.integers(1, N_BANKS, size=cross_bank.sum()) - 1
            ) % N_BANKS + 1
            cross_ccy = rng.random(size) < 0.70
            recv_ccy[positions[cross_ccy]] = np.asarray(CURRENCIES)[
                (CURRENCIES.index(home_currency[account]) + rng.integers(1, len(CURRENCIES), size=cross_ccy.sum()))
                % len(CURRENCIES)
            ]
            amount[positions] = typical_amount[account] * rng.uniform(1.0, 4.5, size=size)
            payment_format[positions] = rng.choice(["Wire", "Credit Card"], size=size, p=[0.55, 0.45])

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
            "Is Laundering": is_laundering,
        }
    ).sort_values("Timestamp").reset_index(drop=True)
    df.insert(0, "Transaction Id", [f"TXN{100000 + i}" for i in range(len(df))])
    return df


if __name__ == "__main__":
    import pathlib

    out_dir = pathlib.Path(__file__).resolve().parents[2] / "data"
    out_dir.mkdir(exist_ok=True)
    df = generate()
    out_path = out_dir / "raw_transactions.csv"
    df.to_csv(out_path, index=False)
    print(
        f"{len(df)} transactions écrites dans {out_path} "
        f"({df['Is Laundering'].mean() * 100:.3f}% positives)"
    )
