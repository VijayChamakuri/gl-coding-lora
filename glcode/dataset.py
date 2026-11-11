"""Labelled Stockport lines and the TrustGate rolling split.

`load_lines` reproduces the frame `realgl.load_lines` builds from
`int_stockport__labelled_lines`: sort by purchase date and card line id,
and (for the classical models) fill missing department, service and
merchant category with "∅". The year used for the split is the statement
month on the source file, not the purchase date. One purchase date in the
export is year 0020; it is left as published.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .paths import REFERENCE, REFERENCE_SHA256

TEST_YEARS = [2018, 2019, 2020, 2021, 2022]
CARD_CATS = ["department", "service_category"]
MISSING = "∅"
YEAR_COUNTS = {2017: 3095, 2018: 3387, 2019: 3382, 2020: 2143, 2021: 1910, 2022: 934}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_lines(path=None, check_hash: bool = False) -> pd.DataFrame:
    path = REFERENCE if path is None else path
    if check_hash and sha256_file(path) != REFERENCE_SHA256:
        raise RuntimeError(f"reference file hash mismatch: {path}")
    df = pd.read_csv(path, parse_dates=["statement_month", "purchase_date"])
    expected = {
        "card_line_id",
        "statement_month",
        "purchase_date",
        "merchant_raw",
        "merchant_key",
        "merchant_category",
        "department",
        "service_category",
        "amount",
        "gl_account",
    }
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"labelled file is missing columns: {sorted(missing)}")
    df["amount"] = df["amount"].astype(float)
    df["statement_year"] = df["statement_month"].dt.year.astype(int)
    if not df["card_line_id"].is_unique:
        raise RuntimeError("card_line_id is not unique")
    return df.sort_values(["purchase_date", "card_line_id"], kind="mergesort").reset_index(drop=True)


def model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Frame used by the lookup and logistic baselines (null categories -> ∅)."""
    out = df.copy()
    for col in ("department", "service_category", "merchant_category"):
        out[col] = out[col].fillna(MISSING)
    return out


def iter_folds(df: pd.DataFrame):
    for year in TEST_YEARS:
        train = df[df.statement_year < year]
        test = df[df.statement_year == year]
        yield year, train, test


def candidate_accounts(train: pd.DataFrame) -> list[str]:
    """Chart of accounts the fold is allowed to predict: those seen in training years."""
    return sorted(train.gl_account.unique())


def stratified_sample(df: pd.DataFrame, n: int, seed: int = 0, col: str = "gl_account") -> pd.DataFrame:
    """Proportional sample with the largest-remainder method.

    n <= 0 or n >= len(df) returns every row. Within a class, rows are drawn
    with `numpy.random.Generator(seed)` from card_line_id order. Class
    remainder ties break alphabetically. The result is sorted by purchase
    date and card line id.
    """
    if n <= 0 or n >= len(df):
        return df.sort_values(["purchase_date", "card_line_id"], kind="mergesort").reset_index(drop=True)
    counts = df[col].value_counts()
    raw = counts.astype(float) / float(counts.sum()) * n
    base = {a: min(int(np.floor(raw[a])), int(counts[a])) for a in counts.index}
    short = n - sum(base.values())
    room = {a: int(counts[a]) - base[a] for a in counts.index}
    frac = {a: float(raw[a] - np.floor(raw[a])) for a in counts.index}
    for account in sorted(counts.index, key=lambda a: (-frac[a], a)):
        if short <= 0:
            break
        if room[account] <= 0:
            continue
        base[account] += 1
        room[account] -= 1
        short -= 1
    rng = np.random.default_rng(seed)
    parts = []
    used: set[str] = set()
    for account in sorted(base):
        k = base[account]
        if k <= 0:
            continue
        pool = df[df[col] == account].sort_values("card_line_id", kind="mergesort")
        pick = rng.choice(len(pool), size=k, replace=False)
        chosen = pool.iloc[np.sort(pick)]
        parts.append(chosen)
        used.update(chosen.card_line_id.tolist())
    if short > 0:
        rest = df[~df.card_line_id.isin(used)].sort_values("card_line_id", kind="mergesort")
        if short > len(rest):
            raise RuntimeError("stratified sample asked for more rows than exist")
        pick = rng.choice(len(rest), size=short, replace=False)
        parts.append(rest.iloc[np.sort(pick)])
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["purchase_date", "card_line_id"], kind="mergesort").reset_index(drop=True)


def select_exemplars(train: pd.DataFrame, k: int) -> pd.DataFrame:
    """k few-shot rows from the training fold only.

    The k most frequent training accounts (ties broken alphabetically). For
    each, the earliest line by purchase date then card line id.
    """
    if k <= 0:
        return train.iloc[0:0].copy()
    counts = train.gl_account.value_counts()
    accounts = sorted(counts.index, key=lambda a: (-int(counts[a]), a))[:k]
    ordered = train.sort_values(["purchase_date", "card_line_id"], kind="mergesort")
    rows = [ordered[ordered.gl_account == account].iloc[0] for account in accounts]
    return pd.DataFrame(rows).reset_index(drop=True)
