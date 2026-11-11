"""Classical baselines on the TrustGate rolling split.

Merchant lookup is the procedure in `trustgate.realgl.Lookup`: most frequent
account for the merchant key, then department+service, then the global mode.
Character TF-IDF logistic regression is `LinearCoder` with the same
hyperparameters (char_wb 2–5 grams, min_df 2, one-hot min_frequency 2,
log1p amount, C=6). Their published top-3 is the unweighted mean of yearly
top-3 rates; this module also reports the line-weighted (micro) top-3.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder

from .dataset import CARD_CATS, TEST_YEARS, iter_folds, model_frame
from .metrics import bootstrap_ci

CARD_FEED_COLS = ["merchant_raw", *CARD_CATS]


class MerchantLookup:
    name = "merchant_lookup"

    def fit(self, train: pd.DataFrame, cats: list[str] = CARD_CATS):
        self.cats = cats
        self.by_merchant = train.groupby("merchant_key").gl_account.agg(lambda s: s.value_counts().index[0])
        self.by_cat = train.groupby(cats).gl_account.agg(lambda s: s.value_counts().index[0])
        self.glob = train.gl_account.value_counts().index[0]
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        cat = pd.Series(
            [
                self.by_cat.get(tuple(row) if len(self.cats) > 1 else row[0], np.nan)
                for row in test[self.cats].itertuples(index=False)
            ],
            index=test.index,
        )
        return test.merchant_key.map(self.by_merchant).fillna(cat).fillna(self.glob).to_numpy()


class MerchantTop3:
    """Top-3 analogue of merchant lookup.

    Seen merchant: its most frequent accounts, padded with the global ranking
    until three names. Unseen merchant: the same using department+service,
    then the global ranking. This baseline is not in the TrustGate table.
    """

    name = "merchant_top3"

    def fit(self, train: pd.DataFrame, cats: list[str] = CARD_CATS, k: int = 3):
        self.cats = cats
        self.k = k
        self.by_merchant = {key: list(s.value_counts().index[:k]) for key, s in train.groupby("merchant_key").gl_account}
        self.by_cat = {key: list(s.value_counts().index[:k]) for key, s in train.groupby(cats).gl_account}
        self.glob = list(train.gl_account.value_counts().index[:k])
        return self

    def predict_sets(self, test: pd.DataFrame) -> list[list[str]]:
        sets = []
        for row in test.itertuples(index=False):
            if row.merchant_key in self.by_merchant:
                chosen = list(self.by_merchant[row.merchant_key])
            else:
                key = tuple(getattr(row, c) for c in self.cats) if len(self.cats) > 1 else getattr(row, self.cats[0])
                chosen = list(self.by_cat.get(key, []))
            for account in self.glob:
                if len(chosen) >= self.k:
                    break
                if account not in chosen:
                    chosen.append(account)
            sets.append(chosen[: self.k])
        return sets


class TfidfLogReg:
    name = "tfidf_logreg"

    def __init__(self, cats: list[str] = CARD_CATS, C: float = 6.0):
        self.cats = cats
        self.C = C

    def _X(self, frame: pd.DataFrame, fit: bool):
        parts = []
        if fit:
            self.ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=2).fit(frame[self.cats])
        parts.append(self.ohe.transform(frame[self.cats]))
        if fit:
            self.tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True).fit(
                frame.merchant_raw.fillna("")
            )
        parts.append(self.tfidf.transform(frame.merchant_raw.fillna("")))
        parts.append(sparse.csr_matrix(np.log1p(frame.amount.clip(lower=0).to_numpy())[:, None] / 5))
        return sparse.hstack(parts).tocsr()

    def fit(self, train: pd.DataFrame):
        self.model = LogisticRegression(C=self.C, max_iter=2000).fit(self._X(train, True), train.gl_account)
        self.classes_ = self.model.classes_
        return self

    def predict_proba(self, test: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self._X(test, False))


def label_ceiling(df: pd.DataFrame, cols: list[str] = CARD_FEED_COLS) -> float:
    """Share of lines that carry the majority account of their identical inputs."""
    grouped = df.groupby(cols, observed=True).gl_account
    majority = grouped.transform(lambda s: s.value_counts().iloc[0])
    size = grouped.transform("size")
    return float((majority / size).mean())


def _empty_bucket():
    return {"y": [], "pred": [], "top3": [], "seen": [], "year": []}


def _top3_hits(proba: np.ndarray, classes: np.ndarray, y: np.ndarray, k: int = 3) -> np.ndarray:
    k = min(k, proba.shape[1])
    top = classes[np.argsort(-proba, axis=1, kind="stable")[:, :k]]
    return np.array([label in row for label, row in zip(y, top, strict=True)])


def rolling_eval(df: pd.DataFrame, models: Iterable[str] = ("majority", "merchant_lookup", "merchant_top3", "tfidf_logreg")) -> dict:
    frame = model_frame(df)
    wanted = list(models)
    pooled = {name: _empty_bucket() for name in wanted}
    per_year: list[dict] = []
    for year, train, test in iter_folds(frame):
        seen = test.merchant_key.isin(set(train.merchant_key)).to_numpy()
        y = test.gl_account.to_numpy()
        fitted: dict[str, object] = {}
        if "merchant_lookup" in wanted or "merchant_top3" in wanted:
            fitted["lookup"] = MerchantLookup().fit(train, CARD_CATS)
        if "merchant_top3" in wanted:
            fitted["top3"] = MerchantTop3().fit(train, CARD_CATS)
        if "tfidf_logreg" in wanted:
            t0 = time.time()
            fitted["tfidf"] = TfidfLogReg(CARD_CATS).fit(train)
            tfidf_fit_s = time.time() - t0
        else:
            tfidf_fit_s = None
        for name in wanted:
            t0 = time.time()
            top3_rate = None
            if name == "majority":
                ranking = list(train.gl_account.value_counts().index[:3])
                pred = np.full(len(test), ranking[0])
                hits = np.array([label in ranking for label in y])
                top3_rate = float(hits.mean())
            elif name == "merchant_lookup":
                pred = fitted["lookup"].predict(test)
                hits = np.zeros(len(test), dtype=bool)
            elif name == "merchant_top3":
                sets = fitted["top3"].predict_sets(test)
                pred = np.array([row[0] for row in sets])
                hits = np.array([label in row for label, row in zip(y, sets, strict=True)])
                top3_rate = float(hits.mean())
            elif name == "tfidf_logreg":
                proba = fitted["tfidf"].predict_proba(test)
                pred = fitted["tfidf"].classes_[proba.argmax(1)]
                hits = _top3_hits(proba, fitted["tfidf"].classes_, y)
                top3_rate = float(hits.mean())
            else:
                raise ValueError(name)
            per_year.append(
                {
                    "model": name,
                    "year": year,
                    "n": int(len(test)),
                    "accuracy": float((pred == y).mean()),
                    "top3": top3_rate,
                    "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                    "seen_merchant_share": float(seen.mean()),
                    "accuracy_new_merchants": float((pred[~seen] == y[~seen]).mean()) if (~seen).any() else None,
                    "seconds": round(time.time() - t0 + (tfidf_fit_s or 0 if name == "tfidf_logreg" else 0), 2),
                }
            )
            if name == "tfidf_logreg":
                tfidf_fit_s = 0.0  # counted once, on this model
            bucket = pooled[name]
            bucket["y"].extend(y.tolist())
            bucket["pred"].extend(pred.tolist())
            if name == "merchant_lookup":
                bucket["top3"].extend([np.nan] * len(y))
            else:
                bucket["top3"].extend(hits.astype(float).tolist())
            bucket["seen"].extend(seen.tolist())
            bucket["year"].extend([year] * len(y))
    summary = {}
    for name, bucket in pooled.items():
        y = np.array(bucket["y"])
        pred = np.array(bucket["pred"])
        seen = np.array(bucket["seen"], dtype=bool)
        correct = (y == pred).astype(float)
        yearly = [row for row in per_year if row["model"] == name]
        top3_values = [row["top3"] for row in yearly]
        has_top3 = any(v is not None for v in top3_values)
        micro = None
        if has_top3:
            micro = float(np.mean([h for h in bucket["top3"] if h == h]))  # drop nan
        summary[name] = {
            "n": int(len(y)),
            "accuracy": float(correct.mean()),
            "accuracy_ci": bootstrap_ci(correct),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "top3_micro": micro,
            "top3_year_macro": float(np.mean([v for v in top3_values if v is not None])) if has_top3 else None,
            "accuracy_seen_merchants": float(correct[seen].mean()),
            "accuracy_new_merchants": float(correct[~seen].mean()),
            "n_new_merchants": int((~seen).sum()),
            "new_merchant_ci": bootstrap_ci(correct[~seen]),
        }
    return {
        "years": TEST_YEARS,
        "n_labelled": int(len(frame)),
        "label_ceiling_card_feed": label_ceiling(frame),
        "models": summary,
        "per_year": per_year,
    }


def lookup_on_rows(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Merchant lookup fit on `train` and applied to `test` (nulls filled here)."""
    return MerchantLookup().fit(model_frame(train)).predict(model_frame(test))
