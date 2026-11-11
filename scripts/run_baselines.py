#!/usr/bin/env python3
"""Fit the rolling-split baselines and write results/baselines.json."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.baselines import rolling_eval
from glcode.dataset import load_lines
from glcode.hardware import describe
from glcode.paths import RESULTS
from glcode.published import CARD_FEED, LABEL_CEILING_CARD_FEED, TRUSTGATE_COMMIT, TRUSTGATE_GL_JSON


def main() -> None:
    t0 = time.time()
    measured = rolling_eval(load_lines())
    deltas = {}
    for name, published in CARD_FEED.items():
        if name not in measured["models"]:
            continue
        got = measured["models"][name]["accuracy"]
        deltas[name] = {
            "published_accuracy": published["accuracy"],
            "remeasured_accuracy": got,
            "accuracy_abs_diff": abs(got - published["accuracy"]),
            "line_delta_at_n11756": round((got - published["accuracy"]) * published["n"], 2),
        }
    ceiling_delta = abs(measured["label_ceiling_card_feed"] - LABEL_CEILING_CARD_FEED)
    report = {
        "measured": True,
        "protocol": "TrustGate card-feed rolling years 2018-2022, pooled",
        "seconds": round(time.time() - t0, 1),
        "hardware": describe(),
        "baselines": measured,
        "published_trustgate": {
            "commit": TRUSTGATE_COMMIT,
            "source": TRUSTGATE_GL_JSON,
            "card_feed": CARD_FEED,
            "label_ceiling_card_feed": LABEL_CEILING_CARD_FEED,
            "remeasured_here": ["majority", "merchant_lookup", "tfidf_logreg"],
            "not_remeasured_here": ["tfidf+embed_logreg", "embed_logreg", "embed_knn"],
        },
        "delta_vs_published": deltas,
        "label_ceiling_abs_diff": ceiling_delta,
        "merchant_top3_definition": (
            "Up to three most frequent training accounts for the merchant, padded with the "
            "global ranking. Unseen merchants use department+service, then the global ranking. "
            "Not reported by TrustGate."
        ),
        "majority_top3_definition": "Gold account is one of the three most frequent training-fold accounts.",
        "top3_year_macro_note": "TrustGate's published top-3 is top3_year_macro, the unweighted mean of yearly rates. top3_micro is line-weighted.",
        "cost_usd": None,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "baselines.json"
    path.write_text(json.dumps(report, indent=2))
    lookup = measured["models"]["merchant_lookup"]
    tfidf = measured["models"]["tfidf_logreg"]
    print(
        f"wrote {path} in {report['seconds']}s lookup={lookup['accuracy']:.6f} "
        f"tfidf={tfidf['accuracy']:.6f} ceiling={measured['label_ceiling_card_feed']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
