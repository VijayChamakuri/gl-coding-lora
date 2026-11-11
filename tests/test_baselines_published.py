from glcode.baselines import rolling_eval
from glcode.dataset import load_lines
from glcode.published import CARD_FEED, LABEL_CEILING_CARD_FEED


def test_merchant_lookup_matches_published_card_feed():
    measured = rolling_eval(load_lines(), models=("majority", "merchant_lookup", "merchant_top3"))
    lookup = measured["models"]["merchant_lookup"]
    published = CARD_FEED["merchant_lookup"]
    assert lookup["n"] == published["n"] == 11756
    assert lookup["accuracy"] == published["accuracy"]
    assert lookup["accuracy_ci"] == published["accuracy_ci"]
    assert lookup["macro_f1"] == published["macro_f1"]
    assert lookup["accuracy_seen_merchants"] == published["accuracy_seen_merchants"]
    assert lookup["accuracy_new_merchants"] == published["accuracy_new_merchants"]
    assert lookup["n_new_merchants"] == 3370
    majority = measured["models"]["majority"]
    assert majority["accuracy"] == CARD_FEED["majority"]["accuracy"]
    assert majority["accuracy_ci"] == CARD_FEED["majority"]["accuracy_ci"]
    assert measured["label_ceiling_card_feed"] == LABEL_CEILING_CARD_FEED
    top3 = measured["models"]["merchant_top3"]
    assert top3["accuracy"] == lookup["accuracy"]
    assert top3["top3_micro"] >= lookup["accuracy"]
