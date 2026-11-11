"""Figures published by TrustGate. Not measurements from this repository.

Source file: results/real/gl_benchmark.json in
https://github.com/VijayChamakuri/trustgate at the commit below.
Card-feed setting, rolling years 2018–2022, pooled over 11,756 lines.
Top-3 there is the unweighted mean of the five yearly top-3 rates.
"""

TRUSTGATE_COMMIT = "66cf7ab2fb866140a75430415ac94dfa1f1907cb"
JUDGEGATE_COMMIT = "ff0b8f6b8d3ec119c3f3d2131dc1e8a8b190bb4e"
TRUSTGATE_GL_JSON = (
    "https://github.com/VijayChamakuri/trustgate/blob/"
    f"{TRUSTGATE_COMMIT}/results/real/gl_benchmark.json"
)

# Autonomy-rollout top-3 (a different window: live months after a 2020 shadow
# period, model refit quarterly). Do not compare it to the rolling-benchmark top-3.
TRUSTGATE_TOP3_LIVE = 0.8640878201777313
TRUSTGATE_ACCURACY_LIVE = 0.6664924202822792

CARD_FEED = {
    "majority": {
        "n": 11756,
        "accuracy": 0.16970057842803674,
        "accuracy_ci": [0.16306566859476013, 0.17608242599523646],
        "macro_f1": 0.0190146121766018,
        "top3": None,
        "accuracy_seen_merchants": 0.19222513713331743,
        "accuracy_new_merchants": 0.11364985163204748,
    },
    "merchant_lookup": {
        "n": 11756,
        "accuracy": 0.5450833616876488,
        "accuracy_ci": [0.53623681524328, 0.55435522286492],
        "macro_f1": 0.31269065179412053,
        "top3": None,
        "accuracy_seen_merchants": 0.7005723825423325,
        "accuracy_new_merchants": 0.15816023738872403,
    },
    "tfidf_logreg": {
        "n": 11756,
        "accuracy": 0.6092208234093229,
        "accuracy_ci": [0.6000297720313031, 0.6184097482136781],
        "macro_f1": 0.35416892500608343,
        "top3": 0.8067477572193578,
        "accuracy_seen_merchants": 0.7148819461006439,
        "accuracy_new_merchants": 0.3462908011869436,
    },
    "tfidf+embed_logreg": {
        "n": 11756,
        "accuracy": 0.624106839060905,
        "accuracy_ci": [0.6150880401497107, 0.6333787002381762],
        "macro_f1": 0.3787634063780648,
        "top3": 0.8292839883606696,
        "accuracy_seen_merchants": 0.7114238015740519,
        "accuracy_new_merchants": 0.40682492581602375,
    },
}

LABEL_CEILING_CARD_FEED = 0.8608174533701434
LABEL_CEILING_UPSTREAM = 0.9907076964514174
