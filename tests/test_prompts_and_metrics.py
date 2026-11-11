import numpy as np
import pandas as pd

from glcode.dataset import candidate_accounts, load_lines, select_exemplars, stratified_sample
from glcode.metrics import bootstrap_ci
from glcode.prompts import user_message
from glcode.train import resolve_backend, resolve_targets


def test_bootstrap_percentile_interval():
    correct = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=float)
    assert bootstrap_ci(correct) == [0.3, 0.9]


def test_prompt_uses_card_feed_fields_only():
    row = pd.Series(
        {
            "merchant_raw": "ASDA STORES 4840",
            "department": "Services to People",
            "service_category": "Childrens",
            "amount": 12.4,
            "merchant_category": "SENTINEL_CATEGORY_XYZ",
            "gl_account": "Catering Supplies",
            "card_line_id": "t1",
        }
    )
    text = user_message(row, ["Catering Supplies", "Fuel"])
    assert "SENTINEL_CATEGORY_XYZ" not in text
    assert "Merchant: ASDA STORES 4840" in text
    assert "Department: Services to People" in text
    assert "Service: Childrens" in text
    assert "Amount GBP: 12.40" in text
    assert "Account:" not in text
    assert text.index("- Catering Supplies") < text.index("- Fuel")


def test_missing_department_is_unknown_not_the_baseline_symbol():
    row = pd.Series(
        {
            "merchant_raw": "SHOP",
            "department": np.nan,
            "service_category": None,
            "amount": 1,
            "gl_account": "Fuel",
        }
    )
    text = user_message(row, ["Fuel"])
    assert "Department: unknown" in text
    assert "Service: unknown" in text
    assert "∅" not in text


def test_exemplars_come_from_training_years_only():
    df = load_lines()
    train = df[df.statement_year < 2018]
    test = df[df.statement_year == 2018]
    exemplars = select_exemplars(train, 4)
    assert len(exemplars) == 4
    assert set(exemplars.card_line_id).isdisjoint(test.card_line_id)
    assert set(exemplars.statement_year) == {2017}
    text = user_message(test.iloc[0], candidate_accounts(train), exemplars)
    assert "Examples:" in text
    assert "Code this transaction." in text
    assert text.index("Examples:") < text.index("Code this transaction.")


def test_stratified_sample_is_deterministic_and_sized():
    df = load_lines()
    test = df[df.statement_year == 2018]
    a = stratified_sample(test, 100, seed=0)
    b = stratified_sample(test, 100, seed=0)
    c = stratified_sample(test, 100, seed=1)
    assert len(a) == 100
    assert a.card_line_id.tolist() == b.card_line_id.tolist()
    assert a.card_line_id.tolist() != c.card_line_id.tolist()
    assert set(a.card_line_id).issubset(set(test.card_line_id))


def test_backend_selection_without_claiming_a_gpu():
    assert resolve_backend("no", cuda=True) == "lora"
    assert resolve_backend("auto", cuda=False) == "lora"
    assert resolve_backend("no", cuda=False) == "lora"
    try:
        resolve_backend("yes", cuda=False)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    assert resolve_targets("attn") == ["q_proj", "k_proj", "v_proj", "o_proj"]
    assert "gate_proj" in resolve_targets("all-linear")
