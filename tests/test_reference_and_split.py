import pandas as pd

from glcode.dataset import TEST_YEARS, YEAR_COUNTS, candidate_accounts, iter_folds, load_lines, sha256_file
from glcode.normalize import normalize_merchant, period_from_url
from glcode.paths import REFERENCE, REFERENCE_SHA256


def test_reference_hash_and_years():
    assert sha256_file(REFERENCE) == REFERENCE_SHA256
    df = load_lines()
    assert len(df) == 14851
    assert df.gl_account.nunique() == 21
    assert df.merchant_key.nunique() == 1992
    assert df.statement_year.value_counts().sort_index().to_dict() == YEAR_COUNTS
    assert df.card_line_id.is_unique


def test_merchant_key_matches_every_published_row():
    df = pd.read_csv(REFERENCE)
    mismatch = df[df.merchant_raw.map(normalize_merchant) != df.merchant_key]
    assert mismatch.empty, mismatch[["merchant_raw", "merchant_key"]].head(5).to_dict("records")


def test_period_parser():
    assert period_from_url("https://x/Purchasing+Card+-+February+17.csv") == "2017-02"
    assert period_from_url("https://x/Purchase+Card+-+August+2026.csv") == "2026-08"
    assert period_from_url("https://x/Purchasing+Card+-+March+17+.csv") == "2017-03"


def test_rolling_folds_are_the_trustgate_split():
    df = load_lines()
    seen = []
    for year, train, test in iter_folds(df):
        assert year in TEST_YEARS
        assert train.statement_year.max() < year
        assert set(test.statement_year.unique()) == {year}
        assert set(train.card_line_id).isdisjoint(test.card_line_id)
        seen.append(len(test))
        assert set(candidate_accounts(train)).issubset(set(df.gl_account.unique()))
    assert seen == [YEAR_COUNTS[y] for y in TEST_YEARS]
    assert sum(seen) == 11756
