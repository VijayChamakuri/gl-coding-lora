from pathlib import Path

import openpyxl

from glcode.prepare import _parse_date, parse_rows, rows_from_bytes

FIX = Path(__file__).parent / "fixtures"


def test_two_digit_year_matches_the_published_export():
    parsed = _parse_date("28.02.20")
    assert str(parsed) == "0020-02-28"
    assert str(_parse_date("28.02.2020")) == "2020-02-28"


def test_v1_layout_keeps_labels_and_drops_blank_rows():
    rows = parse_rows(rows_from_bytes((FIX / "stockport_v1.csv").read_bytes()), "2019-05", "2019-05_test.csv")
    assert len(rows) == 2
    assert rows[0]["schema_version"] == "v1_net_10col"
    assert rows[0]["purpose"] == "Furniture and Equipment"
    assert rows[1]["amount"] == "1007.45" and rows[1]["amount_basis"] == "net"


def test_v3_layout_skips_banner_and_has_no_label():
    rows = parse_rows(rows_from_bytes((FIX / "stockport_v3.csv").read_bytes()), "2024-06", "2024-06_test.csv")
    assert [r["merchant_raw"] for r in rows] == ["INVISIONAPP, INC", "FASTHOSTS"]
    assert all(r["schema_version"] == "v3_gross_6col" and r["purpose"] is None for r in rows)
    assert rows[1]["amount"] == "1437.19" and rows[1]["amount_basis"] == "gross"


def test_xlsx_published_with_csv_extension(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["MAY 2024"])
    ws.append([])
    ws.append(
        [
            "Organisation Name",
            "Supplier (Beneficiary)",
            "Procurement (Merchant) Category",
            "Purchase Date",
            "Settlement Date",
            "Gross Amount (£)",
        ]
    )
    ws.append(["STOCKPORT MBC", "TRAINLINE", "PASSENGER RAILWAYS", "02/05/2024", "16/06/2024", 54.1])
    path = tmp_path / "may.csv"
    wb.save(path)
    rows = parse_rows(rows_from_bytes(path.read_bytes()), "2024-05", "2024-05_x.csv")
    assert len(rows) == 1 and rows[0]["merchant_raw"] == "TRAINLINE" and rows[0]["amount"] == "54.1"
