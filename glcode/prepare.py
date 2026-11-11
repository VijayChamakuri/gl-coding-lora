"""Download Stockport purchase-card files and rebuild the labelled population.

The committed reference file is the JudgeGate export of TrustGate's dbt model
`int_stockport__labelled_lines` (14,851 lines). `--download` fetches the
publisher's monthly files from data.gov.uk and applies the same cleaning
rules. It does not replace the reference unless the rebuild matches it.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pandas as pd

from .normalize import normalize_merchant, period_from_url
from .paths import PROCESSED, PURPOSE_MAP, RAW, REFERENCE

CKAN = "https://ckan.publishing.service.gov.uk/api/3/action/package_show?id=purchase-card-transactions"
UA = {"User-Agent": "glcode research (open data; OGL v3; local reproduction of TrustGate labelled lines)"}

COLUMN_MAP = {
    "organisation name": "organisation",
    "department where exp. incurred": "department",
    "service category label": "service_category",
    "supplier (beneficiary)": "merchant_raw",
    "purpose of expenditure": "purpose",
    "procurement (merchant) category": "merchant_category",
    "purchase date": "purchase_date",
    "settlement date": "settlement_date",
    "transaction number": "transaction_number",
    "net amount (£)": "amount_net",
    "net amount (�)": "amount_net",
    "gross amount (£)": "amount_gross",
    "gross amount (�)": "amount_gross",
}


def _get(url: str, tries: int = 4) -> bytes:
    import time

    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as response:
                return response.read()
        except Exception as exc:  # noqa: BLE001 - retry transient network errors
            last = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"failed to fetch {url}: {last}")


def list_resources() -> list[dict]:
    meta = json.loads(_get(CKAN))["result"]
    out = []
    for resource in meta["resources"]:
        if (resource.get("format") or "").upper() != "CSV":
            continue
        out.append(
            {
                "url": resource["url"],
                "period": period_from_url(resource["url"]),
                "resource_id": resource.get("id"),
            }
        )
    return sorted(out, key=lambda r: (r["period"] or "9999", r["url"]))


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _map_header(cell: str) -> str:
    key = cell.strip().lower()
    if key in COLUMN_MAP:
        return COLUMN_MAP[key]
    if "gross amount" in key:
        return "amount_gross"
    if "net amount" in key:
        return "amount_net"
    return key


def rows_from_bytes(raw: bytes) -> list[list[str]]:
    if raw[:2] == b"PK":
        import openpyxl

        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        out = []
        for row in wb.worksheets[0].iter_rows(values_only=True):
            out.append(
                ["" if v is None else (v.strftime("%d/%m/%Y") if hasattr(v, "strftime") else str(v)) for v in row]
            )
        return out
    return list(csv.reader(io.StringIO(decode(raw), newline="")))


def parse_rows(rows: list[list[str]], period: str, source_file: str) -> list[dict]:
    hdr_idx = next(i for i, r in enumerate(rows[:10]) if any(c.strip().lower() == "organisation name" for c in r))
    header = [_map_header(c) for c in rows[hdr_idx]]
    if "amount_gross" in header:
        version = "v3_gross_6col"
    elif "transaction_number" in header:
        version = "v1_net_10col"
    else:
        version = "v2_net_8col"
    out = []
    for i, row in enumerate(rows[hdr_idx + 1 :], start=hdr_idx + 2):
        rec = {h: (row[j].strip() if j < len(row) else "") for j, h in enumerate(header) if h}
        if not rec.get("merchant_raw") and not rec.get("amount_net") and not rec.get("amount_gross"):
            continue
        amount = rec.get("amount_net") or rec.get("amount_gross") or ""
        out.append(
            {
                "period": period,
                "source_file": source_file,
                "source_row": i,
                "schema_version": version,
                "department": rec.get("department") or None,
                "service_category": rec.get("service_category") or None,
                "merchant_raw": rec.get("merchant_raw") or "",
                "purpose": rec.get("purpose") or None,
                "merchant_category": rec.get("merchant_category") or None,
                "purchase_date": rec.get("purchase_date") or None,
                "amount": amount.replace(",", "").replace("£", "").replace("\xa3", "") or None,
                "amount_basis": "gross" if rec.get("amount_gross") else "net",
            }
        )
    return out


def _parse_date(value):
    """Match DuckDB try_strptime('%d.%m.%Y', '%d/%m/%Y', '%d-%m-%Y').

    DuckDB's %Y accepts a two-digit year as year 00xx. The February 2020 file
    contains one '28.02.20', and the published export stores it as 0020-02-28.
    """
    if value is None or str(value).strip() == "":
        return pd.NaT
    text = str(value).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    match = re.fullmatch(r"(\d{1,2})([./-])(\d{1,2})\2(\d{2})", text)
    if match:
        padded = f"{int(match.group(1)):02d}{match.group(2)}{int(match.group(3)):02d}{match.group(2)}{int(match.group(4)):04d}"
        fmt = f"%d{match.group(2)}%m{match.group(2)}%Y"
        try:
            return datetime.strptime(padded, fmt).date()
        except ValueError:
            return pd.NaT
    return pd.NaT


def _parse_amount(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("£", "").replace("\xa3", "")
    if text == "":
        return None
    try:
        return float(Decimal(text).quantize(Decimal("0.01")))
    except (InvalidOperation, ValueError):
        return None


def load_purpose_map(path: Path = PURPOSE_MAP) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["is_placeholder"] = table["is_placeholder"].str.lower().eq("true")
    table["gl_account"] = table["gl_account"].replace({"": pd.NA})
    table["raw_purpose_key"] = table["raw_purpose_key"].str.lower().str.strip()
    return table[["raw_purpose_key", "gl_account", "is_placeholder"]]


def labelled_from_records(records: list[dict], purpose_map: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    purpose_map = load_purpose_map() if purpose_map is None else purpose_map
    frame = pd.DataFrame.from_records(records)
    frame["statement_month"] = pd.to_datetime(frame["period"].fillna("") + "-01", errors="coerce")
    frame.loc[frame["period"].isna() | (frame["period"] == ""), "statement_month"] = pd.NaT
    frame["purchase_date"] = frame["purchase_date"].map(_parse_date)
    frame["amount"] = frame["amount"].map(_parse_amount)
    frame["merchant_raw"] = frame["merchant_raw"].fillna("").map(lambda s: str(s).strip())
    frame["merchant_key"] = frame["merchant_raw"].map(normalize_merchant)
    frame["department"] = frame["department"].map(lambda v: v.strip() if isinstance(v, str) and v.strip() else pd.NA)
    frame["service_category"] = frame["service_category"].map(
        lambda v: v.strip() if isinstance(v, str) and v.strip() else pd.NA
    )
    frame["merchant_category"] = frame["merchant_category"].map(
        lambda v: v.strip().upper() if isinstance(v, str) and v.strip() else pd.NA
    )
    frame["raw_purpose_key"] = frame["purpose"].map(lambda v: v.strip().lower() if isinstance(v, str) and v.strip() else pd.NA)
    frame["card_line_id"] = frame["source_file"] + ":" + frame["source_row"].astype(str)
    merged = frame.merge(purpose_map, on="raw_purpose_key", how="left")
    merged["is_placeholder"] = merged["is_placeholder"].fillna(False).astype(bool)
    stats = {
        "parsed_rows": int(len(frame)),
        "with_gl_account": int(merged["gl_account"].notna().sum()),
        "placeholder": int(merged["is_placeholder"].sum()),
        "refund_or_null_amount": int((merged["amount"].isna() | (merged["amount"] < 0)).sum()),
    }
    keep = merged["gl_account"].notna() & ~merged["is_placeholder"] & merged["amount"].notna() & (merged["amount"] >= 0)
    labelled = merged.loc[keep, [
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
    ]].copy()
    labelled["statement_month"] = pd.to_datetime(labelled["statement_month"])
    labelled["purchase_date"] = pd.to_datetime(labelled["purchase_date"])
    stats["labelled_lines"] = int(len(labelled))
    return labelled.reset_index(drop=True), stats


def _canon(df: pd.DataFrame) -> Counter:
    work = pd.DataFrame(
        {
            "statement_month": pd.to_datetime(df["statement_month"]).dt.strftime("%Y-%m-%d").fillna(""),
            "purchase_date": pd.to_datetime(df["purchase_date"]).dt.strftime("%Y-%m-%d").fillna(""),
            "merchant_raw": df["merchant_raw"].fillna(""),
            "merchant_key": df["merchant_key"].fillna(""),
            "merchant_category": df["merchant_category"].fillna(""),
            "department": df["department"].fillna(""),
            "service_category": df["service_category"].fillna(""),
            "amount": df["amount"].map(lambda v: "" if pd.isna(v) else f"{float(v):.2f}"),
            "gl_account": df["gl_account"].fillna(""),
        }
    )
    keys = work.astype(str).agg("|".join, axis=1)
    return Counter(keys.tolist())


def compare_to_reference(labelled: pd.DataFrame, reference: pd.DataFrame) -> dict:
    built = _canon(labelled)
    ref = _canon(reference)
    only_built = sum((built - ref).values())
    only_ref = sum((ref - built).values())
    return {
        "reference_rows": int(len(reference)),
        "rebuilt_rows": int(len(labelled)),
        "rows_only_in_rebuild": int(only_built),
        "rows_only_in_reference": int(only_ref),
        "content_match": only_built == 0 and only_ref == 0 and len(labelled) == len(reference),
    }


def download(dest: Path = RAW, workers: int = 6) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": []}
    have = {item["url"]: item for item in manifest["files"]}
    todo = [res for res in list_resources() if res["url"] not in have]

    def fetch(res):
        raw = _get(res["url"])
        digest = hashlib.sha256(raw).hexdigest()
        # TrustGate names files from the URL hash, which is what card_line_id uses.
        url_hash = hashlib.sha256(res["url"].encode()).hexdigest()[:8]
        fname = f"{res['period'] or 'unknown'}_{url_hash}.csv"
        (dest / fname).write_bytes(raw)
        return {
            "period": res["period"],
            "url": res["url"],
            "file": fname,
            "bytes": len(raw),
            "sha256": digest,
        }

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, res) for res in todo]
            for fut in as_completed(futures):
                manifest["files"].append(fut.result())
    manifest["files"].sort(key=lambda f: (f["period"] or "9999", f["url"]))
    manifest["source"] = CKAN
    manifest["licence"] = "Open Government Licence v3.0"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def rebuild_from_manifest(dest: Path = RAW) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((dest / "manifest.json").read_text())
    records: list[dict] = []
    seen: set[str] = set()
    skipped_duplicate = 0
    for item in manifest["files"]:
        if item["sha256"] in seen:
            skipped_duplicate += 1
            continue
        seen.add(item["sha256"])
        raw = (dest / item["file"]).read_bytes()
        url_hash = hashlib.sha256(item["url"].encode()).hexdigest()[:8]
        source_file = f"{item['period'] or 'unknown'}_{url_hash}.csv"
        records.extend(parse_rows(rows_from_bytes(raw), item["period"], source_file))
    labelled, stats = labelled_from_records(records)
    stats["files"] = len(manifest["files"])
    stats["duplicate_sha_skipped"] = skipped_duplicate
    return labelled, stats


def run_download() -> dict:
    manifest = download()
    labelled, stats = rebuild_from_manifest()
    reference = pd.read_csv(REFERENCE, parse_dates=["statement_month", "purchase_date"])
    comparison = compare_to_reference(labelled, reference)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    report = {"download_files": len(manifest["files"]), "build": stats, "comparison": comparison}
    (PROCESSED / "rebuild_report.json").write_text(json.dumps(report, indent=2))
    from .paths import RESULTS

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rebuild_report.json").write_text(json.dumps(report, indent=2))
    if comparison["content_match"]:
        labelled.to_csv(PROCESSED / "labelled_lines.csv.gz", index=False)
    return report
