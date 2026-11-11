"""Card-feed prompts.

The model sees the merchant string, department, service and amount. It does
not see the procurement category: council staff assigned that field with the
label, and TrustGate treats it as leakage. Missing department or service is
the word "unknown". The allowed accounts are those observed in the training
years of the fold, listed alphabetically.
"""

from __future__ import annotations

import pandas as pd

SYSTEM = (
    "You assign UK council purchase-card transactions to exactly one general-ledger account. "
    "Reply with the account name and nothing else."
)


def _field(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value) or value == "":
        return "unknown"
    return str(value)


def render_transaction(row) -> str:
    return "\n".join(
        [
            f"Merchant: {_field(row.merchant_raw)}",
            f"Department: {_field(row.department)}",
            f"Service: {_field(row.service_category)}",
            f"Amount GBP: {float(row.amount):.2f}",
        ]
    )


def user_message(row, accounts: list[str], exemplars: pd.DataFrame | None = None) -> str:
    lines = ["Allowed accounts:", *[f"- {account}" for account in accounts], ""]
    if exemplars is not None and len(exemplars):
        lines.append("Examples:")
        for exemplar in exemplars.itertuples(index=False):
            lines.append(render_transaction(exemplar))
            lines.append(f"Account: {exemplar.gl_account}")
            lines.append("")
    lines.append("Code this transaction.")
    lines.append(render_transaction(row))
    return "\n".join(lines)


def messages(user: str, assistant: str | None = None) -> list[dict]:
    convo = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    if assistant is not None:
        convo.append({"role": "assistant", "content": assistant})
    return convo
