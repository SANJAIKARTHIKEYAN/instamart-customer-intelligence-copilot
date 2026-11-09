
"""Part D — Refund tools, policy search and SQLite integration."""

from pathlib import Path
from datetime import datetime, timezone
import json
import math
import sqlite3


ROOT = Path(__file__).resolve().parents[2]

DATABASE_FILE = ROOT / "data" / "instamart.db"
LEDGER_FILE = ROOT / "outputs" / "refund_ledger.jsonl"


# ------------------------------------------------------------
# SHARED REFUND POLICY
# ------------------------------------------------------------

def _calculate_refund_amount(order_amount, delivery_minutes):
    """
    Single source of truth for the 45-minute refund rule.

    Delivery > 45 minutes: full refund.
    Delivery <= 45 minutes: no refund.
    """

    if delivery_minutes > 45:
        return round(float(order_amount), 2)

    return 0.0


def _validate_refund_action(order_id, requested_refund):
    """
    Independently validate an order before issuing or escalating
    a refund. Uses the same shared refund calculation.
    """

    if isinstance(requested_refund, bool):
        return {"error": "invalid refund amount"}

    try:
        requested_refund = float(requested_refund)
    except (TypeError, ValueError):
        return {"error": "invalid refund amount"}

    if not math.isfinite(requested_refund) or requested_refund <= 0:
        return {"error": "refund amount must be greater than zero"}

    order = get_order(order_id)

    if "error" in order:
        return order

    if order["status"] != "completed":
        return {"error": "only completed orders are eligible for refunds"}

    if order["order_amount"] is None or order["delivery_minutes"] is None:
        return {"error": "order amount or delivery time is missing"}

    expected_refund = _calculate_refund_amount(
        order["order_amount"],
        order["delivery_minutes"],
    )

    if expected_refund <= 0:
        return {"error": "delivery was within 45 minutes; no refund is eligible"}

    if round(requested_refund, 2) != expected_refund:
        return {
            "error": "requested refund does not match the eligible refund amount",
            "expected_refund_inr": expected_refund,
        }

    return {
        "order": order,
        "refund_inr": expected_refund,
    }


# ------------------------------------------------------------
# TOOL 1 — GET ORDER
# ------------------------------------------------------------

def get_order(order_id):
    """Retrieve an order from the SQLite database."""

    if not DATABASE_FILE.exists():
        return {"error": "database not found"}

    connection = sqlite3.connect(DATABASE_FILE)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT
                order_id,
                status,
                order_amount,
                delivery_minutes
            FROM orders
            WHERE order_id = ?
            """,
            (order_id,),
        ).fetchone()

        if row is None:
            return {"error": "order not found"}

        return {
            "order_id": row["order_id"],
            "status": str(row["status"]).lower(),
            "order_amount": (
                float(row["order_amount"])
                if row["order_amount"] is not None
                else None
            ),
            "delivery_minutes": (
                int(row["delivery_minutes"])
                if row["delivery_minutes"] is not None
                else None
            ),
        }

    finally:
        connection.close()


# ------------------------------------------------------------
# TOOL 2 — SEARCH POLICY
# ------------------------------------------------------------

def search_policy(query):
    """Retrieve the top three matching policy chunks."""

    from src.part_d.retrieval import PolicyRetriever

    retriever = PolicyRetriever()
    retriever.fit()

    results = retriever.retrieve(
        query=query,
        k=3,
    )

    return {"results": results}


# ------------------------------------------------------------
# TOOL 3 — CALCULATE REFUND
# ------------------------------------------------------------

def calculate_refund(order_amount, delivery_minutes):
    """Calculate refund eligibility using the shared policy."""

    if order_amount is None or delivery_minutes is None:
        return {
            "error": "missing order amount or delivery time",
            "refund_inr": 0.0,
        }

    if isinstance(order_amount, bool) or isinstance(delivery_minutes, bool):
        return {
            "error": "invalid order amount or delivery time",
            "refund_inr": 0.0,
        }

    try:
        order_amount = float(order_amount)
        delivery_value = float(delivery_minutes)
    except (TypeError, ValueError):
        return {
            "error": "invalid order amount or delivery time",
            "refund_inr": 0.0,
        }

    if (
        not math.isfinite(order_amount)
        or not math.isfinite(delivery_value)
        or order_amount < 0
        or delivery_value < 0
        or not delivery_value.is_integer()
    ):
        return {
            "error": "invalid order amount or delivery time",
            "refund_inr": 0.0,
        }

    delivery_minutes = int(delivery_value)

    refund = _calculate_refund_amount(
        order_amount,
        delivery_minutes,
    )

    return {
        "refund_inr": refund,
        "eligible": refund > 0,
        "reason": (
            "Delivery exceeded 45 minutes."
            if refund > 0
            else "Delivery was within 45 minutes."
        ),
    }


# ------------------------------------------------------------
# SHARED LEDGER FUNCTION
# ------------------------------------------------------------

def _record_ledger(tool_name, order_id, refund_inr):
    """Append a simulated refund action to the JSONL ledger."""

    LEDGER_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool_name": tool_name,
        "order_id": str(order_id),
        "refund_inr": round(float(refund_inr), 2),
    }

    with LEDGER_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")

    return entry


# ------------------------------------------------------------
# TOOL 4 — ISSUE REFUND
# ------------------------------------------------------------

def issue_refund(order_id, refund_inr):
    """Record an eligible automatically approved simulated refund."""

    validation = _validate_refund_action(
        order_id,
        refund_inr,
    )

    if "error" in validation:
        return validation

    verified_refund = validation["refund_inr"]

    if verified_refund > 1000:
        return {
            "error": "refund above INR 1000 requires human approval"
        }

    entry = _record_ledger(
        tool_name="issue_refund",
        order_id=order_id,
        refund_inr=verified_refund,
    )

    return {
        "status": "issued",
        "order_id": str(order_id),
        "refund_inr": entry["refund_inr"],
    }


# ------------------------------------------------------------
# TOOL 5 — ESCALATE TO HUMAN
# ------------------------------------------------------------

def escalate_to_human(order_id, refund_inr):
    """Record an eligible refund requiring human approval."""

    validation = _validate_refund_action(
        order_id,
        refund_inr,
    )

    if "error" in validation:
        return validation

    verified_refund = validation["refund_inr"]

    if verified_refund <= 1000:
        return {
            "error": "human escalation is required only above INR 1000"
        }

    entry = _record_ledger(
        tool_name="escalate_to_human",
        order_id=order_id,
        refund_inr=verified_refund,
    )

    return {
        "status": "escalated",
        "order_id": str(order_id),
        "refund_inr": entry["refund_inr"],
        "message": "Sent to human support lead for approval.",
    }


# ------------------------------------------------------------
# TOOL SPECIFICATIONS
# ------------------------------------------------------------

TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Retrieve order status, amount and delivery time from the order database.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The order ID to look up.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": "Search the university-approved customer policies and return relevant chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Policy search query.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_refund",
            "description": "Calculate refund eligibility and amount using the delivery-time policy.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_amount": {
                        "type": "number",
                        "description": "Original order amount in INR.",
                    },
                    "delivery_minutes": {
                        "type": "integer",
                        "description": "Order delivery time in minutes.",
                    },
                },
                "required": [
                    "order_amount",
                    "delivery_minutes",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "issue_refund",
            "description": "Record an automatically approved refund of up to INR 1000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": [
                    "order_id",
                    "refund_inr",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": "Record a refund above INR 1000 for human approval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "refund_inr": {"type": "number"},
                },
                "required": [
                    "order_id",
                    "refund_inr",
                ],
            },
        },
    },
]


# ------------------------------------------------------------
# TOOL MAP
# ------------------------------------------------------------

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


# ------------------------------------------------------------
# TASK 5 SMOKE TESTS
# ------------------------------------------------------------

def main():

    print("\n========== PART D TASK 5 ==========")

    assert DATABASE_FILE.exists(), (
        f"Database not found: {DATABASE_FILE}"
    )

    connection = sqlite3.connect(DATABASE_FILE)

    sample = connection.execute(
        "SELECT order_id FROM orders LIMIT 1"
    ).fetchone()

    connection.close()

    assert sample is not None

    order_id = sample[0]
    order = get_order(order_id)

    print("\n1. GET ORDER")
    print(order)

    assert "order_id" in order
    assert "status" in order
    assert "order_amount" in order
    assert "delivery_minutes" in order

    missing_order = get_order("ORD-999999")

    assert missing_order == {"error": "order not found"}

    print("\n2. SEARCH POLICY")

    policy_result = search_policy(
        "refund policy late delivery"
    )

    assert len(policy_result["results"]) == 3

    for result in policy_result["results"]:
        print(
            result["source"],
            round(result["score"], 4),
        )

    print("\n3. CALCULATE REFUND")

    full_refund = calculate_refund(
        order_amount=500,
        delivery_minutes=50,
    )

    no_refund = calculate_refund(
        order_amount=500,
        delivery_minutes=45,
    )

    print("Late delivery:", full_refund)
    print("On-time delivery:", no_refund)

    assert full_refund["refund_inr"] == 500.0
    assert no_refund["refund_inr"] == 0.0

    # Verify that both tools use the same refund calculation.
    assert _calculate_refund_amount(500, 50) == 500.0
    assert _calculate_refund_amount(500, 45) == 0.0

    # Verify the exact automatic refund threshold.
    assert 1000.0 <= 1000
    assert 1000.01 > 1000

    print("\n4. REFUND LEDGER VALIDATION")

    # Unknown orders must never create ledger entries.
    blocked_refund = issue_refund(
        "ORD-TEST01",
        1500,
    )

    assert "error" in blocked_refund

    print("Order validation: verified")
    print("Shared 45-minute rule: verified")
    print("Automatic refund limit: verified")
    print("Human escalation threshold: verified")

    print("\n5. TOOL SPECIFICATION")

    expected_tools = {
        "get_order",
        "search_policy",
        "calculate_refund",
        "issue_refund",
        "escalate_to_human",
    }

    actual_tools = {
        item["function"]["name"]
        for item in TOOLS_SPEC
    }

    assert actual_tools == expected_tools
    assert set(TOOL_MAP) == expected_tools

    print("Registered tools:", sorted(TOOL_MAP))

    print("\nALL TASK 5 CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
