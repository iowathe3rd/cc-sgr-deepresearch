from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from examples.call_center_mock.contracts import SCHEMA_VERSION

OUTPUT_PATH = Path(__file__).parent / "mock_data" / "call_center_records.json"

SEGMENTS = ["mass_market", "mass_affluent", "small_business", "private_banking"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
REGIONS = ["NE", "SE", "MW", "SW", "W"]
LANGUAGES = ["en", "es"]
CHANNELS = ["voice", "chat", "email"]

BRANCHES = [
    "boston_central",
    "atlanta_midtown",
    "chicago_loop",
    "phoenix_market",
    "seattle_central",
    "newark_main",
    "denver_market",
    "san_francisco_financial",
    "miami_biscayne",
    "dallas_commerce",
    "minneapolis_north",
    "orlando_west",
    "philadelphia_market",
    "charlotte_uptown",
]

QUEUE_PRODUCT_MAP = {
    "cards_support": ["credit_card", "debit_card"],
    "lending_support": ["mortgage", "auto_loan"],
    "small_business_support": ["merchant_services", "wire_transfer"],
    "retail_support": ["checking"],
    "fraud_support": ["credit_card"],
    "wealth_support": ["investment"],
    "payments_support": ["ach_transfer"],
    "digital_support": ["digital_banking"],
}

QUEUE_TEAM_MAP = {
    "cards_support": "cards",
    "lending_support": "lending",
    "small_business_support": "business_banking",
    "retail_support": "retail",
    "fraud_support": "fraud",
    "wealth_support": "wealth",
    "payments_support": "payments",
    "digital_support": "digital",
}

ISSUES = {
    "credit_card": [
        {
            "issue_category": "chargeback",
            "issue_subcategory": "merchant_dispute",
            "intent": "file_dispute",
            "tags": ["complaint", "regulatory_risk", "high_value"],
            "topics": ["chargeback", "merchant dispute", "timeline"],
        },
        {
            "issue_category": "fraud_alert",
            "issue_subcategory": "card_present",
            "intent": "report_fraud",
            "tags": ["fraud_risk", "complaint", "high_value"],
            "topics": ["fraud", "security", "charge"],
        },
        {
            "issue_category": "credit_limit",
            "issue_subcategory": "increase_request",
            "intent": "increase_limit",
            "tags": ["upsell"],
            "topics": ["limit", "income", "approval"],
        },
        {
            "issue_category": "travel_notice",
            "issue_subcategory": "international_travel",
            "intent": "set_travel_notice",
            "tags": ["travel_notice"],
            "topics": ["travel", "international", "alerts"],
        },
    ],
    "debit_card": [
        {
            "issue_category": "lost_card",
            "issue_subcategory": "replacement",
            "intent": "replace_card",
            "tags": ["card_replacement"],
            "topics": ["replacement", "shipping"],
        }
    ],
    "mortgage": [
        {
            "issue_category": "forbearance",
            "issue_subcategory": "hardship_request",
            "intent": "request_forbearance",
            "tags": ["hardship", "regulatory_risk", "repeat_caller"],
            "topics": ["hardship", "payment", "documentation"],
        }
    ],
    "auto_loan": [
        {
            "issue_category": "payoff_quote",
            "issue_subcategory": "interest_calculation",
            "intent": "payoff_quote",
            "tags": ["quote_request"],
            "topics": ["payoff", "interest", "schedule"],
        }
    ],
    "merchant_services": [
        {
            "issue_category": "pricing_dispute",
            "issue_subcategory": "interchange_fee",
            "intent": "fee_dispute",
            "tags": ["complaint", "pricing", "repeat_caller"],
            "topics": ["fees", "pricing", "contract"],
        }
    ],
    "wire_transfer": [
        {
            "issue_category": "payment_failure",
            "issue_subcategory": "incoming_wire_missing",
            "intent": "trace_wire",
            "tags": ["complaint", "high_value", "repeat_caller"],
            "topics": ["wire", "cash_flow", "trace"],
        }
    ],
    "checking": [
        {
            "issue_category": "fee_dispute",
            "issue_subcategory": "overdraft_fee",
            "intent": "fee_reversal",
            "tags": ["complaint", "fee_waiver"],
            "topics": ["overdraft", "fee", "policy"],
        },
        {
            "issue_category": "atm_issue",
            "issue_subcategory": "cash_withdrawal_error",
            "intent": "atm_dispute",
            "tags": ["complaint", "branch_followup"],
            "topics": ["atm", "cash", "dispute"],
        },
        {
            "issue_category": "service_complaint",
            "issue_subcategory": "agent_behavior",
            "intent": "file_complaint",
            "tags": ["complaint", "service_quality", "repeat_caller"],
            "topics": ["service", "hold_time", "respect"],
        },
    ],
    "investment": [
        {
            "issue_category": "portfolio_review",
            "issue_subcategory": "allocation_question",
            "intent": "portfolio_guidance",
            "tags": ["upsell", "high_value"],
            "topics": ["portfolio", "allocation", "market_outlook"],
        }
    ],
    "ach_transfer": [
        {
            "issue_category": "payment_pending",
            "issue_subcategory": "ach_delay",
            "intent": "ach_status",
            "tags": ["payment_delay", "repeat_caller"],
            "topics": ["ach", "pending", "processing"],
        }
    ],
    "digital_banking": [
        {
            "issue_category": "login_issue",
            "issue_subcategory": "password_reset",
            "intent": "reset_password",
            "tags": ["digital_deflection"],
            "topics": ["login", "reset"],
        }
    ],
}


def _random_date(rng: random.Random, start: datetime, end: datetime) -> datetime:
    delta = int((end - start).total_seconds())
    return start + timedelta(seconds=rng.randint(0, max(delta, 1)))


def _sentiment_from_tags(tags: list[str], rng: random.Random) -> float:
    base = 0.0
    if "complaint" in tags or "fraud_risk" in tags or "pricing" in tags or "service_quality" in tags:
        base -= 0.4
    if "hardship" in tags or "payment_delay" in tags:
        base -= 0.25
    if "upsell" in tags or "digital_deflection" in tags or "travel_notice" in tags:
        base += 0.25
    jitter = rng.uniform(-0.12, 0.12)
    return max(-1.0, min(1.0, base + jitter))


def _sentiment_label(score: float) -> str:
    if score >= 0.15:
        return "positive"
    if score <= -0.15:
        return "negative"
    return "neutral"


def _resolution_for_sentiment(score: float, rng: random.Random) -> str:
    if score < -0.3 and rng.random() < 0.6:
        return "pending"
    if rng.random() < 0.05:
        return "failed"
    return "resolved"


def _bool_by_probability(rng: random.Random, probability: float) -> bool:
    return rng.random() < probability


def generate_records(count: int, start: datetime, end: datetime, seed: int) -> list[dict]:
    rng = random.Random(seed)
    records: list[dict] = []

    for idx in range(count):
        call_id = f"C{idx + 1:05d}"
        queue = rng.choice(list(QUEUE_PRODUCT_MAP.keys()))
        product = rng.choice(QUEUE_PRODUCT_MAP[queue])
        issue = rng.choice(ISSUES[product])

        call_start = _random_date(rng, start, end)
        duration = rng.randint(240, 1800)
        call_end = call_start + timedelta(seconds=duration)

        segment = rng.choices(SEGMENTS, weights=[0.5, 0.2, 0.2, 0.1], k=1)[0]
        age_band = rng.choice(AGE_BANDS)
        churn_risk = rng.choices(["low", "medium", "high"], weights=[0.5, 0.3, 0.2], k=1)[0]

        tags = list(issue["tags"])
        if rng.random() < 0.2:
            tags.append("repeat_caller")
        if rng.random() < 0.12:
            tags.append("regulatory_risk")

        sentiment_score = _sentiment_from_tags(tags, rng)
        sentiment_label = _sentiment_label(sentiment_score)

        escalated = _bool_by_probability(
            rng, 0.15 + (0.4 if sentiment_score < -0.2 else 0) + (0.1 if "high_value" in tags else 0)
        )
        fcr = _bool_by_probability(rng, 0.75 if not escalated else 0.15)

        wait_time = rng.randint(15, 160)
        if escalated:
            wait_time += rng.randint(10, 80)
        hold_time = rng.randint(5, 140)

        resolution_status = _resolution_for_sentiment(sentiment_score, rng)
        resolution_days = rng.randint(0, 6) if resolution_status == "resolved" else None

        nps_last = None if rng.random() < 0.1 else max(0, min(10, int(6 + sentiment_score * 6)))

        record = {
            "call_id": call_id,
            "call_start": call_start.isoformat().replace("+00:00", "Z"),
            "call_end": call_end.isoformat().replace("+00:00", "Z"),
            "duration_sec": duration,
            "queue": queue,
            "channel": rng.choices(CHANNELS, weights=[0.75, 0.2, 0.05], k=1)[0],
            "language": rng.choices(LANGUAGES, weights=[0.85, 0.15], k=1)[0],
            "region": rng.choice(REGIONS),
            "branch": rng.choice(BRANCHES),
            "agent": {
                "agent_id": f"A{rng.randint(100, 799)}",
                "name": f"Agent {rng.randint(10, 99)}",
                "team": QUEUE_TEAM_MAP[queue],
                "tenure_months": rng.randint(3, 60),
                "location": rng.choice(["Austin", "Chicago", "Dallas", "Miami", "Phoenix", "Seattle"]),
                "quality_score": rng.randint(72, 96),
            },
            "caller": {
                "customer_id": f"U{rng.randint(10000, 99999)}",
                "segment": segment,
                "tenure_years": round(rng.uniform(0.5, 12.0), 1),
                "age_band": age_band,
                "products": [product] if rng.random() < 0.8 else [product, "savings"],
                "risk_score": round(rng.uniform(0.05, 0.65), 2),
                "churn_risk": churn_risk,
                "nps_last": nps_last,
                "delinquency_days": rng.choice([0, 0, 0, 5, 10, 15, 30]),
                "digital_usage_score": round(rng.uniform(0.2, 0.95), 2),
                "avg_monthly_balance_usd": round(rng.uniform(500, 300000), 2),
                "income_band": rng.choice(["<30k", "30k-50k", "40k-70k", "70k-100k", "100k-150k", "150k-200k", ">200k"]),
            },
            "interaction": {
                "wait_time_sec": wait_time,
                "hold_time_sec": hold_time,
                "transfer_count": rng.randint(0, 3),
                "after_call_work_sec": rng.randint(40, 240),
                "first_call_resolution": fcr,
                "escalated": escalated,
                "escalation_reason": rng.choice(["policy_dispute", "hardship_review", "pricing_dispute", "fraud_investigation", None])
                if escalated
                else None,
                "auth_method": rng.choice(["otp_sms", "otp_app", "knowledge_based", "voice_biometrics"]),
                "auth_passed": _bool_by_probability(rng, 0.96),
            },
            "case": {
                "product": product,
                "issue_category": issue["issue_category"],
                "issue_subcategory": issue["issue_subcategory"],
                "resolution": resolution_status,
                "resolution_time_days": resolution_days,
            },
            "nlp": {
                "sentiment_score": round(sentiment_score, 3),
                "sentiment_label": sentiment_label,
                "emotion_peaks": rng.sample(["frustration", "concern", "relief", "anger", "confidence", "anxiety"], k=2),
                "topics": issue["topics"],
                "tags": tags,
                "summary": f"Caller contacted {queue.replace('_', ' ')} regarding {issue['issue_category']}.",
                "call_intent": issue["intent"],
                "resolution_summary": "Resolution provided or follow-up scheduled.",
                "transcript": "Agent: Thanks for calling. Customer: I need help. Agent: I can assist with that.",
            },
            "quality": {
                "silence_ratio": round(rng.uniform(0.03, 0.2), 2),
                "overlap_ratio": round(rng.uniform(0.01, 0.08), 2),
                "speech_rate_wpm": rng.randint(130, 185),
                "script_adherence": round(rng.uniform(0.6, 0.95), 2),
            },
        }
        records.append(record)

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mock call center dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=str, default="2024-06-01")
    parser.add_argument("--end", type=str, default="2024-09-30")
    parser.add_argument("--count", type=int, default=240)
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "source_system": "mock_call_center",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "records": generate_records(args.count, start_dt, end_dt, args.seed),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Wrote {args.count} records to {args.output}")


if __name__ == "__main__":
    main()
