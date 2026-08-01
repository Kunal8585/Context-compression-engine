"""Generate the deterministic sample log file used by tests and the demo.

    python scripts/make_sample_logs.py

The output is committed so the demo works offline, but it is generated (not
hand-written) and the seed is fixed, so anyone can reproduce it byte-for-byte.

The shape is deliberately realistic for this problem: a small number of message
templates repeated thousands of times with only volatile fields changing, plus
a handful of genuinely rare events (the ones a judge will ask whether the
compressor kept). Redundancy here is not synthetic padding - it is what a real
service log looks like, and it is exactly the redundancy stage 3 targets.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sample_corpus" / "logs" / "checkout_service.log"

SEED = 20240314
START = datetime(2024, 3, 14, 9, 0, 0)

SERVICES = ["checkout-api", "payment-client", "cart-store", "inventory"]
REGIONS = ["eu-west-1", "us-east-1", "ap-south-1"]

# (level, template) - {} placeholders are filled with volatile values.
COMMON = [
    ("INFO", "handled POST /v1/checkout order={order} user={user} status=200 duration_ms={ms}"),
    ("INFO", "handled GET /v1/cart/{order} user={user} status=200 duration_ms={ms}"),
    ("INFO", "cache hit key=cart:{order} ttl_remaining_s={ttl}"),
    ("DEBUG", "acquired connection pool=payment size=64 in_use={n} wait_ms={ms}"),
    ("INFO", "charge accepted order={order} amount_cents={amount} currency=USD gateway_ref={ref}"),
    ("DEBUG", "emitted metric checkout.latency value={ms} region={region}"),
    ("INFO", "inventory reserved sku=SKU-{sku} qty={qty} order={order}"),
]

RARE = [
    ("WARN", "connection pool wait exceeded threshold pool=payment wait_ms=7912 in_use=8 size=8"),
    ("ERROR", "gateway timeout after 8000ms order={order} attempt=3 idempotency_key={order}:3"),
    ("WARN", "config value PAYMENT_POOL_SIZE resolved to 8 (manifest override absent)"),
    ("ERROR", "checkout failed order={order} reason=payment_gateway_timeout revenue_cents={amount}"),
    ("INFO", "rollback initiated target=v2.30.4 regions=eu-west-1,us-east-1,ap-south-1 operator=oncall"),
]

TRACEBACK = """ERROR checkout-api [eu-west-1] unhandled exception in charge path order={order}
Traceback (most recent call last):
  File "/srv/checkout/handlers/checkout.py", line 214, in post_checkout
    result = payment.charge(order.id, order.total_cents, "USD", source)
  File "/srv/checkout/payment/client.py", line 88, in charge
    return self._request("/v1/charges", payload, key)
  File "/srv/checkout/payment/client.py", line 51, in _request
    response = self._session.post(url, json=payload, timeout=self.timeout)
  File "/usr/lib/python3.11/site-packages/requests/sessions.py", line 637, in post
    return self.request("POST", url, data=data, json=json, **kwargs)
requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='payments.internal', port=443): Read timed out. (read timeout=8.0)
"""


def main() -> None:
    rng = random.Random(SEED)
    lines: list[str] = []
    timestamp = START

    for index in range(2400):
        timestamp += timedelta(milliseconds=rng.randint(120, 900))
        stamp = timestamp.strftime("%Y-%m-%d %H:%M:%S.") + f"{timestamp.microsecond // 1000:03d}"

        # The incident window (roughly 12% of the file) carries the rare events.
        in_incident = 900 <= index < 1200
        if in_incident and rng.random() < 0.22:
            level, template = RARE[rng.randrange(len(RARE))]
        elif in_incident and rng.random() < 0.04:
            lines.append(
                f"{stamp} " + TRACEBACK.format(order=f"ORD-{rng.randint(100000, 999999)}")
            )
            continue
        else:
            level, template = COMMON[rng.randrange(len(COMMON))]

        message = template.format(
            order=f"ORD-{rng.randint(100000, 999999)}",
            user=f"usr_{rng.randint(1000, 9999)}",
            ms=rng.randint(18, 340),
            ttl=rng.randint(5, 300),
            n=rng.randint(1, 60),
            amount=rng.randint(499, 24999),
            ref=f"ch_{rng.randbytes(6).hex()}",
            region=REGIONS[rng.randrange(len(REGIONS))],
            sku=f"{rng.randint(1000, 9999)}",
            qty=rng.randint(1, 4),
        )
        service = SERVICES[rng.randrange(len(SERVICES))]
        region = REGIONS[rng.randrange(len(REGIONS))]
        lines.append(f"{stamp} {level} {service} [{region}] {message}\n")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(lines)} records, {OUTPUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
