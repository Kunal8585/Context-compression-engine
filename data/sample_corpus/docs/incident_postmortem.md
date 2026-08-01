# Incident Postmortem: Checkout Latency Spike (INC-4471)

**Status:** Resolved
**Severity:** SEV-2
**Duration:** 2024-03-14 09:12 UTC to 2024-03-14 11:48 UTC (2h 36m)
**Author:** Platform Reliability

## Summary

On 14 March 2024 the checkout service experienced a sustained latency spike.
The p99 latency for `POST /v1/checkout` rose from 240 ms to 8.4 s, and 3.2% of
checkout requests failed with a gateway timeout. The root cause was connection
pool exhaustion in the payment client after a configuration change reduced the
pool size from 64 to 8 connections.

## Impact

Approximately 14,200 checkout attempts were affected. Of those, 4,510 failed
outright and the remainder completed with degraded latency. Estimated revenue
impact is $86,000 based on the average order value of $19.10 and a 33% recovery
rate on retried carts.

Customer support received 212 tickets during the window. No data was lost and
no duplicate charges were issued, because the payment client uses idempotency
keys on every charge request.

## Timeline

All times UTC.

- **09:05** Deploy of `checkout-service` v2.31.0 begins in eu-west-1.
- **09:12** p99 latency alert fires. On-call is paged.
- **09:18** On-call acknowledges. Initial hypothesis is a slow database query.
- **09:31** Database dashboards ruled out; query latency is flat at 12 ms p99.
- **09:44** Engineer notices `pool_wait_ms` metric climbing to 7,900 ms.
- **10:02** Connection pool exhaustion confirmed as the proximate cause.
- **10:15** Config diff on v2.31.0 reveals `PAYMENT_POOL_SIZE` changed 64 to 8.
- **10:26** Decision made to roll back rather than hot-patch the config.
- **10:41** Rollback to v2.30.4 begins.
- **11:03** Rollback complete in eu-west-1. Latency begins recovering.
- **11:22** Rollback complete in us-east-1 and ap-south-1.
- **11:48** p99 latency back under 300 ms. Incident declared resolved.

## Root Cause

The v2.31.0 release included an unrelated refactor of the configuration loader.
During that refactor the default value for `PAYMENT_POOL_SIZE` was moved from
the deployment manifest into the application defaults, and the value was typed
as `8` rather than `64`. Because the deployment manifest no longer set the key,
the new default took effect in every region simultaneously.

The checkout service opens one payment connection per in-flight charge. At peak
the service sustains roughly 45 concurrent charges per pod. With a pool of 8,
requests queued for a connection, and the queue wait was counted inside the
request timeout of 8 seconds. Once the queue wait exceeded that budget, requests
began failing with gateway timeouts.

```yaml
# The offending diff in config/defaults.yaml
payment:
-  pool_size: 64
+  pool_size: 8
   timeout_ms: 8000
   max_retries: 3
```

## Contributing Factors

1. **No load test on the release candidate.** The v2.31.0 RC was validated with
   functional tests only. A 30-minute soak at production concurrency would have
   surfaced the queue wait immediately.
2. **Pool metrics were not alerted on.** The `pool_wait_ms` metric existed and
   was recorded, but no alert was attached to it. The team discovered the
   climbing value only by manually browsing a dashboard.
3. **Configuration defaults are not diffed in review.** The pull request showed
   the refactor as a large file move, and the single changed literal was not
   visible in the diff summary that reviewers read.
4. **Simultaneous multi-region rollout.** The deploy pipeline rolls out to all
   three regions in parallel, so there was no canary region that would have
   isolated the blast radius.

## What Went Well

The rollback tooling worked exactly as designed and completed in 22 minutes
across three regions. Idempotency keys on the payment path meant that customer
retries never produced duplicate charges, which avoided what would otherwise
have been a much more serious financial incident. On-call response time was
6 minutes from page to acknowledgement, well inside the 15-minute SEV-2 target.

## Action Items

| ID | Action | Owner | Due |
|----|--------|-------|-----|
| A-1 | Add an alert on `pool_wait_ms` p99 above 500 ms | Platform | 2024-03-21 |
| A-2 | Add a 30-minute soak test at 1.5x peak concurrency to the release gate | Checkout | 2024-03-28 |
| A-3 | Make eu-west-1 a canary region with a 20-minute bake | Platform | 2024-04-04 |
| A-4 | Require explicit review of `config/defaults.yaml` changes via CODEOWNERS | Platform | 2024-03-18 |
| A-5 | Emit a startup log line with every resolved config value | Checkout | 2024-03-25 |

## Lessons Learned

Configuration is code, and a one-character change to a default can have the same
blast radius as a bad deploy. Our review process treats config and code
differently, and this incident shows that distinction is not justified.

The second lesson is about metrics that exist but are not alerted on. The
`pool_wait_ms` metric had been recorded for eleven months. It told the exact
story of this incident, in real time, and nobody was watching it. A metric with
no alert is documentation, not observability.

## Appendix: Detection Query

The following query would have detected the condition 4 minutes before the
latency alert fired, and is now the basis for action item A-1.

```sql
SELECT
  service,
  region,
  percentile_cont(0.99) WITHIN GROUP (ORDER BY pool_wait_ms) AS p99_wait
FROM connection_pool_metrics
WHERE observed_at > now() - interval '5 minutes'
GROUP BY service, region
HAVING percentile_cont(0.99) WITHIN GROUP (ORDER BY pool_wait_ms) > 500;
```
