/**
 * Payment gateway client for the Northwind checkout platform.
 * Wraps the Stripe-compatible REST API with retries and idempotency keys.
 */

const DEFAULT_TIMEOUT_MS = 8000;
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY_MS = 250;
const IDEMPOTENCY_HEADER = "Idempotency-Key";

class PaymentError extends Error {
  constructor(message, code, retryable) {
    super(message);
    this.name = "PaymentError";
    this.code = code;
    this.retryable = retryable;
  }
}

// Sleep helper used by the exponential backoff loop.
function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRetryable(status) {
  return status === 429 || status === 502 || status === 503 || status === 504;
}

function buildIdempotencyKey(orderId, attempt) {
  return `${orderId}:${attempt}`;
}

class PaymentClient {
  constructor(baseUrl, apiKey, options = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeoutMs = options.timeoutMs || DEFAULT_TIMEOUT_MS;
    this.maxRetries = options.maxRetries || MAX_RETRIES;
  }

  async request(path, body, idempotencyKey) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.timeoutMs);
    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        signal: controller.signal,
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${this.apiKey}`,
          [IDEMPOTENCY_HEADER]: idempotencyKey,
        },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new PaymentError(
          payload.message || `gateway returned ${response.status}`,
          payload.code || String(response.status),
          isRetryable(response.status)
        );
      }
      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  // Charge a card with exponential backoff on retryable gateway failures.
  async charge(orderId, amountCents, currency, source) {
    let lastError = null;
    for (let attempt = 0; attempt < this.maxRetries; attempt += 1) {
      try {
        return await this.request(
          "/v1/charges",
          { order_id: orderId, amount: amountCents, currency, source },
          buildIdempotencyKey(orderId, attempt)
        );
      } catch (error) {
        lastError = error;
        if (!(error instanceof PaymentError) || !error.retryable) {
          throw error;
        }
        await sleep(RETRY_BASE_DELAY_MS * 2 ** attempt);
      }
    }
    throw lastError;
  }

  async refund(chargeId, amountCents) {
    return this.request(
      "/v1/refunds",
      { charge_id: chargeId, amount: amountCents },
      `refund:${chargeId}`
    );
  }

  async capture(chargeId) {
    return this.request("/v1/captures", { charge_id: chargeId }, `capture:${chargeId}`);
  }
}

module.exports = { PaymentClient, PaymentError, isRetryable, buildIdempotencyKey };
