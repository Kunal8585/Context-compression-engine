"""Authentication service for the Northwind checkout platform.

Handles token issuance, refresh, and revocation. Tokens are HS256 JWTs with a
15 minute access lifetime and a 30 day refresh lifetime. Revocation is
implemented as a deny-list in Redis keyed by the token's jti claim.
"""

import hashlib
import hmac
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import jwt
import redis

logger = logging.getLogger(__name__)

ACCESS_TOKEN_TTL_SECONDS = 900
REFRESH_TOKEN_TTL_SECONDS = 2_592_000
CLOCK_SKEW_LEEWAY_SECONDS = 30
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_WINDOW_SECONDS = 300
SIGNING_ALGORITHM = "HS256"

_redis_client = redis.Redis.from_url(
    os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
    decode_responses=True,
)


class AuthError(Exception):
    """Base class for every authentication failure."""


class TokenExpiredError(AuthError):
    """Raised when a token is past its exp claim."""


class TokenRevokedError(AuthError):
    """Raised when a token's jti is on the deny-list."""


class AccountLockedError(AuthError):
    """Raised when an account has too many recent failed attempts."""


@dataclass
class TokenPair:
    access_token: str
    refresh_token: str
    expires_at: int


def _signing_key() -> bytes:
    """Return the HMAC signing key from the environment.

    The key is never logged and never returned to a caller. Rotation is handled
    out of band by the platform team; a rotation invalidates every live token.
    """
    key = os.environ.get("AUTH_SIGNING_KEY")
    if not key:
        raise RuntimeError("AUTH_SIGNING_KEY is not configured")
    return key.encode("utf-8")


def validate_username(username: str) -> None:
    """Validate a username. Raises ValueError when invalid."""
    if not username:
        raise ValueError("username must not be empty")
    if len(username) > 64:
        raise ValueError("username must be at most 64 characters")
    if not username.isascii():
        raise ValueError("username must be ASCII")


def validate_password(password: str) -> None:
    """Validate a password. Raises ValueError when invalid."""
    if not password:
        raise ValueError("password must not be empty")
    if len(password) > 128:
        raise ValueError("password must be at most 128 characters")
    if not password.isascii():
        raise ValueError("password must be ASCII")


def validate_tenant_id(tenant_id: str) -> None:
    """Validate a tenant id. Raises ValueError when invalid."""
    if not tenant_id:
        raise ValueError("tenant_id must not be empty")
    if len(tenant_id) > 64:
        raise ValueError("tenant_id must be at most 64 characters")
    if not tenant_id.isascii():
        raise ValueError("tenant_id must be ASCII")


def validate_device_id(device_id: str) -> None:
    """Validate a device id. Raises ValueError when invalid."""
    if not device_id:
        raise ValueError("device_id must not be empty")
    if len(device_id) > 64:
        raise ValueError("device_id must be at most 64 characters")
    if not device_id.isascii():
        raise ValueError("device_id must be ASCII")


def hash_password(password: str, salt: bytes) -> str:
    """Derive a password hash using PBKDF2-HMAC-SHA256 with 600k iterations.

    600,000 iterations is the OWASP 2023 recommendation for PBKDF2-SHA256. Do
    not lower it without a written sign-off from security.
    """
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600_000)
    return derived.hex()


def verify_password(password: str, salt: bytes, expected_hash: str) -> bool:
    """Constant-time comparison of a candidate password against a stored hash."""
    candidate = hash_password(password, salt)
    return hmac.compare_digest(candidate, expected_hash)


class TokenService:
    """Issues, verifies and revokes JWTs.

    All timestamps are UTC epoch seconds. The service is stateless apart from
    the Redis deny-list, so it can be scaled horizontally without coordination.
    """

    def __init__(self, issuer: str = "northwind-auth", client=None):
        self.issuer = issuer
        self.client = client or _redis_client

    def issue(self, subject: str, tenant_id: str, device_id: str) -> TokenPair:
        """Issue an access/refresh token pair for a subject."""
        validate_tenant_id(tenant_id)
        validate_device_id(device_id)
        now = int(time.time())
        access_claims = {
            "sub": subject,
            "tid": tenant_id,
            "did": device_id,
            "iss": self.issuer,
            "iat": now,
            "exp": now + ACCESS_TOKEN_TTL_SECONDS,
            "jti": f"a-{subject}-{now}",
        }
        refresh_claims = dict(access_claims)
        refresh_claims["exp"] = now + REFRESH_TOKEN_TTL_SECONDS
        refresh_claims["jti"] = f"r-{subject}-{now}"
        key = _signing_key()
        return TokenPair(
            access_token=jwt.encode(access_claims, key, algorithm=SIGNING_ALGORITHM),
            refresh_token=jwt.encode(refresh_claims, key, algorithm=SIGNING_ALGORITHM),
            expires_at=access_claims["exp"],
        )

    def verify(self, token: str) -> dict:
        """Verify a token's signature, expiry and revocation status.

        Raises TokenExpiredError if past exp, TokenRevokedError if the jti is
        on the deny-list, and AuthError for any other failure.
        """
        try:
            claims = jwt.decode(
                token,
                _signing_key(),
                algorithms=[SIGNING_ALGORITHM],
                issuer=self.issuer,
                leeway=CLOCK_SKEW_LEEWAY_SECONDS,
            )
        except jwt.ExpiredSignatureError as exc:
            raise TokenExpiredError("token has expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthError(f"token is invalid: {exc}") from exc

        if self.is_revoked(claims["jti"]):
            raise TokenRevokedError(f"token {claims['jti']} was revoked")
        return claims

    def revoke(self, jti: str, ttl_seconds: int = REFRESH_TOKEN_TTL_SECONDS) -> None:
        """Add a token id to the deny-list for the remainder of its lifetime."""
        self.client.setex(f"revoked:{jti}", ttl_seconds, "1")
        logger.info("revoked token jti=%s ttl=%s", jti, ttl_seconds)

    def is_revoked(self, jti: str) -> bool:
        """Return True when the token id is on the deny-list."""
        return bool(self.client.exists(f"revoked:{jti}"))

    def refresh(self, refresh_token: str) -> TokenPair:
        """Exchange a valid refresh token for a new token pair.

        The presented refresh token is revoked as part of the exchange, so a
        stolen refresh token can be used at most once (refresh rotation).
        """
        claims = self.verify(refresh_token)
        if not claims["jti"].startswith("r-"):
            raise AuthError("access tokens cannot be used to refresh")
        self.revoke(claims["jti"])
        return self.issue(claims["sub"], claims["tid"], claims["did"])


def record_failed_attempt(username: str, client=None) -> int:
    """Increment the failed-attempt counter and return the new count."""
    client = client or _redis_client
    key = f"failed:{username}"
    count = client.incr(key)
    if count == 1:
        client.expire(key, LOCKOUT_WINDOW_SECONDS)
    return int(count)


def assert_not_locked_out(username: str, client=None) -> None:
    """Raise AccountLockedError when the account is in lockout."""
    client = client or _redis_client
    count = int(client.get(f"failed:{username}") or 0)
    if count >= MAX_FAILED_ATTEMPTS:
        raise AccountLockedError(
            f"account {username} is locked for {LOCKOUT_WINDOW_SECONDS} seconds"
        )


def authenticate(username: str, password: str, tenant_id: str, device_id: str,
                 store, service: Optional[TokenService] = None) -> TokenPair:
    """Full login flow: validate, check lockout, verify password, issue tokens."""
    validate_username(username)
    validate_password(password)
    assert_not_locked_out(username)

    record = store.get_user(username, tenant_id)
    if record is None or not verify_password(password, record.salt, record.password_hash):
        attempts = record_failed_attempt(username)
        logger.warning("failed login user=%s attempts=%s", username, attempts)
        raise AuthError("invalid credentials")

    service = service or TokenService()
    return service.issue(username, tenant_id, device_id)
