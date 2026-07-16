"""Shared httpx connection pool limits for outbound API clients."""

from __future__ import annotations

import httpx

from app import config


def default_limits() -> httpx.Limits:
    return httpx.Limits(
        max_connections=config.HTTP_MAX_CONNECTIONS,
        max_keepalive_connections=config.HTTP_MAX_KEEPALIVE,
        keepalive_expiry=config.HTTP_KEEPALIVE_EXPIRY,
    )


def default_timeout(total: float = 30.0) -> httpx.Timeout:
    return httpx.Timeout(total, connect=min(10.0, total))
