from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, TypeVar

import redis
from redis.exceptions import BusyLoadingError, ClusterDownError, ConnectionError as RedisConnectionError, TimeoutError as RedisTimeoutError, TryAgainError
from requests import exceptions as requests_exceptions
from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError, InternalServerError, PermissionDeniedError, RateLimitError, UnprocessableEntityError


T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 8.0


class ExternalServiceError(RuntimeError):
    pass


def _error_message(exc: Exception) -> str:
    return str(exc).strip() or exc.__class__.__name__


def is_context_length_error(exc: Exception) -> bool:
    if isinstance(exc, (BadRequestError, UnprocessableEntityError, ValueError)):
        message = _error_message(exc).lower()
        return any(
            phrase in message
            for phrase in (
                "context length",
                "maximum context length",
                "max context length",
                "token limit",
                "too many tokens",
                "maximum number of tokens",
            )
        )
    return False


def is_retryable_exception(exc: Exception) -> bool:
    if is_context_length_error(exc):
        return False

    if isinstance(exc, (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError)):
        return True

    if isinstance(exc, (AuthenticationError, BadRequestError, PermissionDeniedError, UnprocessableEntityError)):
        return False

    if isinstance(exc, (requests_exceptions.Timeout, requests_exceptions.ConnectionError)):
        return True

    if isinstance(exc, requests_exceptions.HTTPError):
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        return status_code in {408, 425, 429, 500, 502, 503, 504}

    if isinstance(exc, (RedisTimeoutError, RedisConnectionError, BusyLoadingError, ClusterDownError, TryAgainError)):
        return True

    if isinstance(exc, redis.exceptions.RedisError):
        message = _error_message(exc).lower()
        return any(token in message for token in ("timeout", "temporarily", "try again", "loading", "busy", "connection"))

    return False


def run_with_retry(operation_name: str, operation: Callable[[], T], policy: RetryPolicy) -> T:
    attempts = max(1, policy.max_attempts)
    delay = max(0.0, policy.initial_backoff_seconds)

    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:
            message = _error_message(exc)
            if is_context_length_error(exc):
                raise ExternalServiceError(
                    f"{operation_name} failed because the request exceeds the model context window. "
                    f"Reduce the prompt or retrieved context and try again. Original error: {message}"
                ) from exc

            if not is_retryable_exception(exc) or attempt >= attempts:
                raise ExternalServiceError(
                    f"{operation_name} failed after {attempt} attempt(s): {message}"
                ) from exc

            time.sleep(min(policy.max_backoff_seconds, delay))
            delay = max(delay * 2.0, 0.0)

    raise ExternalServiceError(f"{operation_name} failed unexpectedly after retries")