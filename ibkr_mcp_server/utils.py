"""Utility functions for IBKR MCP Server."""

import asyncio
import functools
import logging
import time
from typing import Any, Callable, TypeVar, Union
from decimal import Decimal

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


def rate_limit(calls_per_second: float = 2.0):
    """
    Rate limiting decorator for IBKR API calls.
    
    Args:
        calls_per_second: Maximum calls allowed per second
    """
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                await asyncio.sleep(left_to_wait)
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                last_called[0] = time.time()
        
        return wrapper
    return decorator


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Retry decorator for failed operations.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries
        backoff: Backoff multiplier for delay
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"Attempt {attempt + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {wait_time:.1f}s..."
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed for {func.__name__}")

            raise last_exception

        return wrapper
    return decorator


def retry_on_transient(max_attempts: int = 2, delay: float = 5.0, backoff: float = 1.5):
    """
    Selective retry decorator for order placement (audit M2).

    Only retries on transient failures: network errors, timeouts, IBKR
    connection drops. Does NOT retry on validation errors or business-logic
    rejections (insufficient buying power, halted symbol, etc.) — those
    will keep failing the same way and just delay reporting.

    Args:
        max_attempts: total attempts including initial; default 2
        delay: initial backoff seconds
        backoff: multiplier
    """
    # Transient = exceptions whose class name suggests connectivity / timing.
    # We do not retry on ValidationError, TradingError (live-port refusal),
    # or APIError (which is the catch-all for IB-side rejections).
    TRANSIENT_TYPES = (asyncio.TimeoutError, ConnectionError)
    TRANSIENT_NAME_HINTS = ("timeout", "disconnect", "reset", "broken pipe",
                             "connection", "temporarily")

    def _is_transient(exc: Exception) -> bool:
        if isinstance(exc, TRANSIENT_TYPES):
            return True
        msg = str(exc).lower()
        return any(h in msg for h in TRANSIENT_NAME_HINTS)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if not _is_transient(e):
                        # Non-transient — surface immediately; retrying would
                        # just delay the same rejection.
                        raise
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(
                            f"transient failure on {func.__name__} "
                            f"(attempt {attempt + 1}/{max_attempts}): {e}. "
                            f"retrying in {wait_time:.1f}s"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"all {max_attempts} attempts exhausted on {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator


def format_currency(value: Union[float, Decimal, str], currency: str = "USD") -> str:
    """Format currency values for display."""
    try:
        num_value = float(value) if value else 0.0
        if currency == "USD":
            return f"${num_value:,.2f}"
        else:
            return f"{num_value:,.2f} {currency}"
    except (ValueError, TypeError):
        return f"N/A {currency}"


def format_percentage(value: Union[float, Decimal, str]) -> str:
    """Format percentage values for display."""
    try:
        num_value = float(value) if value else 0.0
        return f"{num_value:.2f}%"
    except (ValueError, TypeError):
        return "N/A%"


def validate_symbol(symbol: str) -> str:
    """Validate and clean stock symbol."""
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol must be a non-empty string")
    
    cleaned = symbol.strip().upper()
    if not cleaned.isalnum():
        raise ValueError("Symbol must contain only alphanumeric characters")
    
    if len(cleaned) > 12:  # IBKR symbol length limit
        raise ValueError("Symbol too long (max 12 characters)")
    
    return cleaned


def validate_symbols(symbols_str: str) -> list[str]:
    """Validate and clean a comma-separated list of symbols."""
    if not symbols_str:
        raise ValueError("Symbols string cannot be empty")
    
    symbols = []
    for symbol in symbols_str.split(','):
        cleaned = validate_symbol(symbol)
        if cleaned not in symbols:  # Avoid duplicates
            symbols.append(cleaned)
    
    if len(symbols) > 50:  # Reasonable limit to avoid API overload
        raise ValueError("Too many symbols (max 50)")
    
    return symbols


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        if value is None or value == '':
            return default
        return int(float(value))  # Handle string floats like "100.0"
    except (ValueError, TypeError):
        return default


class IBKRError(Exception):
    """Base exception for IBKR-related errors."""
    pass


class ConnectionError(IBKRError):
    """IBKR connection-related errors."""
    pass


class APIError(IBKRError):
    """IBKR API-related errors."""
    pass


class ValidationError(IBKRError):
    """Input validation errors."""
    pass


class TradingError(IBKRError):
    """Trading-related errors."""
    pass
