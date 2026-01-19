"""
Helper utilities for the Agentic Observability Platform.
Common functions used across the codebase.
"""

import asyncio
import hashlib
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, AsyncIterator, Callable, Iterator, List, Optional, TypeVar

T = TypeVar("T")


def generate_id(prefix: str = "") -> str:
    """
    Generate a unique ID with optional prefix.
    
    Args:
        prefix: Optional prefix for the ID
        
    Returns:
        Unique identifier string
    """
    uid = str(uuid.uuid4())
    if prefix:
        return f"{prefix}_{uid}"
    return uid


def now_utc() -> datetime:
    """
    Get current UTC datetime.
    
    Returns:
        Current datetime in UTC
    """
    return datetime.now(timezone.utc)


def timestamp_to_datetime(ts: float) -> datetime:
    """
    Convert Unix timestamp to datetime.
    
    Args:
        ts: Unix timestamp
        
    Returns:
        Datetime object in UTC
    """
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def datetime_to_timestamp(dt: datetime) -> float:
    """
    Convert datetime to Unix timestamp.
    
    Args:
        dt: Datetime object
        
    Returns:
        Unix timestamp
    """
    return dt.timestamp()


def chunks(lst: List[T], n: int) -> Iterator[List[T]]:
    """
    Yield successive n-sized chunks from a list.
    
    Args:
        lst: List to chunk
        n: Chunk size
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


async def async_chunks(lst: List[T], n: int) -> AsyncIterator[List[T]]:
    """
    Async version of chunks.
    
    Args:
        lst: List to chunk
        n: Chunk size
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def exponential_backoff(
    attempt: int,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True
) -> float:
    """
    Calculate exponential backoff delay.
    
    Args:
        attempt: Current attempt number (0-indexed)
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        jitter: Whether to add random jitter
        
    Returns:
        Delay in seconds
    """
    import random
    
    delay = min(base_delay * (2 ** attempt), max_delay)
    if jitter:
        delay = delay * (0.5 + random.random())
    return delay


def async_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    Decorator for async retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        base_delay: Base delay between attempts
        max_delay: Maximum delay cap
        exceptions: Tuple of exceptions to catch
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        delay = exponential_backoff(attempt, base_delay, max_delay)
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def hash_metric_signature(name: str, labels: dict) -> str:
    """
    Create a hash signature for a metric.
    
    Args:
        name: Metric name
        labels: Metric labels
        
    Returns:
        Hash string
    """
    label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
    signature = f"{name}|{label_str}"
    return hashlib.sha256(signature.encode()).hexdigest()[:16]


def normalize_metric_name(name: str) -> str:
    """
    Normalize metric name to standard format.
    
    Args:
        name: Raw metric name
        
    Returns:
        Normalized metric name
    """
    # Replace common separators with underscores
    normalized = name.replace("-", "_").replace(".", "_").replace("/", "_")
    # Remove consecutive underscores
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    # Strip leading/trailing underscores
    return normalized.strip("_").lower()


def calculate_percentile(values: List[float], percentile: float) -> float:
    """
    Calculate percentile of a list of values.
    
    Args:
        values: List of numeric values
        percentile: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    index = (percentile / 100) * (len(sorted_values) - 1)
    lower = int(index)
    upper = lower + 1
    
    if upper >= len(sorted_values):
        return sorted_values[-1]
    
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp a value between min and max.
    
    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(value, max_val))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default on division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if denominator is zero
        
    Returns:
        Division result or default
    """
    if denominator == 0:
        return default
    return numerator / denominator
