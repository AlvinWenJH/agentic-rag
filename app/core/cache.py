"""
Redis cache connection and management.
"""

from typing import Optional, Any
import json
import structlog
import redis.asyncio as redis
from redis.exceptions import ConnectionError, TimeoutError

from app.core.config import get_settings
from app.core.exceptions import StorageError, ConfigurationError


logger = structlog.get_logger()

# Global Redis client instance
_client: Optional[redis.Redis] = None


async def init_cache() -> None:
    """Initialize Redis cache connection."""
    global _client

    settings = get_settings()

    try:
        # Create Redis client
        _client = redis.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            retry_on_timeout=settings.REDIS_RETRY_ON_TIMEOUT,
            decode_responses=True,
        )

        # Test connection
        await _client.ping()

        logger.info(
            "Cache connection established",
            url=settings.REDIS_URL.split("@")[-1],  # Hide credentials
        )

    except (ConnectionError, TimeoutError) as e:
        logger.error("Failed to connect to Redis", error=str(e))
        raise StorageError(f"Failed to connect to cache: {str(e)}")
    except Exception as e:
        logger.error("Unexpected cache initialization error", error=str(e))
        raise ConfigurationError(f"Cache configuration error: {str(e)}")


async def close_cache() -> None:
    """Close Redis cache connection."""
    global _client

    if _client:
        await _client.close()
        logger.info("Cache connection closed")


def get_cache_client() -> redis.Redis:
    """Get Redis client instance."""
    if _client is None:
        raise StorageError("Cache not initialized. Call init_cache() first.")
    return _client


def _get_cache_key(key: str) -> str:
    """Get prefixed cache key."""
    settings = get_settings()
    return f"{settings.CACHE_PREFIX}:{key}"


async def set_cache(key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """Set a value in cache."""
    try:
        client = get_cache_client()
        settings = get_settings()

        cache_key = _get_cache_key(key)
        cache_ttl = ttl or settings.CACHE_TTL

        # Serialize value to JSON
        serialized_value = json.dumps(value, default=str)

        # Set value with TTL
        result = await client.setex(cache_key, cache_ttl, serialized_value)

        logger.debug("Cache value set", key=cache_key, ttl=cache_ttl)

        return result

    except Exception as e:
        logger.error("Failed to set cache value", key=key, error=str(e))
        return False


async def get_cache(key: str) -> Optional[Any]:
    """Get a value from cache."""
    try:
        client = get_cache_client()
        cache_key = _get_cache_key(key)

        # Get value
        value = await client.get(cache_key)

        if value is None:
            logger.debug("Cache miss", key=cache_key)
            return None

        # Deserialize value from JSON
        deserialized_value = json.loads(value)

        logger.debug("Cache hit", key=cache_key)
        return deserialized_value

    except Exception as e:
        logger.error("Failed to get cache value", key=key, error=str(e))
        return None


async def delete_cache(key: str) -> bool:
    """Delete a value from cache."""
    try:
        client = get_cache_client()
        cache_key = _get_cache_key(key)

        # Delete value
        result = await client.delete(cache_key)

        logger.debug("Cache value deleted", key=cache_key, deleted=bool(result))

        return bool(result)

    except Exception as e:
        logger.error("Failed to delete cache value", key=key, error=str(e))
        return False


async def exists_cache(key: str) -> bool:
    """Check if a key exists in cache."""
    try:
        client = get_cache_client()
        cache_key = _get_cache_key(key)

        # Check existence
        result = await client.exists(cache_key)

        logger.debug("Cache existence check", key=cache_key, exists=bool(result))

        return bool(result)

    except Exception as e:
        logger.error("Failed to check cache existence", key=key, error=str(e))
        return False


async def increment_cache(key: str, amount: int = 1) -> Optional[int]:
    """Increment a numeric value in cache."""
    try:
        client = get_cache_client()
        cache_key = _get_cache_key(key)

        # Increment value
        result = await client.incrby(cache_key, amount)

        logger.debug(
            "Cache value incremented", key=cache_key, amount=amount, new_value=result
        )

        return result

    except Exception as e:
        logger.error("Failed to increment cache value", key=key, error=str(e))
        return None


async def set_cache_hash(
    key: str, field: str, value: Any, ttl: Optional[int] = None
) -> bool:
    """Set a field in a hash cache."""
    try:
        client = get_cache_client()
        settings = get_settings()

        cache_key = _get_cache_key(key)
        cache_ttl = ttl or settings.CACHE_TTL

        # Serialize value to JSON
        serialized_value = json.dumps(value, default=str)

        # Set hash field
        await client.hset(cache_key, field, serialized_value)

        # Set TTL if this is a new hash
        await client.expire(cache_key, cache_ttl)

        logger.debug("Cache hash field set", key=cache_key, field=field, ttl=cache_ttl)

        return True

    except Exception as e:
        logger.error(
            "Failed to set cache hash field", key=key, field=field, error=str(e)
        )
        return False


async def get_cache_hash(key: str, field: str) -> Optional[Any]:
    """Get a field from a hash cache."""
    try:
        client = get_cache_client()
        cache_key = _get_cache_key(key)

        # Get hash field
        value = await client.hget(cache_key, field)

        if value is None:
            logger.debug("Cache hash miss", key=cache_key, field=field)
            return None

        # Deserialize value from JSON
        deserialized_value = json.loads(value)

        logger.debug("Cache hash hit", key=cache_key, field=field)
        return deserialized_value

    except Exception as e:
        logger.error(
            "Failed to get cache hash field", key=key, field=field, error=str(e)
        )
        return None


async def delete_cache_pattern(pattern: str) -> int:
    """Delete all keys matching a pattern."""
    try:
        client = get_cache_client()
        cache_pattern = _get_cache_key(pattern)

        # Find matching keys
        keys = await client.keys(cache_pattern)

        if not keys:
            return 0

        # Delete keys
        result = await client.delete(*keys)

        logger.debug(
            "Cache pattern deleted", pattern=cache_pattern, deleted_count=result
        )

        return result

    except Exception as e:
        logger.error("Failed to delete cache pattern", pattern=pattern, error=str(e))
        return 0
