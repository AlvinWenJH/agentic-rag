"""
MongoDB database connection and management.
"""

from typing import Optional
import structlog
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app.core.config import get_settings
from app.core.exceptions import DatabaseError, ConfigurationError


logger = structlog.get_logger()

# Global database client and database instances
_client: Optional[AsyncIOMotorClient] = None
_database: Optional[AsyncIOMotorDatabase] = None


async def init_database() -> None:
    """Initialize database connection."""
    global _client, _database

    settings = get_settings()

    try:
        # Create MongoDB client
        _client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            minPoolSize=settings.MONGODB_MIN_POOL_SIZE,
            maxPoolSize=settings.MONGODB_MAX_POOL_SIZE,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
        )

        # Get database
        _database = _client[settings.MONGODB_DATABASE]

        # Test connection
        await _client.admin.command("ping")

        logger.info(
            "Database connection established",
            database=settings.MONGODB_DATABASE,
            url=settings.MONGODB_URL.split("@")[-1],  # Hide credentials
        )

    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error("Failed to connect to MongoDB", error=str(e))
        raise DatabaseError(f"Failed to connect to database: {str(e)}")
    except Exception as e:
        logger.error("Unexpected database initialization error", error=str(e))
        raise ConfigurationError(f"Database configuration error: {str(e)}")


async def close_database() -> None:
    """Close database connection."""
    global _client

    if _client:
        _client.close()
        logger.info("Database connection closed")


def get_database() -> AsyncIOMotorDatabase:
    """Get database instance."""
    if _database is None:
        raise DatabaseError("Database not initialized. Call init_database() first.")
    return _database


def get_client() -> AsyncIOMotorClient:
    """Get database client instance."""
    if _client is None:
        raise DatabaseError(
            "Database client not initialized. Call init_database() first."
        )
    return _client


# Collection getters
def get_documents_collection():
    """Get documents collection."""
    return get_database().documents


def get_subtrees_collection():
    """Get subtrees collection."""
    return get_database().subtrees


def get_users_collection():
    """Get users collection."""
    return get_database().users


def get_queries_collection():
    """Get queries collection."""
    return get_database().queries


def get_tree_collection():
    """Get tree collection."""
    return get_database().tree
