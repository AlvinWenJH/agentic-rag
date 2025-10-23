"""
User management API endpoints.
Basic user operations for the Vectorless RAG system.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import structlog
from datetime import datetime, timedelta
from pydantic import BaseModel, EmailStr
from bson import ObjectId
from bson.errors import InvalidId

from app.core.database import get_users_collection
from app.core.exceptions import NotFoundError, ValidationError, ConflictError


logger = structlog.get_logger()
router = APIRouter()


# User models (simplified for this implementation)
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    password: str


class UserUpdate(BaseModel):
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    id: str
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    query_count: int = 0
    document_count: int = 0


class UserListResponse(BaseModel):
    users: List[UserResponse]
    total: int
    skip: int
    limit: int


class UserStats(BaseModel):
    total_users: int
    active_users: int
    inactive_users: int
    users_with_documents: int
    users_with_queries: int
    average_queries_per_user: float
    average_documents_per_user: float


@router.post("/", response_model=UserResponse)
async def create_user(user_data: UserCreate):
    """Create a new user."""
    try:
        logger.info(
            "Creating new user", username=user_data.username, email=user_data.email
        )

        users_collection = get_users_collection()

        # Check if user already exists
        existing_user = await users_collection.find_one(
            {"$or": [{"email": user_data.email}, {"username": user_data.username}]}
        )

        if existing_user:
            if existing_user["email"] == user_data.email:
                raise ConflictError("User with this email already exists")
            else:
                raise ConflictError("User with this username already exists")

        # Create user document
        now = datetime.utcnow()
        user_doc = {
            "email": user_data.email,
            "username": user_data.username,
            "full_name": user_data.full_name,
            "is_active": True,
            "password_hash": "hashed_" + user_data.password,  # Simplified hashing
            "created_at": now,
            "updated_at": now,
            "last_login": None,
            "query_count": 0,
            "document_count": 0,
        }

        # Insert user
        result = await users_collection.insert_one(user_doc)
        user_id = str(result.inserted_id)

        # Return user response
        user_doc["id"] = user_id
        del user_doc["_id"]
        del user_doc["password_hash"]

        logger.info(
            "User created successfully", user_id=user_id, username=user_data.username
        )

        return UserResponse(**user_doc)

    except ConflictError:
        raise HTTPException(status_code=409, detail="User already exists")
    except Exception as e:
        logger.error("Failed to create user", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create user")


@router.get("/stats", response_model=UserStats)
async def get_user_stats():
    """Get user statistics."""
    try:
        users_collection = get_users_collection()

        # Get total users
        total_users = await users_collection.count_documents({})

        # Get active/inactive users
        active_users = await users_collection.count_documents({"is_active": True})
        inactive_users = total_users - active_users

        # Get users with documents
        users_with_documents = await users_collection.count_documents(
            {"document_count": {"$gt": 0}}
        )

        # Get users with queries
        users_with_queries = await users_collection.count_documents(
            {"query_count": {"$gt": 0}}
        )

        # Get average queries per user
        avg_queries_pipeline = [
            {"$group": {"_id": None, "avg_queries": {"$avg": "$query_count"}}}
        ]
        avg_queries_result = await users_collection.aggregate(
            avg_queries_pipeline
        ).to_list(1)
        avg_queries = avg_queries_result[0]["avg_queries"] if avg_queries_result else 0

        # Get average documents per user
        avg_docs_pipeline = [
            {"$group": {"_id": None, "avg_documents": {"$avg": "$document_count"}}}
        ]
        avg_docs_result = await users_collection.aggregate(avg_docs_pipeline).to_list(1)
        avg_documents = avg_docs_result[0]["avg_documents"] if avg_docs_result else 0

        return UserStats(
            total_users=total_users,
            active_users=active_users,
            inactive_users=inactive_users,
            users_with_documents=users_with_documents,
            users_with_queries=users_with_queries,
            average_queries_per_user=avg_queries,
            average_documents_per_user=avg_documents,
        )

    except Exception as e:
        logger.error("Failed to get user stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get user statistics")


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """Get user by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        users_collection = get_users_collection()
        user = await users_collection.find_one({"_id": object_id})

        if not user:
            raise NotFoundError(f"User {user_id} not found")

        # Convert ObjectId to string and remove sensitive data
        user["id"] = str(user["_id"])
        del user["_id"]
        if "password_hash" in user:
            del user["password_hash"]

        return UserResponse(**user)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to get user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get user")


@router.get("/", response_model=UserListResponse)
async def list_users(
    is_active: Optional[bool] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List users with optional filtering."""
    try:
        users_collection = get_users_collection()

        # Build filter
        filter_dict = {}
        if is_active is not None:
            filter_dict["is_active"] = is_active

        # Get total count
        total = await users_collection.count_documents(filter_dict)

        # Get users
        cursor = (
            users_collection.find(filter_dict, {"password_hash": 0})
            .skip(skip)
            .limit(limit)
            .sort("created_timestamp", -1)
        )
        users = await cursor.to_list(length=limit)

        # Convert ObjectIds to strings
        for user in users:
            user["id"] = str(user["_id"])
            del user["_id"]

        return UserListResponse(
            users=[UserResponse(**user) for user in users],
            total=total,
            skip=skip,
            limit=limit,
        )

    except Exception as e:
        logger.error("Failed to list users", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list users")


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(user_id: str, update_data: UserUpdate):
    """Update user information."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        users_collection = get_users_collection()

        # Check if user exists
        existing_user = await users_collection.find_one({"_id": object_id})
        if not existing_user:
            raise NotFoundError(f"User {user_id} not found")

        # Prepare update data
        update_dict = update_data.dict(exclude_unset=True)
        if update_dict:
            update_dict["updated_at"] = datetime.utcnow()

            # Check for conflicts if email or username is being updated
            if "email" in update_dict or "username" in update_dict:
                conflict_filter = {"_id": {"$ne": object_id}}
                if "email" in update_dict:
                    conflict_filter["email"] = update_dict["email"]
                if "username" in update_dict:
                    conflict_filter["username"] = update_dict["username"]

                conflicting_user = await users_collection.find_one(conflict_filter)
                if conflicting_user:
                    raise ConflictError("Email or username already exists")

            # Update user
            await users_collection.update_one({"_id": object_id}, {"$set": update_dict})

        # Get updated user
        updated_user = await users_collection.find_one(
            {"_id": object_id}, {"password_hash": 0}
        )
        updated_user["id"] = str(updated_user["_id"])
        del updated_user["_id"]

        return UserResponse(**updated_user)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except ConflictError:
        raise HTTPException(status_code=409, detail="Email or username already exists")
    except Exception as e:
        logger.error("Failed to update user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update user")


@router.delete("/{user_id}")
async def delete_user(user_id: str):
    """Delete user."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        users_collection = get_users_collection()

        # Check if user exists
        user = await users_collection.find_one({"_id": object_id})
        if not user:
            raise NotFoundError(f"User {user_id} not found")

        # Delete user
        await users_collection.delete_one({"_id": object_id})

        logger.info("User deleted", user_id=user_id)

        return {"message": "User deleted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to delete user", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete user")


@router.post("/{user_id}/login")
async def user_login(user_id: str):
    """Record user login."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        users_collection = get_users_collection()

        # Check if user exists
        user = await users_collection.find_one({"_id": object_id})
        if not user:
            raise NotFoundError(f"User {user_id} not found")

        # Update last login
        await users_collection.update_one(
            {"_id": object_id},
            {
                "$set": {
                    "last_login": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        logger.info("User login recorded", user_id=user_id)

        return {
            "message": "Login recorded successfully",
            "timestamp": datetime.utcnow(),
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to record login", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record login")


@router.get("/{user_id}/activity")
async def get_user_activity(user_id: str, days: int = Query(30, ge=1, le=365)):
    """
    Get user activity summary.

    Args:
        user_id: User ID
        days: Number of days to analyze

    Returns:
        User activity data
    """
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(user_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid user ID format")

        from app.core.database import get_documents_collection, get_queries_collection

        users_collection = get_users_collection()

        # Check if user exists
        user = await users_collection.find_one({"_id": object_id})
        if not user:
            raise NotFoundError(f"User {user_id} not found")

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get user documents in date range
        documents_collection = get_documents_collection()
        recent_documents = await documents_collection.count_documents(
            {
                "user_id": user_id,
                "created_at": {"$gte": start_date, "$lte": end_date},
            }
        )

        # Get user queries in date range
        queries_collection = get_queries_collection()
        recent_queries = await queries_collection.count_documents(
            {
                "user_id": user_id,
                "created_at": {"$gte": start_date, "$lte": end_date},
            }
        )

        # Get total counts
        total_documents = await documents_collection.count_documents(
            {"user_id": user_id}
        )
        total_queries = await queries_collection.count_documents({"user_id": user_id})

        return {
            "user_id": user_id,
            "period": f"{days} days",
            "recent_activity": {
                "documents_uploaded": recent_documents,
                "queries_executed": recent_queries,
            },
            "total_activity": {
                "total_documents": total_documents,
                "total_queries": total_queries,
            },
            "last_login": user.get("last_login"),
            "account_created": user.get("created_at"),
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.error("Failed to get user activity", user_id=user_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get user activity")
