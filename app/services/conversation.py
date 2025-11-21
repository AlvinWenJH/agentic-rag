"""
Conversation management service module.
Handles CRUD operations for conversation history.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, UTC
import structlog
from bson import ObjectId

from app.core.database import get_conversations_collection
from app.core.exceptions import NotFoundError

logger = structlog.get_logger()


async def create_conversation(
    document_id: str, user_id: Optional[str] = None, title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new conversation.

    Args:
        document_id: ID of the document for this conversation
        user_id: Optional user ID
        title: Optional initial title (defaults to "New Conversation")

    Returns:
        Created conversation document
    """
    conversations_collection = get_conversations_collection()

    conversation = {
        "document_id": document_id,
        "user_id": user_id,
        "title": title or "New Conversation",
        "created_at": datetime.now(UTC),
        "updated_at": datetime.now(UTC),
        "is_deleted": False,
        "messages": [],
    }

    result = await conversations_collection.insert_one(conversation)

    logger.info(
        "Created new conversation",
        conversation_id=str(result.inserted_id),
        document_id=document_id,
        user_id=user_id,
    )

    conversation["_id"] = result.inserted_id
    return conversation


async def list_conversations(
    user_id: Optional[str] = None, limit: int = 20, skip: int = 0
) -> Dict[str, Any]:
    """
    List conversations for a user.

    Args:
        user_id: Optional user ID to filter conversations
        limit: Maximum number of conversations to return
        skip: Number of conversations to skip for pagination

    Returns:
        Dictionary with conversations list and total count
    """
    conversations_collection = get_conversations_collection()

    # Build query
    query: Dict[str, Any] = {"is_deleted": {"$ne": True}}
    if user_id:
        query["user_id"] = user_id

    # Get total count
    total = await conversations_collection.count_documents(query)

    # Get conversations
    cursor = (
        conversations_collection.find(query)
        .sort("updated_at", -1)
        .skip(skip)
        .limit(limit)
    )

    conversations = []
    async for doc in cursor:
        conversations.append(
            {
                "id": str(doc["_id"]),
                "document_id": doc.get("document_id"),
                "user_id": doc.get("user_id"),
                "title": doc.get("title", "New Conversation"),
                "created_at": doc.get("created_at").isoformat()
                if doc.get("created_at")
                else None,
                "updated_at": doc.get("updated_at").isoformat()
                if doc.get("updated_at")
                else None,
                "message_count": len(doc.get("messages", [])),
            }
        )

    logger.info(
        "Listed conversations",
        user_id=user_id,
        total=total,
        returned=len(conversations),
    )

    return {"conversations": conversations, "total": total}


async def get_conversation(conversation_id: str) -> Dict[str, Any]:
    """
    Get a conversation by ID.

    Args:
        conversation_id: ID of the conversation

    Returns:
        Conversation document

    Raises:
        NotFoundError: If conversation not found or deleted
    """
    conversations_collection = get_conversations_collection()

    try:
        doc = await conversations_collection.find_one(
            {"_id": ObjectId(conversation_id), "is_deleted": {"$ne": True}}
        )
    except Exception as e:
        logger.error(
            "Invalid conversation ID format",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    if not doc:
        logger.warning("Conversation not found", conversation_id=conversation_id)
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    return {
        "id": str(doc["_id"]),
        "document_id": doc.get("document_id"),
        "user_id": doc.get("user_id"),
        "title": doc.get("title", "New Conversation"),
        "created_at": doc.get("created_at").isoformat()
        if doc.get("created_at")
        else None,
        "updated_at": doc.get("updated_at").isoformat()
        if doc.get("updated_at")
        else None,
        "messages": doc.get("messages", []),
    }


async def delete_conversation(conversation_id: str) -> bool:
    """
    Soft delete a conversation.

    Args:
        conversation_id: ID of the conversation to delete

    Returns:
        True if successful

    Raises:
        NotFoundError: If conversation not found
    """
    conversations_collection = get_conversations_collection()

    try:
        result = await conversations_collection.update_one(
            {"_id": ObjectId(conversation_id), "is_deleted": {"$ne": True}},
            {"$set": {"is_deleted": True, "updated_at": datetime.now(UTC)}},
        )
    except Exception as e:
        logger.error(
            "Invalid conversation ID format",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    if result.matched_count == 0:
        logger.warning(
            "Conversation not found for deletion", conversation_id=conversation_id
        )
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    logger.info("Deleted conversation", conversation_id=conversation_id)
    return True


async def add_message_to_conversation(
    conversation_id: str,
    role: str,
    content: str,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add a message to a conversation.

    Args:
        conversation_id: ID of the conversation
        role: Message role (user/assistant/system)
        content: Message content
        meta: Optional metadata (usage, references)

    Returns:
        Updated conversation document

    Raises:
        NotFoundError: If conversation not found
    """
    conversations_collection = get_conversations_collection()

    message = {
        "role": role,
        "content": content,
        "meta": meta or {},
        "timestamp": datetime.now(UTC),
    }

    try:
        # Add message and update timestamps
        result = await conversations_collection.find_one_and_update(
            {"_id": ObjectId(conversation_id), "is_deleted": {"$ne": True}},
            {
                "$push": {"messages": message},
                "$set": {"updated_at": datetime.now(UTC)},
            },
            return_document=True,
        )
    except Exception as e:
        logger.error(
            "Invalid conversation ID format",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    if not result:
        logger.warning(
            "Conversation not found for message add", conversation_id=conversation_id
        )
        raise NotFoundError(f"Conversation not found: {conversation_id}")

    # Auto-update title based on first user message
    if role == "user" and len(result.get("messages", [])) == 1:
        title = content[:50] + ("..." if len(content) > 50 else "")
        await conversations_collection.update_one(
            {"_id": ObjectId(conversation_id)},
            {"$set": {"title": title}},
        )
        result["title"] = title

    logger.info(
        "Added message to conversation",
        conversation_id=conversation_id,
        role=role,
    )

    return {
        "id": str(result["_id"]),
        "document_id": result.get("document_id"),
        "user_id": result.get("user_id"),
        "title": result.get("title", "New Conversation"),
        "created_at": result.get("created_at").isoformat()
        if result.get("created_at")
        else None,
        "updated_at": result.get("updated_at").isoformat()
        if result.get("updated_at")
        else None,
        "messages": result.get("messages", []),
    }


async def get_conversation_messages(conversation_id: str) -> List[Dict[str, Any]]:
    """
    Get all messages in a conversation.

    Args:
        conversation_id: ID of the conversation

    Returns:
        List of messages

    Raises:
        NotFoundError: If conversation not found
    """
    conversation = await get_conversation(conversation_id)
    return conversation.get("messages", [])
