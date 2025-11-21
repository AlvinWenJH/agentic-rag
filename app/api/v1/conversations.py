"""
Conversation management API endpoints.
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional
import structlog

from app.services import conversation as conversation_service
from app.core.exceptions import NotFoundError

logger = structlog.get_logger()
router = APIRouter()


class CreateConversationRequest(BaseModel):
    """Request model for creating a conversation."""

    document_id: str
    user_id: Optional[str] = None
    title: Optional[str] = None


class AddMessageRequest(BaseModel):
    """Request model for adding a message to a conversation."""

    role: str
    content: str
    meta: Optional[dict] = None


@router.post("/")
async def create_conversation(request: CreateConversationRequest):
    """Create a new conversation."""
    try:
        conversation = await conversation_service.create_conversation(
            document_id=request.document_id,
            user_id=request.user_id,
            title=request.title,
        )

        return {
            "conversation_id": str(conversation["_id"]),
            "document_id": conversation["document_id"],
            "title": conversation["title"],
            "created_at": conversation["created_at"].isoformat(),
        }

    except Exception as e:
        logger.error("Failed to create conversation", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create conversation")


@router.get("/")
async def list_conversations(
    user_id: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
):
    """List conversations."""
    try:
        result = await conversation_service.list_conversations(
            user_id=user_id,
            limit=limit,
            skip=skip,
        )

        return result

    except Exception as e:
        logger.error("Failed to list conversations", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list conversations")


@router.get("/recent")
async def get_recent_conversations(
    user_id: Optional[str] = Query(None),
    limit: int = Query(4, ge=1, le=10),
):
    """Get recent conversations for the home dashboard."""
    try:
        result = await conversation_service.list_conversations(
            user_id=user_id,
            limit=limit,
            skip=0,
        )

        return result

    except Exception as e:
        logger.error("Failed to get recent conversations", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to get recent conversations"
        )


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get a conversation by ID."""
    try:
        conversation = await conversation_service.get_conversation(conversation_id)
        return conversation

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get conversation",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to get conversation")


@router.delete("/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    try:
        await conversation_service.delete_conversation(conversation_id)
        return {"success": True}

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to delete conversation",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to delete conversation")


@router.get("/{conversation_id}/messages")
async def get_conversation_messages(conversation_id: str):
    """Get messages in a conversation."""
    try:
        messages = await conversation_service.get_conversation_messages(conversation_id)
        return {"messages": messages}

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get conversation messages",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to get conversation messages"
        )


@router.post("/{conversation_id}/messages")
async def add_message_to_conversation(conversation_id: str, request: AddMessageRequest):
    """Add a message to a conversation."""
    try:
        conversation = await conversation_service.add_message_to_conversation(
            conversation_id=conversation_id,
            role=request.role,
            content=request.content,
            meta=request.meta,
        )

        return {
            "success": True,
            "conversation": conversation,
        }

    except NotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to add message to conversation",
            conversation_id=conversation_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to add message")
