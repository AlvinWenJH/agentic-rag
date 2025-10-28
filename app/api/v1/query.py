"""
Document management API endpoints.
Handles document upload, processing, and retrieval.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import structlog
import json
from typing import Optional

from app.core.config import get_settings
from app.services.query import QueryService


class QueryRequest(BaseModel):
    """Request model for document query."""

    query: str
    user_id: Optional[str] = None


logger = structlog.get_logger()
router = APIRouter()
settings = get_settings()


@router.post("/document/{document_id}")
async def query(document_id: str, request: QueryRequest):
    """Query a document with streaming response."""
    try:
        query_service = QueryService()

        logger.info(
            "Starting document query",
            document_id=document_id,
            query=request.query,
            user_id=request.user_id,
        )

        async def generate_response():
            """Generate streaming response from query service."""
            try:
                async for event in query_service.query_doc(
                    document_id=document_id,
                    query=request.query,
                    user_id=request.user_id,
                ):
                    # Convert event to JSON and yield as server-sent event
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                # Send error event if something goes wrong during streaming
                error_event = {
                    "type": "error",
                    "error": str(e),
                    "document_id": document_id,
                    "user_id": request.user_id,
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream",
            },
        )

    except Exception as e:
        logger.error(
            "Failed to start document query",
            document_id=document_id,
            query=request.query,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to start document query")
