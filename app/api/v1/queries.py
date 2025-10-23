"""
Query processing API endpoints.
Handles intelligent query processing using Pydantic AI and tree-based search.
"""

from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Body
import structlog
from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId

from app.models.query import (
    QueryRequest,
    QueryResponse,
    QueryListResponse,
    QueryExecutionRequest,
    QueryExecutionResponse,
    QuerySuggestionsRequest,
    QuerySuggestionsResponse,
    QueryStats,
    QueryStatus,
    QueryType,
    QueryScope,
)
from app.services.pydantic_ai_service import pydantic_ai_processor
from app.core.database import get_queries_collection, get_documents_collection
from app.core.exceptions import NotFoundError, ValidationError, ProcessingError


logger = structlog.get_logger()
router = APIRouter()


@router.post("/execute", response_model=QueryExecutionResponse)
async def execute_query(request: QueryExecutionRequest):
    """
    Execute a query using the Pydantic AI processing pipeline.

    Args:
        request: Query execution request

    Returns:
        Query execution response with results
    """
    try:
        logger.info(
            "Query execution started",
            query=request.query_text[:100] + "..."
            if len(request.query_text) > 100
            else request.query_text,
            scope=request.scope,
            document_id=request.document_id,
        )

        # Validate document ID if provided
        if request.document_id:
            from bson import ObjectId
            documents_collection = get_documents_collection()
            doc = await documents_collection.find_one({"_id": ObjectId(request.document_id)})
            if not doc:
                raise NotFoundError(f"Document {request.document_id} not found")

        # Process query using Pydantic AI
        query_result = await pydantic_ai_processor.process_complete_query(
            query=request.query_text,
            document_ids=[request.document_id] if request.document_id else [],
            scope=request.scope,
            context={},
        )

        # Store query in database
        queries_collection = get_queries_collection()
        now = datetime.utcnow()
        query_record = {
            "query_text": request.query_text,
            "query_type": QueryType.search,  # Default type
            "status": QueryStatus.completed,
            "scope": request.scope,
            "document_ids": request.document_ids or [],
            "context": request.context or {},
            "result": query_result,
            "created_at": now,
            "updated_at": now,
            "processing_time": query_result.get("processing_time", 0),
            "confidence_score": query_result.get("confidence_score", 0.0),
        }

        result = await queries_collection.insert_one(query_record)
        query_id = str(result.inserted_id)

        logger.info(
            "Query execution completed",
            query_id=query_id,
            processing_time=query_result.get("processing_time", 0),
            confidence_score=query_result.get("confidence_score", 0.0),
        )

        return QueryExecutionResponse(
            query_id=query_id,
            query=request.query,
            status="completed",
            result=query_result,
            processing_time=query_result.get("processing_time", 0),
            confidence_score=query_result.get("confidence_score", 0.0),
            suggestions=query_result.get("suggestions", []),
        )

    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ProcessingError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error("Query execution failed", query=request.query_text, error=str(e))
        raise HTTPException(status_code=500, detail="Query execution failed")


@router.post("/", response_model=QueryResponse)
async def create_query(query_request: QueryRequest):
    """
    Create a new query record.

    Args:
        query_request: Query creation request

    Returns:
        Created query response
    """
    try:
        queries_collection = get_queries_collection()

        # Create query record
        now = datetime.utcnow()
        query_data = {
            "query_text": query_request.query_text,
            "query_type": query_request.query_type,
            "status": QueryStatus.pending,
            "scope": query_request.scope,
            "document_ids": query_request.document_ids or [],
            "context": query_request.context or {},
            "created_at": now,
            "updated_at": now,
            "processing_time": 0,
            "confidence_score": 0.0,
        }

        result = await queries_collection.insert_one(query_data)
        query_id = str(result.inserted_id)

        # Get created query
        created_query = await queries_collection.find_one({"_id": query_id})
        created_query["id"] = str(created_query["_id"])
        del created_query["_id"]

        logger.info(
            "Query created", query_id=query_id, query_type=query_request.query_type
        )

        return QueryResponse(**created_query)

    except Exception as e:
        logger.error("Failed to create query", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create query")


@router.get("/{query_id}", response_model=QueryResponse)
async def get_query(query_id: str):
    """Get query by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(query_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid query ID format")

        queries_collection = get_queries_collection()
        query = await queries_collection.find_one({"_id": object_id})

        if not query:
            raise NotFoundError(f"Query {query_id} not found")

        # Convert ObjectId to string
        query["id"] = str(query["_id"])
        del query["_id"]

        return QueryResponse(**query)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Query not found")
    except Exception as e:
        logger.error("Failed to get query", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve query")


@router.get("/", response_model=QueryListResponse)
async def list_queries(
    query_type: Optional[QueryType] = Query(None),
    status: Optional[QueryStatus] = Query(None),
    scope: Optional[QueryScope] = Query(None),
    document_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List queries with optional filtering."""
    try:
        queries_collection = get_queries_collection()

        # Build filter
        filter_dict = {}
        if query_type:
            filter_dict["query_type"] = query_type
        if status:
            filter_dict["status"] = status
        if scope:
            filter_dict["scope"] = scope
        if document_id:
            filter_dict["document_ids"] = {"$in": [document_id]}

        # Get total count
        total = await queries_collection.count_documents(filter_dict)

        # Get queries
        cursor = (
            queries_collection.find(filter_dict)
            .skip(skip)
            .limit(limit)
            .sort("created_at", -1)
        )
        queries = await cursor.to_list(length=limit)

        # Convert ObjectIds to strings
        for query in queries:
            query["id"] = str(query["_id"])
            del query["_id"]

        # Calculate pagination
        page = (skip // limit) + 1
        pages = (total + limit - 1) // limit

        return QueryListResponse(
            queries=[QueryResponse(**query) for query in queries],
            total=total,
            page=page,
            size=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error("Failed to list queries", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list queries")


@router.post("/suggestions", response_model=QuerySuggestionsResponse)
async def get_query_suggestions(request: QuerySuggestionsRequest):
    """
    Get query suggestions based on context.

    Args:
        request: Query suggestions request

    Returns:
        Query suggestions response
    """
    try:
        logger.info(
            "Generating query suggestions",
            context_type=request.context_type,
            document_ids=request.document_ids,
        )

        # Generate suggestions using Pydantic AI
        suggestions = await pydantic_ai_processor.generate_suggestions(
            context_type=request.context_type,
            document_ids=request.document_ids,
            current_query=request.current_query,
            user_context=request.user_context,
        )

        logger.info(
            "Query suggestions generated",
            suggestion_count=len(suggestions.get("suggestions", [])),
        )

        return QuerySuggestionsResponse(
            context_type=request.context_type,
            suggestions=suggestions.get("suggestions", []),
            categories=suggestions.get("categories", []),
            related_topics=suggestions.get("related_topics", []),
        )

    except Exception as e:
        logger.error("Failed to generate query suggestions", error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to generate query suggestions"
        )


@router.post("/{query_id}/feedback")
async def submit_query_feedback(query_id: str, feedback: Dict[str, Any] = Body(...)):
    """
    Submit feedback for a query result.

    Args:
        query_id: Query ID
        feedback: Feedback data

    Returns:
        Success message
    """
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(query_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid query ID format")

        queries_collection = get_queries_collection()

        # Check if query exists
        query = await queries_collection.find_one({"_id": object_id})
        if not query:
            raise NotFoundError(f"Query {query_id} not found")

        # Update query with feedback
        feedback_data = {
            "feedback": feedback,
            "feedback_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow(),
        }

        await queries_collection.update_one({"_id": object_id}, {"$set": feedback_data})

        logger.info("Query feedback submitted", query_id=query_id)

        return {"message": "Feedback submitted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Query not found")
    except Exception as e:
        logger.error("Failed to submit query feedback", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to submit feedback")


@router.delete("/{query_id}")
async def delete_query(query_id: str):
    """Delete query."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(query_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid query ID format")

        queries_collection = get_queries_collection()

        # Check if query exists
        query = await queries_collection.find_one({"_id": object_id})
        if not query:
            raise NotFoundError(f"Query {query_id} not found")

        # Delete query
        await queries_collection.delete_one({"_id": object_id})

        logger.info("Query deleted", query_id=query_id)

        return {"message": "Query deleted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Query not found")
    except Exception as e:
        logger.error("Failed to delete query", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete query")


@router.get("/{query_id}/similar")
async def get_similar_queries(query_id: str, limit: int = Query(10, ge=1, le=50)):
    """
    Get similar queries based on query text and context.

    Args:
        query_id: Query ID
        limit: Maximum number of similar queries to return

    Returns:
        List of similar queries
    """
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(query_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid query ID format")

        queries_collection = get_queries_collection()

        # Get source query
        source_query = await queries_collection.find_one({"_id": object_id})
        if not source_query:
            raise NotFoundError(f"Query {query_id} not found")

        # Simple similarity search based on text matching
        # In a production system, you'd use proper text similarity algorithms
        source_text = source_query["query_text"].lower()
        source_words = set(source_text.split())

        # Find queries with overlapping words
        all_queries = await queries_collection.find({"_id": {"$ne": object_id}}).to_list(
            length=None
        )

        similar_queries = []
        for query in all_queries:
            query_text = query["query_text"].lower()
            query_words = set(query_text.split())

            # Calculate simple word overlap similarity
            overlap = len(source_words.intersection(query_words))
            total_words = len(source_words.union(query_words))

            if total_words > 0:
                similarity = overlap / total_words
                if similarity > 0.2:  # Threshold for similarity
                    query["id"] = str(query["_id"])
                    del query["_id"]
                    query["similarity_score"] = similarity
                    similar_queries.append(query)

        # Sort by similarity and limit results
        similar_queries.sort(key=lambda x: x["similarity_score"], reverse=True)
        similar_queries = similar_queries[:limit]

        return {
            "query_id": query_id,
            "similar_queries": similar_queries,
            "total_found": len(similar_queries),
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Query not found")
    except Exception as e:
        logger.error("Failed to get similar queries", query_id=query_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get similar queries")


@router.get("/analytics/stats", response_model=QueryStats)
async def get_query_stats():
    """Get query statistics."""
    try:
        queries_collection = get_queries_collection()

        # Get counts by status
        status_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]

        status_counts = {}
        async for result in queries_collection.aggregate(status_pipeline):
            status_counts[result["_id"]] = result["count"]

        # Get counts by type
        type_pipeline = [{"$group": {"_id": "$query_type", "count": {"$sum": 1}}}]

        type_counts = {}
        async for result in queries_collection.aggregate(type_pipeline):
            type_counts[result["_id"]] = result["count"]

        # Get total counts
        total_queries = await queries_collection.count_documents({})

        # Get average processing time
        avg_time_pipeline = [
            {
                "$group": {
                    "_id": None,
                    "avg_processing_time": {"$avg": "$processing_time"},
                }
            }
        ]

        avg_time_result = await queries_collection.aggregate(avg_time_pipeline).to_list(
            1
        )
        avg_processing_time = (
            avg_time_result[0]["avg_processing_time"] if avg_time_result else 0
        )

        # Get average confidence score
        avg_confidence_pipeline = [
            {"$group": {"_id": None, "avg_confidence": {"$avg": "$confidence_score"}}}
        ]

        avg_confidence_result = await queries_collection.aggregate(
            avg_confidence_pipeline
        ).to_list(1)
        avg_confidence = (
            avg_confidence_result[0]["avg_confidence"] if avg_confidence_result else 0
        )

        # Get recent queries (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        recent_queries = await queries_collection.count_documents({
            "created_at": {"$gte": twenty_four_hours_ago}
        })

        # Get popular query types (sorted by frequency)
        popular_types_pipeline = [
            {"$group": {"_id": "$query_type", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        popular_query_types = []
        async for result in queries_collection.aggregate(popular_types_pipeline):
            if result["_id"]:  # Only include non-null query types
                popular_query_types.append(result["_id"])

        # Get user satisfaction rate (percentage of helpful queries)
        total_with_feedback = await queries_collection.count_documents({
            "is_helpful": {"$exists": True, "$ne": None}
        })
        
        user_satisfaction_rate = None
        if total_with_feedback > 0:
            helpful_queries = await queries_collection.count_documents({
                "is_helpful": True
            })
            user_satisfaction_rate = (helpful_queries / total_with_feedback) * 100

        return QueryStats(
            total_queries=total_queries,
            queries_by_status=status_counts,
            queries_by_type=type_counts,
            average_processing_time=avg_processing_time,
            average_confidence_score=avg_confidence,
            recent_queries=recent_queries,
            popular_query_types=popular_query_types,
            user_satisfaction_rate=user_satisfaction_rate,
        )

    except Exception as e:
        logger.error("Failed to get query stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get query statistics")


@router.get("/analytics/trends")
async def get_query_trends(
    days: int = Query(30, ge=1, le=365),
    granularity: str = Query("day", regex="^(hour|day|week|month)$"),
):
    """
    Get query trends over time.

    Args:
        days: Number of days to analyze
        granularity: Time granularity for grouping

    Returns:
        Query trends data
    """
    try:
        queries_collection = get_queries_collection()

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Define grouping format based on granularity
        date_formats = {
            "hour": "%Y-%m-%d %H:00:00",
            "day": "%Y-%m-%d",
            "week": "%Y-W%U",
            "month": "%Y-%m",
        }

        date_format = date_formats[granularity]

        # Aggregation pipeline
        pipeline = [
            {"$match": {"created_timestamp": {"$gte": start_date, "$lte": end_date}}},
            {
                "$group": {
                    "_id": {
                        "date": {
                            "$dateToString": {
                                "format": date_format,
                                "date": "$created_timestamp",
                            }
                        },
                        "status": "$status",
                    },
                    "count": {"$sum": 1},
                }
            },
            {
                "$group": {
                    "_id": "$_id.date",
                    "data": {"$push": {"status": "$_id.status", "count": "$count"}},
                    "total": {"$sum": "$count"},
                }
            },
            {"$sort": {"_id": 1}},
        ]

        trends = []
        async for result in queries_collection.aggregate(pipeline):
            trends.append(
                {
                    "date": result["_id"],
                    "total_queries": result["total"],
                    "by_status": {
                        item["status"]: item["count"] for item in result["data"]
                    },
                }
            )

        return {"period": f"{days} days", "granularity": granularity, "trends": trends}

    except Exception as e:
        logger.error("Failed to get query trends", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get query trends")


@router.get("/analytics/popular-terms")
async def get_popular_query_terms(
    limit: int = Query(20, ge=1, le=100), days: int = Query(30, ge=1, le=365)
):
    """
    Get most popular query terms.

    Args:
        limit: Maximum number of terms to return
        days: Number of days to analyze

    Returns:
        Popular query terms
    """
    try:
        queries_collection = get_queries_collection()

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Get recent queries
        recent_queries = await queries_collection.find(
            {"created_timestamp": {"$gte": start_date, "$lte": end_date}},
            {"query_text": 1},
        ).to_list(length=None)

        # Count word frequencies
        word_counts = {}
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "what",
            "where",
            "when",
            "why",
            "how",
            "who",
            "which",
        }

        for query in recent_queries:
            words = query["query_text"].lower().split()
            for word in words:
                # Clean word (remove punctuation)
                clean_word = "".join(c for c in word if c.isalnum())
                if len(clean_word) > 2 and clean_word not in stop_words:
                    word_counts[clean_word] = word_counts.get(clean_word, 0) + 1

        # Sort by frequency and limit
        popular_terms = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[
            :limit
        ]

        return {
            "period": f"{days} days",
            "popular_terms": [
                {"term": term, "count": count} for term, count in popular_terms
            ],
            "total_queries_analyzed": len(recent_queries),
        }

    except Exception as e:
        logger.error("Failed to get popular query terms", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get popular query terms")
