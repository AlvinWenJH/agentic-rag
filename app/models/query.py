"""
Query data models and schemas for Pydantic AI integration.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId

from app.models.document import PyObjectId
from app.models.tree import NodeType


class QueryType(str, Enum):
    """Query type classification."""

    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SUMMARY = "summary"


class QueryStatus(str, Enum):
    """Query processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class QueryScope(str, Enum):
    """Query scope within document."""

    DOCUMENT = "document"
    CHAPTER = "chapter"
    SECTION = "section"
    SPECIFIC_PAGES = "specific_pages"


class QueryContext(BaseModel):
    """Query context information."""

    document_id: str = Field(..., description="Target document ID")
    tree_id: Optional[str] = Field(None, description="Target tree ID")
    scope: QueryScope = Field(default=QueryScope.DOCUMENT, description="Query scope")
    target_nodes: Optional[List[str]] = Field(
        None, description="Specific node IDs to search"
    )
    page_range: Optional[List[int]] = Field(None, description="Specific page numbers")
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Additional filters"
    )


class QueryRequest(BaseModel):
    """Query request model for Pydantic AI."""

    query_text: str = Field(..., description="Natural language query")
    context: QueryContext = Field(..., description="Query context")
    query_type: Optional[QueryType] = Field(None, description="Query type hint")
    max_results: int = Field(default=5, description="Maximum number of results")
    include_sources: bool = Field(default=True, description="Include source references")
    include_confidence: bool = Field(
        default=True, description="Include confidence scores"
    )
    temperature: float = Field(
        default=0.1, description="AI temperature for response generation"
    )


class QueryEvidence(BaseModel):
    """Evidence supporting a query result."""

    node_id: str = Field(..., description="Source node ID")
    node_title: str = Field(..., description="Source node title")
    node_type: NodeType = Field(..., description="Source node type")
    page_numbers: List[int] = Field(..., description="Referenced page numbers")
    content_excerpt: str = Field(..., description="Relevant content excerpt")
    confidence_score: float = Field(
        ..., description="Confidence in this evidence (0-1)"
    )
    relevance_score: float = Field(..., description="Relevance to query (0-1)")


class QueryResult(BaseModel):
    """Individual query result."""

    answer: str = Field(..., description="Answer text")
    confidence_score: float = Field(..., description="Overall confidence (0-1)")
    evidence: List[QueryEvidence] = Field(..., description="Supporting evidence")
    reasoning: Optional[str] = Field(None, description="AI reasoning process")
    node_path: List[str] = Field(..., description="Path through topic tree")


class QueryBase(BaseModel):
    """Base query model."""

    query_text: str = Field(..., description="Natural language query")
    context: QueryContext = Field(..., description="Query context")
    query_type: Optional[QueryType] = Field(None, description="Detected query type")


class QueryCreate(QueryBase):
    """Query creation model."""

    user_id: str = Field(..., description="User ID who made the query")


class QueryUpdate(BaseModel):
    """Query update model."""

    feedback_rating: Optional[int] = Field(
        None, description="User feedback rating (1-5)"
    )
    feedback_comment: Optional[str] = Field(None, description="User feedback comment")
    is_helpful: Optional[bool] = Field(None, description="Whether result was helpful")


class Query(QueryBase):
    """Complete query model."""

    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who made the query")
    status: QueryStatus = Field(
        default=QueryStatus.PENDING, description="Query processing status"
    )

    # Results
    results: List[QueryResult] = Field(
        default_factory=list, description="Query results"
    )
    primary_result: Optional[QueryResult] = Field(
        None, description="Primary/best result"
    )

    # Processing metadata
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    tokens_used: Optional[int] = Field(None, description="AI tokens consumed")
    model_used: str = Field(default="gemini-2.5-flash", description="AI model used")

    # User feedback
    feedback_rating: Optional[int] = Field(
        None, description="User feedback rating (1-5)"
    )
    feedback_comment: Optional[str] = Field(None, description="User feedback comment")
    is_helpful: Optional[bool] = Field(None, description="Whether result was helpful")

    # Error handling
    error_message: Optional[str] = Field(
        None, description="Error message if processing failed"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update timestamp"
    )
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class QueryResponse(BaseModel):
    """Query response model."""

    id: str = Field(..., description="Query ID")
    query_text: str = Field(..., description="Natural language query")
    status: QueryStatus = Field(..., description="Query processing status")
    query_type: Optional[QueryType] = Field(None, description="Detected query type")

    # Results
    results: List[QueryResult] = Field(..., description="Query results")
    primary_result: Optional[QueryResult] = Field(
        None, description="Primary/best result"
    )

    # Processing metadata
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    model_used: str = Field(..., description="AI model used")

    # Context
    document_id: str = Field(..., description="Target document ID")
    tree_id: Optional[str] = Field(None, description="Target tree ID")

    # User feedback
    feedback_rating: Optional[int] = Field(
        None, description="User feedback rating (1-5)"
    )
    is_helpful: Optional[bool] = Field(None, description="Whether result was helpful")

    # Error handling
    error_message: Optional[str] = Field(
        None, description="Error message if processing failed"
    )

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class QueryListResponse(BaseModel):
    """Query list response model."""

    queries: List[QueryResponse] = Field(..., description="List of queries")
    total: int = Field(..., description="Total number of queries")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class QueryExecutionRequest(BaseModel):
    """Query execution request for immediate processing."""

    query_text: str = Field(..., description="Natural language query")
    document_id: str = Field(..., description="Target document ID")
    tree_id: Optional[str] = Field(None, description="Target tree ID")
    scope: QueryScope = Field(default=QueryScope.DOCUMENT, description="Query scope")
    target_nodes: Optional[List[str]] = Field(
        None, description="Specific node IDs to search"
    )
    page_range: Optional[List[int]] = Field(None, description="Specific page numbers")
    max_results: int = Field(default=5, description="Maximum number of results")
    include_sources: bool = Field(default=True, description="Include source references")
    temperature: float = Field(
        default=0.1, description="AI temperature for response generation"
    )


class QueryExecutionResponse(BaseModel):
    """Query execution response for immediate processing."""

    query_text: str = Field(..., description="Natural language query")
    results: List[QueryResult] = Field(..., description="Query results")
    primary_result: Optional[QueryResult] = Field(
        None, description="Primary/best result"
    )
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="AI model used")
    tokens_used: Optional[int] = Field(None, description="AI tokens consumed")


class QuerySuggestion(BaseModel):
    """Query suggestion model."""

    suggestion_text: str = Field(..., description="Suggested query text")
    query_type: QueryType = Field(..., description="Suggested query type")
    confidence_score: float = Field(..., description="Confidence in suggestion (0-1)")
    target_nodes: List[str] = Field(..., description="Relevant node IDs")
    description: str = Field(
        ..., description="Description of what this query would find"
    )


class QuerySuggestionsRequest(BaseModel):
    """Query suggestions request model."""

    document_id: str = Field(..., description="Target document ID")
    tree_id: Optional[str] = Field(None, description="Target tree ID")
    context: Optional[str] = Field(None, description="Context for suggestions")
    max_suggestions: int = Field(default=5, description="Maximum number of suggestions")
    query_types: Optional[List[QueryType]] = Field(
        None, description="Filter by query types"
    )


class QuerySuggestionsResponse(BaseModel):
    """Query suggestions response model."""

    document_id: str = Field(..., description="Target document ID")
    suggestions: List[QuerySuggestion] = Field(..., description="Query suggestions")
    generation_time: float = Field(..., description="Generation time in seconds")


class QueryStats(BaseModel):
    """Query statistics model."""

    total_queries: int = Field(..., description="Total number of queries")
    queries_by_status: Dict[QueryStatus, int] = Field(
        ..., description="Queries grouped by status"
    )
    queries_by_type: Dict[QueryType, int] = Field(
        ..., description="Queries grouped by type"
    )
    average_processing_time: Optional[float] = Field(
        None, description="Average processing time in seconds"
    )
    average_confidence_score: Optional[float] = Field(
        None, description="Average confidence score"
    )
    user_satisfaction_rate: Optional[float] = Field(
        None, description="Percentage of helpful queries"
    )
    recent_queries: int = Field(
        ..., description="Number of recent queries (last 24 hours)"
    )
    popular_query_types: List[QueryType] = Field(
        ..., description="Most popular query types"
    )


class QueryAnalytics(BaseModel):
    """Query analytics model."""

    document_id: str = Field(..., description="Document ID")
    total_queries: int = Field(..., description="Total queries for this document")
    most_queried_nodes: List[Dict[str, Any]] = Field(
        ..., description="Most frequently queried nodes"
    )
    query_patterns: List[str] = Field(..., description="Common query patterns")
    user_engagement: Dict[str, Any] = Field(..., description="User engagement metrics")
    performance_metrics: Dict[str, float] = Field(
        ..., description="Performance metrics"
    )
