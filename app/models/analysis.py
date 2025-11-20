"""
Analysis data models and schemas.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AnalysisStatus(str, Enum):
    """Analysis processing status."""

    DRAFT = "draft"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisItem(BaseModel):
    """Individual analysis item/question."""

    question: str = Field(..., description="Analysis question to be answered")
    context: Optional[str] = Field(
        None, description="Additional context for the question"
    )
    order: int = Field(..., description="Order of the item in the analysis")


class AnalysisBase(BaseModel):
    """Base analysis model."""

    title: str = Field(..., description="Analysis title")
    description: str = Field(..., description="Analysis description")
    items: List[AnalysisItem] = Field(
        ..., description="List of analysis items/questions"
    )
    tags: List[str] = Field(default_factory=list, description="Analysis tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class AnalysisCreate(AnalysisBase):
    """Analysis creation model."""

    user_id: Optional[str] = Field(None, description="User ID who created the analysis")


class AnalysisUpdate(BaseModel):
    """Analysis update model."""

    title: Optional[str] = Field(None, description="Analysis title")
    description: Optional[str] = Field(None, description="Analysis description")
    items: Optional[List[AnalysisItem]] = Field(
        None, description="List of analysis items"
    )
    tags: Optional[List[str]] = Field(None, description="Analysis tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    status: Optional[AnalysisStatus] = Field(None, description="Analysis status")


class AnalysisResponse(BaseModel):
    """Analysis response model."""

    id: str = Field(..., description="Analysis ID")
    title: str = Field(..., description="Analysis title")
    description: str = Field(..., description="Analysis description")
    items: List[AnalysisItem] = Field(
        ..., description="List of analysis items/questions"
    )
    status: AnalysisStatus = Field(..., description="Analysis status")
    tags: List[str] = Field(..., description="Analysis tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    user_id: Optional[str] = Field(None, description="User ID who created the analysis")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")


class AnalysisListResponse(BaseModel):
    """Analysis list response model."""

    analyses: List[AnalysisResponse] = Field(..., description="List of analyses")
    total: int = Field(..., description="Total number of analyses")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class DraftAnalysisRequest(BaseModel):
    """Request model for draft analysis from free text."""

    text: str = Field(..., description="Free text describing the analysis needs")
    user_id: Optional[str] = Field(None, description="User ID")


class DraftAnalysisResponse(BaseModel):
    """Response model for draft analysis with structured output."""

    title: str = Field(..., description="Generated analysis title")
    description: str = Field(..., description="Generated analysis description")
    items: List[AnalysisItem] = Field(..., description="Generated analysis items")


class AnalysisResultItem(BaseModel):
    """Single analysis result item."""

    question: str = Field(..., description="The analysis question")
    answer: str = Field(..., description="The answer to the question")
    context: Optional[str] = Field(None, description="Context used for the answer")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")
    sources: List[str] = Field(default_factory=list, description="Source references")


class AnalysisResultStatus(str, Enum):
    """Analysis result processing status."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentAnalysisRequest(BaseModel):
    """Request model for document analysis."""

    document_id: str = Field(..., description="Document ID to analyze")
    analysis_id: str = Field(..., description="Analysis ID with items to process")
    user_id: Optional[str] = Field(None, description="User ID")


class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis."""

    id: str = Field(..., description="Analysis result ID")
    document_id: str = Field(..., description="Document ID")
    analysis_id: str = Field(..., description="Analysis ID")
    status: AnalysisResultStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Creation timestamp")


class AnalysisResultResponse(BaseModel):
    """Response model for completed analysis results."""

    id: str = Field(..., description="Analysis result ID")
    document_id: str = Field(..., description="Document ID")
    analysis_id: str = Field(..., description="Analysis ID")
    analysis_title: str = Field(..., description="Analysis title")
    document_title: str = Field(..., description="Document title")
    results: List[AnalysisResultItem] = Field(..., description="Analysis results")
    status: AnalysisResultStatus = Field(..., description="Processing status")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    user_id: Optional[str] = Field(None, description="User ID")

    # Processing information
    processing_time: Optional[float] = Field(
        None, description="Processing time in seconds"
    )
    total_items: int = Field(..., description="Total number of items")
    completed_items: int = Field(..., description="Number of completed items")

    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
