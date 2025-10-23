"""
Document data models and schemas.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId


class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentType(str, Enum):
    """Document type."""
    PDF = "pdf"


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}


class DocumentBase(BaseModel):
    """Base document model."""
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    description: Optional[str] = Field(None, description="Document description")
    tags: List[str] = Field(default_factory=list, description="Document tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DocumentCreate(DocumentBase):
    """Document creation model."""
    user_id: str = Field(..., description="User ID who uploaded the document")


class DocumentUpdate(BaseModel):
    """Document update model."""
    title: Optional[str] = Field(None, description="Document title")
    description: Optional[str] = Field(None, description="Document description")
    tags: Optional[List[str]] = Field(None, description="Document tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class Document(DocumentBase):
    """Complete document model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who uploaded the document")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED, description="Processing status")
    document_type: DocumentType = Field(..., description="Document type")
    
    # Storage paths
    storage_path: str = Field(..., description="Storage path in MinIO")
    images_path: Optional[str] = Field(None, description="Images storage path")
    processed_path: Optional[str] = Field(None, description="Processed data storage path")
    
    # Processing information
    page_count: Optional[int] = Field(None, description="Number of pages")
    image_count: Optional[int] = Field(None, description="Number of images generated")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class DocumentResponse(BaseModel):
    """Document response model."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME content type")
    file_size: int = Field(..., description="File size in bytes")
    status: DocumentStatus = Field(..., description="Processing status")
    document_type: DocumentType = Field(..., description="Document type")
    description: Optional[str] = Field(None, description="Document description")
    tags: List[str] = Field(..., description="Document tags")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")
    
    # Processing information
    page_count: Optional[int] = Field(None, description="Number of pages")
    image_count: Optional[int] = Field(None, description="Number of images generated")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    processed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentResponse] = Field(..., description="List of documents")
    total: int = Field(..., description="Total number of documents")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    filename: str = Field(..., description="Original filename")
    status: DocumentStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Upload status message")


class DocumentProcessingStatus(BaseModel):
    """Document processing status model."""
    id: str = Field(..., description="Document ID")
    status: DocumentStatus = Field(..., description="Processing status")
    progress: Optional[float] = Field(None, description="Processing progress (0-100)")
    current_step: Optional[str] = Field(None, description="Current processing step")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")


class DocumentStats(BaseModel):
    """Document statistics model."""
    total_documents: int = Field(..., description="Total number of documents")
    documents_by_status: Dict[DocumentStatus, int] = Field(..., description="Documents grouped by status")
    documents_by_type: Dict[DocumentType, int] = Field(..., description="Documents grouped by type")
    total_file_size: int = Field(..., description="Total file size in bytes")
    average_processing_time: Optional[float] = Field(None, description="Average processing time in seconds")
    recent_uploads: int = Field(..., description="Number of recent uploads (last 24 hours)")