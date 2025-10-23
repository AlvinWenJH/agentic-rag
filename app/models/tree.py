"""
Topic tree data models and schemas.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from bson import ObjectId

from app.models.document import PyObjectId


class NodeType(str, Enum):
    """Topic tree node type."""
    ROOT = "root"
    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    TOPIC = "topic"
    SUBTOPIC = "subtopic"
    LEAF = "leaf"


class TreeStatus(str, Enum):
    """Tree processing status."""
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"
    UPDATING = "updating"


class TopicNode(BaseModel):
    """Individual topic node in the tree."""
    id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., description="Node title")
    description: Optional[str] = Field(None, description="Node description")
    node_type: NodeType = Field(..., description="Type of node")
    level: int = Field(..., description="Depth level in tree (0 = root)")
    
    # Content references
    page_numbers: List[int] = Field(default_factory=list, description="Referenced page numbers")
    image_references: List[str] = Field(default_factory=list, description="Referenced image paths")
    content_summary: Optional[str] = Field(None, description="Summary of content in this node")
    
    # Tree structure
    parent_id: Optional[str] = Field(None, description="Parent node ID")
    children_ids: List[str] = Field(default_factory=list, description="Child node IDs")
    
    # Metadata
    confidence_score: Optional[float] = Field(None, description="AI confidence in node classification")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")


class TreeBase(BaseModel):
    """Base tree model."""
    document_id: str = Field(..., description="Associated document ID")
    title: str = Field(..., description="Tree title")
    description: Optional[str] = Field(None, description="Tree description")
    version: int = Field(default=1, description="Tree version number")


class TreeCreate(TreeBase):
    """Tree creation model."""
    user_id: str = Field(..., description="User ID who owns the tree")


class TreeUpdate(BaseModel):
    """Tree update model."""
    title: Optional[str] = Field(None, description="Tree title")
    description: Optional[str] = Field(None, description="Tree description")
    nodes: Optional[List[TopicNode]] = Field(None, description="Updated tree nodes")


class Tree(TreeBase):
    """Complete tree model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who owns the tree")
    status: TreeStatus = Field(default=TreeStatus.GENERATING, description="Tree generation status")
    
    # Tree structure
    nodes: List[TopicNode] = Field(default_factory=list, description="All nodes in the tree")
    root_node_id: Optional[str] = Field(None, description="Root node ID")
    
    # Generation metadata
    generation_method: str = Field(default="gemini_visual", description="Method used to generate tree")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    node_count: int = Field(default=0, description="Total number of nodes")
    max_depth: int = Field(default=0, description="Maximum tree depth")
    
    # Processing information
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    processing_logs: List[str] = Field(default_factory=list, description="Processing logs")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class TreeResponse(BaseModel):
    """Tree response model."""
    id: str = Field(..., description="Tree ID")
    document_id: str = Field(..., description="Associated document ID")
    title: str = Field(..., description="Tree title")
    description: Optional[str] = Field(None, description="Tree description")
    status: TreeStatus = Field(..., description="Tree generation status")
    version: int = Field(..., description="Tree version number")
    
    # Tree structure
    nodes: List[TopicNode] = Field(..., description="All nodes in the tree")
    root_node_id: Optional[str] = Field(None, description="Root node ID")
    
    # Generation metadata
    generation_method: str = Field(..., description="Method used to generate tree")
    generation_time: Optional[float] = Field(None, description="Generation time in seconds")
    node_count: int = Field(..., description="Total number of nodes")
    max_depth: int = Field(..., description="Maximum tree depth")
    
    # Processing information
    error_message: Optional[str] = Field(None, description="Error message if generation failed")
    
    # Timestamps
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class TreeListResponse(BaseModel):
    """Tree list response model."""
    trees: List[TreeResponse] = Field(..., description="List of trees")
    total: int = Field(..., description="Total number of trees")
    page: int = Field(..., description="Current page number")
    size: int = Field(..., description="Page size")
    pages: int = Field(..., description="Total number of pages")


class TreeGenerationRequest(BaseModel):
    """Tree generation request model."""
    document_id: str = Field(..., description="Document ID to generate tree for")
    title: Optional[str] = Field(None, description="Custom tree title")
    description: Optional[str] = Field(None, description="Custom tree description")
    generation_options: Dict[str, Any] = Field(default_factory=dict, description="Generation options")


class TreeGenerationResponse(BaseModel):
    """Tree generation response model."""
    id: str = Field(..., description="Tree ID")
    document_id: str = Field(..., description="Associated document ID")
    status: TreeStatus = Field(..., description="Generation status")
    message: str = Field(..., description="Generation status message")


class TreePatchOperation(BaseModel):
    """JSON Patch operation for tree modification."""
    op: str = Field(..., description="Operation type (add, remove, replace, move, copy, test)")
    path: str = Field(..., description="JSON Pointer path")
    value: Optional[Any] = Field(None, description="Value for operation")
    from_path: Optional[str] = Field(None, alias="from", description="Source path for move/copy operations")


class TreePatchRequest(BaseModel):
    """Tree patch request model."""
    operations: List[TreePatchOperation] = Field(..., description="List of patch operations")
    description: Optional[str] = Field(None, description="Description of changes")


class TreePatchResponse(BaseModel):
    """Tree patch response model."""
    id: str = Field(..., description="Tree ID")
    version: int = Field(..., description="New tree version")
    applied_operations: int = Field(..., description="Number of operations applied")
    message: str = Field(..., description="Patch status message")


class TreeSearchRequest(BaseModel):
    """Tree search request model."""
    query: str = Field(..., description="Search query")
    node_types: Optional[List[NodeType]] = Field(None, description="Filter by node types")
    max_results: int = Field(default=10, description="Maximum number of results")
    include_content: bool = Field(default=True, description="Include content summaries")


class TreeSearchResult(BaseModel):
    """Tree search result model."""
    node_id: str = Field(..., description="Matching node ID")
    title: str = Field(..., description="Node title")
    description: Optional[str] = Field(None, description="Node description")
    node_type: NodeType = Field(..., description="Node type")
    level: int = Field(..., description="Node level")
    relevance_score: float = Field(..., description="Relevance score (0-1)")
    content_summary: Optional[str] = Field(None, description="Content summary")
    path: List[str] = Field(..., description="Path from root to this node")


class TreeSearchResponse(BaseModel):
    """Tree search response model."""
    tree_id: str = Field(..., description="Tree ID")
    query: str = Field(..., description="Search query")
    results: List[TreeSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time: float = Field(..., description="Search time in seconds")


class TreeStats(BaseModel):
    """Tree statistics model."""
    total_trees: int = Field(..., description="Total number of trees")
    trees_by_status: Dict[TreeStatus, int] = Field(..., description="Trees grouped by status")
    average_node_count: float = Field(..., description="Average number of nodes per tree")
    average_depth: float = Field(..., description="Average tree depth")
    average_generation_time: Optional[float] = Field(None, description="Average generation time in seconds")
    recent_generations: int = Field(..., description="Number of recent generations (last 24 hours)")