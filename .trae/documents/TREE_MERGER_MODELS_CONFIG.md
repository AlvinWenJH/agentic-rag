# Tree Merger Models and Configuration

## File: app/models/tree.py (New File)

```python
"""
Tree-related Pydantic models for document tree merging and management.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
from bson import ObjectId

class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    
    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)
    
    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MergeOperation(str, Enum):
    """Types of merge operations for tree nodes."""
    MERGE_NODES = "merge_nodes"
    COMBINE_CHILDREN = "combine_children"
    CONSOLIDATE_CONTENT = "consolidate_content"
    UPDATE_HIERARCHY = "update_hierarchy"
    SPLIT_NODE = "split_node"
    RELOCATE_NODE = "relocate_node"

class NodeReference(BaseModel):
    """Reference to a specific node in a subtree structure."""
    page_number: int = Field(..., description="Source page number")
    section_index: int = Field(..., description="Section index in page tree")
    subject_index: Optional[int] = Field(None, description="Subject index (if applicable)")
    topic_index: Optional[int] = Field(None, description="Topic index (if applicable)")
    node_path: str = Field(..., description="JSON path to the node (e.g., '/sections/0/subjects/1')")
    node_id: Optional[str] = Field(None, description="Unique node identifier")

class MergePatch(BaseModel):
    """Structured JSON Patch operation for intelligent tree merging."""
    operation: MergeOperation = Field(..., description="Type of merge operation to perform")
    target_path: str = Field(..., description="JSON path where merge result will be placed")
    source_nodes: List[NodeReference] = Field(..., description="Source nodes to be merged")
    merge_strategy: str = Field(..., description="Strategy used for merging content")
    merged_content: Dict[str, Any] = Field(..., description="Resulting merged content structure")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="AI confidence in merge decision")
    reasoning: str = Field(..., description="AI reasoning and justification for the merge")
    page_associations: List[int] = Field(..., description="All associated page numbers")
    keywords: List[str] = Field(default_factory=list, description="Combined and deduplicated keywords")
    priority: int = Field(default=1, description="Merge priority (1=highest, 5=lowest)")
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v

class TreeMergeResult(BaseModel):
    """Result from AI-powered tree merging analysis."""
    merge_patches: List[MergePatch] = Field(..., description="Generated merge patches")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics and metadata")
    analysis_summary: str = Field(default="", description="Summary of merge analysis")
    total_confidence: float = Field(default=0.0, description="Overall confidence in merge operations")

class DocumentTreeNode(BaseModel):
    """Complete document tree node with hierarchical structure."""
    id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., description="Node title or heading")
    summary: str = Field(default="", description="Node content summary")
    content: str = Field(default="", description="Full node content")
    level: int = Field(..., ge=1, le=3, description="Hierarchical level (1=Section, 2=Subject, 3=Topic)")
    children: List['DocumentTreeNode'] = Field(default_factory=list, description="Child nodes")
    page_numbers: List[int] = Field(..., description="Associated page numbers")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")
    confidence_score: float = Field(default=1.0, description="Merge confidence score")
    source_nodes: List[NodeReference] = Field(default_factory=list, description="Original source nodes")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional node metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('level')
    def validate_level(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError('Level must be 1 (Section), 2 (Subject), or 3 (Topic)')
        return v

class CompleteDocumentTree(BaseModel):
    """Complete merged document tree structure."""
    document_id: str = Field(..., description="Source document identifier")
    title: str = Field(..., description="Document title")
    root_nodes: List[DocumentTreeNode] = Field(..., description="Root level sections")
    total_pages: int = Field(..., ge=1, description="Total pages processed")
    merge_statistics: Dict[str, int] = Field(default_factory=dict, description="Merge operation statistics")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata and metrics")
    quality_metrics: Dict[str, float] = Field(default_factory=dict, description="Tree quality metrics")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default="1.0", description="Tree structure version")
    status: str = Field(default="completed", description="Tree processing status")

class TreeProcessingStatus(str, Enum):
    """Tree processing status values."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TreeGenerationRequest(BaseModel):
    """Request model for tree generation."""
    document_id: str = Field(..., description="Document ID to process")
    merge_strategy: str = Field(default="intelligent", description="Merge strategy to use")
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence for merges")
    max_depth: int = Field(default=3, ge=1, le=5, description="Maximum tree depth")
    preserve_structure: bool = Field(default=True, description="Preserve original structure when possible")

class TreeGenerationResponse(BaseModel):
    """Response model for tree generation."""
    message: str = Field(..., description="Response message")
    document_id: str = Field(..., description="Document ID")
    tree_id: Optional[str] = Field(None, description="Generated tree ID")
    status: TreeProcessingStatus = Field(..., description="Processing status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class TreeListResponse(BaseModel):
    """Response model for tree listing."""
    trees: List[Dict[str, Any]] = Field(..., description="List of trees")
    total: int = Field(..., description="Total number of trees")
    limit: int = Field(..., description="Results limit")
    skip: int = Field(..., description="Results offset")

class TreeStatsResponse(BaseModel):
    """Response model for tree statistics."""
    document_id: str = Field(..., description="Document ID")
    total_nodes: int = Field(..., description="Total number of nodes")
    sections_count: int = Field(..., description="Number of sections")
    subjects_count: int = Field(..., description="Number of subjects")
    topics_count: int = Field(..., description="Number of topics")
    merge_operations: int = Field(..., description="Number of merge operations applied")
    confidence_average: float = Field(..., description="Average confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    token_usage: Dict[str, int] = Field(..., description="AI token usage statistics")

# Update DocumentTreeNode to allow forward references
DocumentTreeNode.model_rebuild()
```

## File: app/core/config.py (Updates)

Add these configuration settings to the existing config file:

```python
# Tree Merger Service Configuration
TREE_MERGER_ENABLED: bool = Field(default=True, description="Enable tree merger service")
TREE_MERGE_CONFIDENCE_THRESHOLD: float = Field(default=0.7, description="Minimum confidence for merges")
TREE_MERGE_MAX_DEPTH: int = Field(default=3, description="Maximum tree depth")
TREE_MERGE_BATCH_SIZE: int = Field(default=10, description="Batch size for processing subtrees")
TREE_MERGE_TIMEOUT: int = Field(default=300, description="Timeout for tree merging in seconds")

# AI Model Configuration for Tree Merging
TREE_MERGE_MODEL: str = Field(default="gemini-1.5-pro", description="AI model for tree merging")
TREE_MERGE_TEMPERATURE: float = Field(default=0.3, description="Temperature for tree merge analysis")
TREE_MERGE_MAX_TOKENS: int = Field(default=8192, description="Maximum tokens for tree merge requests")

# MongoDB Collections
TREES_COLLECTION: str = Field(default="trees", description="MongoDB trees collection name")
SUBTREES_COLLECTION: str = Field(default="subtrees", description="MongoDB subtrees collection name")

# Performance and Monitoring
TREE_MERGE_ENABLE_METRICS: bool = Field(default=True, description="Enable tree merge metrics")
TREE_MERGE_LOG_LEVEL: str = Field(default="INFO", description="Log level for tree merger")
```

## File: app/core/exceptions.py (Updates)

Add tree-specific exceptions:

```python
class TreeMergeError(ProcessingError):
    """Exception raised during tree merging operations."""
    pass

class TreeNotFoundError(ValidationError):
    """Exception raised when requested tree is not found."""
    pass

class TreeGenerationError(ProcessingError):
    """Exception raised during tree generation."""
    pass

class MergePatchError(ValidationError):
    """Exception raised when merge patch is invalid."""
    pass

class TreeValidationError(ValidationError):
    """Exception raised when tree structure validation fails."""
    pass
```

## File: app/utils/tree_utils.py (New File)

```python
"""
Utility functions for tree operations and validation.
"""

from typing import List, Dict, Any, Optional, Tuple
import json
import hashlib
from datetime import datetime

from app.models.tree import DocumentTreeNode, CompleteDocumentTree, NodeReference

def validate_tree_structure(tree: CompleteDocumentTree) -> Tuple[bool, List[str]]:
    """
    Validate tree structure for consistency and completeness.
    
    Args:
        tree: Complete document tree to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check if tree has root nodes
    if not tree.root_nodes:
        errors.append("Tree must have at least one root node")
    
    # Validate each root node
    for i, root_node in enumerate(tree.root_nodes):
        node_errors = _validate_node_recursive(root_node, f"root[{i}]", 1)
        errors.extend(node_errors)
    
    # Check for duplicate node IDs
    all_node_ids = _collect_all_node_ids(tree.root_nodes)
    if len(all_node_ids) != len(set(all_node_ids)):
        errors.append("Duplicate node IDs found in tree")
    
    # Validate page number consistency
    all_pages = set()
    for node in tree.root_nodes:
        all_pages.update(_collect_page_numbers(node))
    
    if tree.total_pages != len(all_pages):
        errors.append(f"Total pages mismatch: expected {tree.total_pages}, found {len(all_pages)}")
    
    return len(errors) == 0, errors

def _validate_node_recursive(node: DocumentTreeNode, path: str, expected_level: int) -> List[str]:
    """Recursively validate a node and its children."""
    errors = []
    
    # Check level consistency
    if node.level != expected_level:
        errors.append(f"Node at {path} has incorrect level: expected {expected_level}, got {node.level}")
    
    # Check required fields
    if not node.title.strip():
        errors.append(f"Node at {path} has empty title")
    
    if not node.id:
        errors.append(f"Node at {path} has empty ID")
    
    if not node.page_numbers:
        errors.append(f"Node at {path} has no associated page numbers")
    
    # Validate children
    if node.level < 3:  # Sections and Subjects can have children
        for i, child in enumerate(node.children):
            child_errors = _validate_node_recursive(child, f"{path}.children[{i}]", expected_level + 1)
            errors.extend(child_errors)
    elif node.children:  # Topics should not have children
        errors.append(f"Topic node at {path} should not have children")
    
    return errors

def _collect_all_node_ids(nodes: List[DocumentTreeNode]) -> List[str]:
    """Collect all node IDs from a tree structure."""
    ids = []
    for node in nodes:
        ids.append(node.id)
        ids.extend(_collect_all_node_ids(node.children))
    return ids

def _collect_page_numbers(node: DocumentTreeNode) -> List[int]:
    """Collect all page numbers from a node and its children."""
    pages = list(node.page_numbers)
    for child in node.children:
        pages.extend(_collect_page_numbers(child))
    return pages

def calculate_tree_metrics(tree: CompleteDocumentTree) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for a document tree.
    
    Args:
        tree: Complete document tree
        
    Returns:
        Dictionary of tree metrics
    """
    metrics = {
        "total_nodes": 0,
        "sections_count": 0,
        "subjects_count": 0,
        "topics_count": 0,
        "max_depth": 0,
        "avg_children_per_node": 0.0,
        "total_keywords": 0,
        "unique_keywords": set(),
        "page_coverage": set(),
        "confidence_scores": [],
    }
    
    def analyze_node(node: DocumentTreeNode, depth: int):
        metrics["total_nodes"] += 1
        metrics["max_depth"] = max(metrics["max_depth"], depth)
        
        # Count by level
        if node.level == 1:
            metrics["sections_count"] += 1
        elif node.level == 2:
            metrics["subjects_count"] += 1
        elif node.level == 3:
            metrics["topics_count"] += 1
        
        # Collect keywords
        metrics["total_keywords"] += len(node.keywords)
        metrics["unique_keywords"].update(node.keywords)
        
        # Collect page numbers
        metrics["page_coverage"].update(node.page_numbers)
        
        # Collect confidence scores
        metrics["confidence_scores"].append(node.confidence_score)
        
        # Analyze children
        for child in node.children:
            analyze_node(child, depth + 1)
    
    # Analyze all root nodes
    for root_node in tree.root_nodes:
        analyze_node(root_node, 1)
    
    # Calculate averages
    if metrics["total_nodes"] > 0:
        total_children = sum(len(node.children) for node in _get_all_nodes(tree.root_nodes))
        metrics["avg_children_per_node"] = total_children / metrics["total_nodes"]
        metrics["avg_confidence"] = sum(metrics["confidence_scores"]) / len(metrics["confidence_scores"])
    
    # Convert sets to counts
    metrics["unique_keywords_count"] = len(metrics["unique_keywords"])
    metrics["page_coverage_count"] = len(metrics["page_coverage"])
    
    # Remove sets (not JSON serializable)
    del metrics["unique_keywords"]
    del metrics["page_coverage"]
    
    return metrics

def _get_all_nodes(nodes: List[DocumentTreeNode]) -> List[DocumentTreeNode]:
    """Get all nodes from a tree structure (flattened)."""
    all_nodes = []
    for node in nodes:
        all_nodes.append(node)
        all_nodes.extend(_get_all_nodes(node.children))
    return all_nodes

def generate_tree_hash(tree: CompleteDocumentTree) -> str:
    """
    Generate a hash for tree structure to detect changes.
    
    Args:
        tree: Complete document tree
        
    Returns:
        SHA-256 hash of tree structure
    """
    # Create a simplified representation for hashing
    tree_data = {
        "document_id": tree.document_id,
        "total_pages": tree.total_pages,
        "nodes": _serialize_nodes_for_hash(tree.root_nodes)
    }
    
    tree_json = json.dumps(tree_data, sort_keys=True)
    return hashlib.sha256(tree_json.encode()).hexdigest()

def _serialize_nodes_for_hash(nodes: List[DocumentTreeNode]) -> List[Dict]:
    """Serialize nodes for hash generation."""
    serialized = []
    for node in nodes:
        node_data = {
            "id": node.id,
            "title": node.title,
            "level": node.level,
            "page_numbers": sorted(node.page_numbers),
            "children": _serialize_nodes_for_hash(node.children)
        }
        serialized.append(node_data)
    return serialized

def find_node_by_id(tree: CompleteDocumentTree, node_id: str) -> Optional[DocumentTreeNode]:
    """
    Find a node by its ID in the tree.
    
    Args:
        tree: Complete document tree
        node_id: Node ID to search for
        
    Returns:
        Found node or None
    """
    def search_nodes(nodes: List[DocumentTreeNode]) -> Optional[DocumentTreeNode]:
        for node in nodes:
            if node.id == node_id:
                return node
            found = search_nodes(node.children)
            if found:
                return found
        return None
    
    return search_nodes(tree.root_nodes)

def get_node_path(tree: CompleteDocumentTree, node_id: str) -> Optional[str]:
    """
    Get the path to a node in the tree.
    
    Args:
        tree: Complete document tree
        node_id: Node ID to find path for
        
    Returns:
        Path string or None if not found
    """
    def find_path(nodes: List[DocumentTreeNode], path: str = "") -> Optional[str]:
        for i, node in enumerate(nodes):
            current_path = f"{path}/root_nodes[{i}]" if not path else f"{path}/children[{i}]"
            
            if node.id == node_id:
                return current_path
            
            found_path = find_path(node.children, current_path)
            if found_path:
                return found_path
        
        return None
    
    return find_path(tree.root_nodes)

def export_tree_to_json(tree: CompleteDocumentTree, include_metadata: bool = True) -> str:
    """
    Export tree to JSON format.
    
    Args:
        tree: Complete document tree
        include_metadata: Whether to include processing metadata
        
    Returns:
        JSON string representation
    """
    tree_dict = tree.model_dump()
    
    if not include_metadata:
        # Remove metadata fields
        tree_dict.pop("processing_metadata", None)
        tree_dict.pop("merge_statistics", None)
        tree_dict.pop("quality_metrics", None)
    
    return json.dumps(tree_dict, indent=2, default=str)
```

## File: requirements.txt (Updates)

Add these dependencies:

```txt
# Tree Merging Dependencies
pydantic-ai>=0.0.13
jsonpatch>=1.33
jsonschema>=4.17.3
deepdiff>=6.7.1

# Enhanced NLP for keyword extraction (optional)
nltk>=3.8.1
spacy>=3.7.2
```

## File: app/core/logging.py (Updates)

Add tree-specific logging configuration:

```python
# Tree Merger Logging Configuration
TREE_MERGER_LOGGER = "tree_merger"

def configure_tree_merger_logging():
    """Configure logging for tree merger service."""
    logger = structlog.get_logger(TREE_MERGER_LOGGER)
    
    # Add tree-specific processors
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    return logger
```

This comprehensive implementation provides:

1. **Complete Data Models**: All Pydantic models for tree merging operations
2. **Configuration Updates**: Settings for tree merger service
3. **Exception Handling**: Tree-specific exceptions
4. **Utility Functions**: Tree validation, metrics, and manipulation utilities
5. **Logging Configuration**: Structured logging for tree operations
6. **Dependencies**: Required packages for the implementation

The models follow the same patterns as the existing codebase and provide full type safety and validation for the tree merging system.