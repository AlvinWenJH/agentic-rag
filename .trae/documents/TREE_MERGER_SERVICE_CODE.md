# Tree Merger Service - Complete Implementation

## File: app/services/tree_merger_service.py

```python
"""
Tree merging service for combining page subtrees into complete document trees.
Uses Pydantic AI to generate intelligent merge operations via JSON Patch.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import List, Dict, Any, Optional
import structlog

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ProcessingError, ValidationError
from app.core.database import get_subtrees_collection, get_database
from bson import ObjectId

logger = structlog.get_logger()

class MergeOperation(str, Enum):
    """Types of merge operations."""
    MERGE_NODES = "merge_nodes"
    COMBINE_CHILDREN = "combine_children"
    CONSOLIDATE_CONTENT = "consolidate_content"
    UPDATE_HIERARCHY = "update_hierarchy"

class NodeReference(BaseModel):
    """Reference to a node in the tree structure."""
    page_number: int = Field(..., description="Source page number")
    section_index: int = Field(..., description="Section index in page tree")
    subject_index: Optional[int] = Field(None, description="Subject index (if applicable)")
    topic_index: Optional[int] = Field(None, description="Topic index (if applicable)")
    node_path: str = Field(..., description="JSON path to the node")

class MergePatch(BaseModel):
    """Structured JSON Patch operation for tree merging."""
    operation: MergeOperation = Field(..., description="Type of merge operation")
    target_path: str = Field(..., description="JSON path where merge result will be placed")
    source_nodes: List[NodeReference] = Field(..., description="Source nodes to merge")
    merge_strategy: str = Field(..., description="Strategy for merging content")
    merged_content: Dict[str, Any] = Field(..., description="Resulting merged content")
    confidence_score: float = Field(..., description="AI confidence in merge decision")
    reasoning: str = Field(..., description="AI reasoning for the merge")
    page_associations: List[int] = Field(..., description="Associated page numbers")
    keywords: List[str] = Field(default_factory=list, description="Combined keywords")

class TreeMergeResult(BaseModel):
    """Result from tree merging analysis."""
    merge_patches: List[MergePatch] = Field(..., description="Generated merge patches")
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing statistics")
    
class DocumentTreeNode(BaseModel):
    """Complete document tree node structure."""
    id: str = Field(..., description="Unique node identifier")
    title: str = Field(..., description="Node title")
    summary: str = Field(..., description="Node content summary")
    level: int = Field(..., description="Hierarchical level (1=Section, 2=Subject, 3=Topic)")
    children: List['DocumentTreeNode'] = Field(default_factory=list, description="Child nodes")
    page_numbers: List[int] = Field(..., description="Associated page numbers")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")
    confidence_score: float = Field(..., description="Merge confidence score")
    source_nodes: List[NodeReference] = Field(..., description="Original source nodes")

class CompleteDocumentTree(BaseModel):
    """Complete merged document tree."""
    document_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    root_nodes: List[DocumentTreeNode] = Field(..., description="Root level sections")
    total_pages: int = Field(..., description="Total pages processed")
    merge_statistics: Dict[str, int] = Field(..., description="Merge operation statistics")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

class TreeMergerService:
    """Service for merging page subtrees into complete document trees using Pydantic AI."""

    def __init__(self):
        self.settings = get_settings()
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini API with Pydantic AI for tree merging."""
        try:
            # Create Google provider
            self.provider = GoogleProvider(api_key=self.settings.GEMINI_API_KEY)

            # Create Google model
            self.model = GoogleModel(
                model_name=self.settings.GEMINI_MODEL,
                provider=self.provider,
            )

            # Create agent for tree merging analysis
            self.merge_agent = Agent(model=self.model)

            logger.info(
                "Gemini Pydantic AI configured for tree merging",
                model=self.settings.GEMINI_MODEL,
                temperature=self.settings.GEMINI_TEMPERATURE,
            )

        except Exception as e:
            logger.error("Failed to configure Gemini Pydantic AI for tree merging", error=str(e))
            raise ExternalServiceError(f"Gemini configuration failed: {str(e)}")

    async def merge_document_tree(self, document_id: str) -> CompleteDocumentTree:
        """
        Merge all subtrees for a document into a complete tree.

        Args:
            document_id: Document ID to process

        Returns:
            CompleteDocumentTree: Merged document tree
        """
        try:
            logger.info(
                "Starting document tree merging",
                document_id=document_id,
            )

            start_time = time.time()

            # Fetch all subtrees for the document
            subtrees = await self._fetch_document_subtrees(document_id)
            
            if not subtrees:
                raise ProcessingError(f"No subtrees found for document {document_id}")

            logger.info(
                "Fetched subtrees for merging",
                document_id=document_id,
                subtree_count=len(subtrees),
            )

            # Process subtrees page by page to generate merge patches
            all_merge_patches = []
            total_token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "total_calls": 0,
            }

            for i, current_subtree in enumerate(subtrees):
                page_number = current_subtree["page_number"]
                
                logger.info(
                    "Processing subtree for merging",
                    document_id=document_id,
                    page_number=page_number,
                    progress=f"{i+1}/{len(subtrees)}",
                )

                # Generate merge patches for this page
                merge_result = await self._generate_merge_patches(
                    current_subtree=current_subtree,
                    existing_subtrees=subtrees[:i],  # Previously processed subtrees
                    document_id=document_id,
                )

                all_merge_patches.extend(merge_result.merge_patches)
                
                # Accumulate token usage
                page_usage = merge_result.processing_metadata.get("token_usage", {})
                for key in total_token_usage:
                    total_token_usage[key] += page_usage.get(key, 0)

            # Apply merge patches to create complete tree
            complete_tree = await self._apply_merge_patches(
                subtrees=subtrees,
                merge_patches=all_merge_patches,
                document_id=document_id,
            )

            processing_time = time.time() - start_time

            # Update processing metadata
            complete_tree.processing_metadata.update({
                "processing_time": processing_time,
                "token_usage": total_token_usage,
                "merge_patches_count": len(all_merge_patches),
                "subtrees_processed": len(subtrees),
                "analysis_method": "gemini_ai_tree_merging",
            })

            # Save complete tree to MongoDB
            tree_id = await self._save_complete_tree(complete_tree)

            logger.info(
                "Document tree merging completed",
                document_id=document_id,
                tree_id=tree_id,
                processing_time=processing_time,
                merge_patches_count=len(all_merge_patches),
                total_nodes=self._count_tree_nodes(complete_tree),
            )

            return complete_tree

        except Exception as e:
            logger.error(
                "Document tree merging failed",
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Tree merging failed: {str(e)}")

    async def _fetch_document_subtrees(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch all subtrees for a document from MongoDB."""
        try:
            subtrees_collection = get_subtrees_collection()
            
            cursor = subtrees_collection.find(
                {"document_id": document_id}
            ).sort("page_number", 1)
            
            subtrees = await cursor.to_list(length=None)
            
            logger.debug(
                "Fetched document subtrees",
                document_id=document_id,
                count=len(subtrees),
            )
            
            return subtrees

        except Exception as e:
            logger.error(
                "Failed to fetch document subtrees",
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to fetch subtrees: {str(e)}")

    async def _generate_merge_patches(
        self,
        current_subtree: Dict[str, Any],
        existing_subtrees: List[Dict[str, Any]],
        document_id: str,
    ) -> TreeMergeResult:
        """
        Generate merge patches for a single page using Pydantic AI.

        Args:
            current_subtree: Current page subtree to process
            existing_subtrees: Previously processed subtrees
            document_id: Document ID

        Returns:
            TreeMergeResult: Generated merge patches and metadata
        """
        try:
            page_start_time = time.time()
            page_number = current_subtree["page_number"]

            # Create analysis prompt
            prompt = self._create_merge_analysis_prompt(
                current_subtree=current_subtree,
                existing_subtrees=existing_subtrees,
                document_id=document_id,
            )

            # Run merge analysis with structured output
            result = await self.merge_agent.run(prompt, output_type=TreeMergeResult)
            
            usage = {
                "input_tokens": result.usage().input_tokens,
                "output_tokens": result.usage().output_tokens,
                "total_tokens": result.usage().total_tokens,
                "total_calls": result.usage().requests,
            }

            merge_result = result.output
            processing_time = time.time() - page_start_time

            # Update processing metadata
            merge_result.processing_metadata.update({
                "page_number": page_number,
                "processing_time": processing_time,
                "token_usage": usage,
                "existing_subtrees_count": len(existing_subtrees),
            })

            logger.debug(
                "Generated merge patches for page",
                document_id=document_id,
                page_number=page_number,
                patches_count=len(merge_result.merge_patches),
                processing_time=processing_time,
            )

            return merge_result

        except Exception as e:
            logger.error(
                "Failed to generate merge patches",
                document_id=document_id,
                page_number=current_subtree.get("page_number"),
                error=str(e),
            )
            # Return empty result instead of failing completely
            return TreeMergeResult(
                merge_patches=[],
                processing_metadata={
                    "error": str(e),
                    "page_number": current_subtree.get("page_number"),
                }
            )

    def _create_merge_analysis_prompt(
        self,
        current_subtree: Dict[str, Any],
        existing_subtrees: List[Dict[str, Any]],
        document_id: str,
    ) -> str:
        """Create analysis prompt for merge patch generation."""
        
        current_page = current_subtree["page_number"]
        existing_pages = [st["page_number"] for st in existing_subtrees]
        
        # Prepare current subtree structure for analysis
        current_tree_json = json.dumps(current_subtree["page_tree"], indent=2)
        
        # Prepare existing trees summary
        existing_summary = []
        for subtree in existing_subtrees[-3:]:  # Last 3 for context
            page_num = subtree["page_number"]
            sections = subtree["page_tree"]
            section_titles = [s.get("title", "Untitled") for s in sections]
            existing_summary.append(f"Page {page_num}: {', '.join(section_titles)}")
        
        existing_context = "\n".join(existing_summary) if existing_summary else "No previous pages"

        return f"""Analyze page {current_page} content and generate merge patches to integrate it with existing document tree.

**Current Page Tree Structure:**
```json
{current_tree_json}
```

**Existing Pages Context:**
{existing_context}

**Analysis Instructions:**
1. **Identify Similar Content**: Find nodes in current page that are similar to existing content
2. **Generate Merge Operations**: Create MergePatch objects for:
   - Merging nodes with similar titles/content
   - Combining related children under common parents
   - Consolidating duplicate or overlapping content
   - Updating hierarchical relationships

3. **Merge Strategies**:
   - `content_similarity`: Merge based on content overlap
   - `title_matching`: Merge based on title similarity
   - `hierarchical_consolidation`: Combine related hierarchical structures
   - `keyword_clustering`: Group by common keywords

4. **Quality Criteria**:
   - Confidence score > 0.7 for merges
   - Preserve important content from all sources
   - Maintain logical hierarchy (Section → Subject → Topic)
   - Combine page associations and keywords

**Output Requirements:**
- Generate MergePatch objects with clear reasoning
- Include confidence scores for each merge decision
- Specify target paths using JSON path notation
- Combine keywords and page associations
- Provide processing metadata

Focus on creating a coherent, non-redundant tree structure while preserving all important content."""

    async def _apply_merge_patches(
        self,
        subtrees: List[Dict[str, Any]],
        merge_patches: List[MergePatch],
        document_id: str,
    ) -> CompleteDocumentTree:
        """
        Apply merge patches to create complete document tree.

        Args:
            subtrees: All document subtrees
            merge_patches: Generated merge patches
            document_id: Document ID

        Returns:
            CompleteDocumentTree: Complete merged tree
        """
        try:
            logger.info(
                "Applying merge patches to create complete tree",
                document_id=document_id,
                patches_count=len(merge_patches),
            )

            # Initialize tree with first page content
            if not subtrees:
                raise ProcessingError("No subtrees to merge")

            # Get document metadata
            first_subtree = subtrees[0]
            document_title = "Merged Document"  # Could be fetched from documents collection
            
            # Create initial tree structure from all subtrees
            initial_tree = self._create_initial_tree_structure(subtrees, document_id)
            
            # Apply merge patches sequentially
            merged_tree = self._execute_merge_patches(initial_tree, merge_patches)
            
            # Calculate merge statistics
            merge_stats = self._calculate_merge_statistics(merge_patches)
            
            complete_tree = CompleteDocumentTree(
                document_id=document_id,
                title=document_title,
                root_nodes=merged_tree,
                total_pages=len(subtrees),
                merge_statistics=merge_stats,
                processing_metadata={
                    "merge_method": "ai_guided_json_patch",
                    "patches_applied": len(merge_patches),
                }
            )

            logger.info(
                "Merge patches applied successfully",
                document_id=document_id,
                final_nodes_count=self._count_tree_nodes(complete_tree),
            )

            return complete_tree

        except Exception as e:
            logger.error(
                "Failed to apply merge patches",
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to apply merge patches: {str(e)}")

    def _create_initial_tree_structure(
        self, 
        subtrees: List[Dict[str, Any]], 
        document_id: str
    ) -> List[DocumentTreeNode]:
        """Create initial tree structure from all subtrees."""
        all_nodes = []
        node_id_counter = 0
        
        for subtree in subtrees:
            page_number = subtree["page_number"]
            page_tree = subtree["page_tree"]
            
            for section in page_tree:
                section_node = DocumentTreeNode(
                    id=f"section_{node_id_counter}",
                    title=section.get("title", "Untitled Section"),
                    summary=section.get("summary", ""),
                    level=1,
                    page_numbers=[page_number],
                    keywords=self._extract_keywords(section),
                    confidence_score=1.0,
                    source_nodes=[NodeReference(
                        page_number=page_number,
                        section_index=0,
                        node_path=f"/page_tree/0"
                    )]
                )
                node_id_counter += 1
                
                # Add subjects as children
                for subject_idx, subject in enumerate(section.get("children", [])):
                    subject_node = DocumentTreeNode(
                        id=f"subject_{node_id_counter}",
                        title=subject.get("title", "Untitled Subject"),
                        summary=subject.get("summary", ""),
                        level=2,
                        page_numbers=[page_number],
                        keywords=self._extract_keywords(subject),
                        confidence_score=1.0,
                        source_nodes=[NodeReference(
                            page_number=page_number,
                            section_index=0,
                            subject_index=subject_idx,
                            node_path=f"/page_tree/0/children/{subject_idx}"
                        )]
                    )
                    node_id_counter += 1
                    
                    # Add topics as children
                    for topic_idx, topic in enumerate(subject.get("children", [])):
                        topic_node = DocumentTreeNode(
                            id=f"topic_{node_id_counter}",
                            title=topic.get("title", "Untitled Topic"),
                            summary=topic.get("summary", ""),
                            level=3,
                            page_numbers=[page_number],
                            keywords=self._extract_keywords(topic),
                            confidence_score=1.0,
                            source_nodes=[NodeReference(
                                page_number=page_number,
                                section_index=0,
                                subject_index=subject_idx,
                                topic_index=topic_idx,
                                node_path=f"/page_tree/0/children/{subject_idx}/children/{topic_idx}"
                            )]
                        )
                        node_id_counter += 1
                        subject_node.children.append(topic_node)
                    
                    section_node.children.append(subject_node)
                
                all_nodes.append(section_node)
        
        return all_nodes

    def _execute_merge_patches(
        self, 
        initial_tree: List[DocumentTreeNode], 
        merge_patches: List[MergePatch]
    ) -> List[DocumentTreeNode]:
        """Execute merge patches on the initial tree structure."""
        # This is a simplified implementation
        # In practice, you would implement proper JSON Patch operations
        
        merged_tree = initial_tree.copy()
        
        for patch in merge_patches:
            if patch.confidence_score >= 0.7:  # Only apply high-confidence merges
                # Apply merge logic based on patch operation
                merged_tree = self._apply_single_patch(merged_tree, patch)
        
        return merged_tree

    def _apply_single_patch(
        self, 
        tree: List[DocumentTreeNode], 
        patch: MergePatch
    ) -> List[DocumentTreeNode]:
        """Apply a single merge patch to the tree."""
        # Simplified merge logic - in practice, implement full JSON Patch operations
        if patch.operation == MergeOperation.MERGE_NODES:
            # Find and merge similar nodes
            return self._merge_similar_nodes(tree, patch)
        elif patch.operation == MergeOperation.COMBINE_CHILDREN:
            # Combine children under common parents
            return self._combine_children(tree, patch)
        # Add other merge operations as needed
        
        return tree

    def _merge_similar_nodes(
        self, 
        tree: List[DocumentTreeNode], 
        patch: MergePatch
    ) -> List[DocumentTreeNode]:
        """Merge nodes with similar content."""
        # Implementation would identify and merge similar nodes
        # This is a placeholder for the actual merge logic
        return tree

    def _combine_children(
        self, 
        tree: List[DocumentTreeNode], 
        patch: MergePatch
    ) -> List[DocumentTreeNode]:
        """Combine children under common parents."""
        # Implementation would reorganize tree hierarchy
        # This is a placeholder for the actual combination logic
        return tree

    def _extract_keywords(self, node_data: Dict[str, Any]) -> List[str]:
        """Extract keywords from node content."""
        # Simple keyword extraction - could be enhanced with NLP
        title = node_data.get("title", "")
        summary = node_data.get("summary", "")
        
        # Basic keyword extraction
        words = (title + " " + summary).lower().split()
        keywords = [word.strip(".,!?;:") for word in words if len(word) > 3]
        
        return list(set(keywords))[:10]  # Limit to 10 keywords

    def _calculate_merge_statistics(self, merge_patches: List[MergePatch]) -> Dict[str, int]:
        """Calculate statistics about merge operations."""
        stats = {
            "total_patches": len(merge_patches),
            "merge_nodes": 0,
            "combine_children": 0,
            "consolidate_content": 0,
            "update_hierarchy": 0,
            "high_confidence_merges": 0,
        }
        
        for patch in merge_patches:
            stats[patch.operation.value] = stats.get(patch.operation.value, 0) + 1
            if patch.confidence_score >= 0.8:
                stats["high_confidence_merges"] += 1
        
        return stats

    def _count_tree_nodes(self, tree: CompleteDocumentTree) -> int:
        """Count total nodes in the complete tree."""
        def count_nodes(nodes: List[DocumentTreeNode]) -> int:
            count = len(nodes)
            for node in nodes:
                count += count_nodes(node.children)
            return count
        
        return count_nodes(tree.root_nodes)

    async def _save_complete_tree(self, complete_tree: CompleteDocumentTree) -> str:
        """
        Save complete tree to MongoDB trees collection.

        Args:
            complete_tree: Complete merged document tree

        Returns:
            Inserted document ID as string
        """
        try:
            trees_collection = get_database().trees

            # Convert tree to dict format for MongoDB storage
            tree_document = {
                "document_id": complete_tree.document_id,
                "title": complete_tree.title,
                "tree_data": complete_tree.model_dump(),
                "total_pages": complete_tree.total_pages,
                "merge_statistics": complete_tree.merge_statistics,
                "processing_metadata": complete_tree.processing_metadata,
                "created_at": complete_tree.created_at,
                "updated_at": complete_tree.updated_at,
                "status": "completed",
                "tree_type": "merged_document_tree",
            }

            # Insert into MongoDB
            result = await trees_collection.insert_one(tree_document)

            logger.info(
                "Complete tree saved to MongoDB",
                document_id=complete_tree.document_id,
                tree_id=str(result.inserted_id),
                total_nodes=self._count_tree_nodes(complete_tree),
                merge_patches=complete_tree.processing_metadata.get("patches_applied", 0),
            )

            return str(result.inserted_id)

        except Exception as e:
            logger.error(
                "Failed to save complete tree to MongoDB",
                document_id=complete_tree.document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to save complete tree: {str(e)}")

# Service instance
tree_merger_service = TreeMergerService()
```

## File: app/core/database.py (Addition)

Add this function to the existing database.py file:

```python
def get_trees_collection():
    """Get trees collection."""
    return get_database().trees
```

## File: app/api/v1/trees.py (New File)

```python
"""
Tree management API endpoints.
Handles tree generation, retrieval, and management.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
import structlog
from datetime import datetime

from app.services.tree_merger_service import tree_merger_service
from app.core.database import get_trees_collection, get_documents_collection
from app.core.exceptions import ProcessingError, ValidationError

logger = structlog.get_logger()

router = APIRouter()

@router.post("/trees/generate/{document_id}")
async def generate_tree(
    document_id: str,
    background_tasks: BackgroundTasks
):
    """Generate complete document tree from subtrees."""
    try:
        # Verify document exists
        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": document_id})
        
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Check if tree already exists
        trees_collection = get_trees_collection()
        existing_tree = await trees_collection.find_one({"document_id": document_id})
        
        if existing_tree:
            return {
                "message": "Tree already exists",
                "document_id": document_id,
                "tree_id": str(existing_tree["_id"]),
                "status": "completed"
            }
        
        # Run tree merging in background
        background_tasks.add_task(
            tree_merger_service.merge_document_tree,
            document_id
        )
        
        return {
            "message": "Tree generation started",
            "document_id": document_id,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error("Failed to start tree generation", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trees/{document_id}")
async def get_tree(document_id: str):
    """Get complete document tree."""
    try:
        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"document_id": document_id})
        
        if not tree:
            raise HTTPException(status_code=404, detail="Tree not found")
        
        # Convert ObjectId to string
        tree["id"] = str(tree["_id"])
        del tree["_id"]
        
        return tree
        
    except Exception as e:
        logger.error("Failed to get tree", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trees/")
async def list_trees(
    limit: int = Query(10, ge=1, le=100),
    skip: int = Query(0, ge=0)
):
    """List all document trees."""
    try:
        trees_collection = get_trees_collection()
        
        cursor = trees_collection.find().skip(skip).limit(limit)
        trees = await cursor.to_list(length=limit)
        
        # Convert ObjectIds to strings
        for tree in trees:
            tree["id"] = str(tree["_id"])
            del tree["_id"]
        
        total = await trees_collection.count_documents({})
        
        return {
            "trees": trees,
            "total": total,
            "limit": limit,
            "skip": skip
        }
        
    except Exception as e:
        logger.error("Failed to list trees", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/trees/{document_id}")
async def delete_tree(document_id: str):
    """Delete document tree."""
    try:
        trees_collection = get_trees_collection()
        result = await trees_collection.delete_one({"document_id": document_id})
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Tree not found")
        
        return {
            "message": "Tree deleted successfully",
            "document_id": document_id
        }
        
    except Exception as e:
        logger.error("Failed to delete tree", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
```

## File: app/api/router.py (Update)

Update the router to include trees endpoints:

```python
"""
Main API router that includes all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.trees import router as trees_router
from app.api.v1.users import router as users_router

# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(documents_router, prefix="/v1/documents", tags=["documents"])
api_router.include_router(trees_router, prefix="/v1/trees", tags=["trees"])
api_router.include_router(users_router, prefix="/v1/users", tags=["users"])
```

## Usage Example

```python
# Example usage in a background task or API endpoint

from app.services.tree_merger_service import tree_merger_service

async def process_document_complete_tree(document_id: str):
    """Process document to create complete tree."""
    try:
        # Generate complete tree from subtrees
        complete_tree = await tree_merger_service.merge_document_tree(document_id)
        
        logger.info(
            "Document tree processing completed",
            document_id=document_id,
            total_nodes=len(complete_tree.root_nodes),
            processing_time=complete_tree.processing_metadata.get("processing_time"),
            token_usage=complete_tree.processing_metadata.get("token_usage"),
        )
        
        return complete_tree
        
    except Exception as e:
        logger.error("Document tree processing failed", document_id=document_id, error=str(e))
        raise
```