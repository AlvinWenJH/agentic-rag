"""
Topic tree management API endpoints.
Handles tree operations, JSON Patch manipulation, and tree queries.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import structlog
from datetime import datetime, timedelta
from bson import ObjectId
from bson.errors import InvalidId

from app.models.tree import (
    TreeUpdate,
    TreeResponse,
    TreeListResponse,
    TreeGenerationRequest,
    TreeGenerationResponse,
    TreePatchRequest,
    TreePatchResponse,
    TreeSearchRequest,
    TreeSearchResponse,
    TreeStats,
    TreeStatus,
)
from app.services.json_patch_service import tree_patch_engine
from app.services.gemini_service import gemini_service
from app.core.database import get_trees_collection, get_documents_collection
from app.core.exceptions import NotFoundError, ValidationError, ProcessingError


logger = structlog.get_logger()
router = APIRouter()


@router.post("/generate", response_model=TreeGenerationResponse)
async def generate_tree_from_document(request: TreeGenerationRequest):
    """
    Generate a topic tree from a document using visual analysis.

    Args:
        request: Tree generation request with document ID and options

    Returns:
        Generated tree response
    """
    try:
        logger.info(
            "Tree generation started",
            document_id=request.document_id,
        )

        # Get document
        from bson import ObjectId
        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": ObjectId(request.document_id)})

        if not document:
            raise NotFoundError(f"Document {request.document_id} not found")

        # Check if tree already exists for this document
        trees_collection = get_trees_collection()
        existing_tree = await trees_collection.find_one(
            {"document_id": request.document_id}
        )

        # Check if we should regenerate (default to False)
        regenerate = request.generation_options.get("regenerate", False)
        
        if existing_tree and not regenerate:
            # Return existing tree
            existing_tree["id"] = str(existing_tree["_id"])
            del existing_tree["_id"]

            return TreeGenerationResponse(
                tree_id=existing_tree["id"],
                document_id=request.document_id,
                status="completed",
                message="Tree already exists. Use regenerate=true to create a new one.",
                tree_data=existing_tree,
            )

        # Get document images
        image_paths = document.get("image_paths", [])
        
        if not image_paths:
            # For testing: create a basic tree structure without images
            logger.warning("Document has no processed images, creating basic tree structure", document_id=request.document_id)
            tree_analysis = {
                "tree_data": {
                    "title": document.get("title", "Unknown Document"),
                    "description": f"Basic tree structure for {document.get('title', 'document')}",
                    "nodes": [
                        {
                            "id": "root",
                            "title": "Document Overview",
                            "description": "Main content of the document",
                            "children": [
                                {
                                    "id": "section1",
                                    "title": "Section 1",
                                    "description": "First section of the document",
                                    "children": []
                                },
                                {
                                    "id": "section2", 
                                    "title": "Section 2",
                                    "description": "Second section of the document",
                                    "children": []
                                }
                            ]
                        }
                    ]
                },
                "processing_time": 0.1,
                "confidence_score": 0.5,
                "method": "fallback_basic_structure"
            }
        else:
            # Generate tree using Gemini
            tree_analysis = await gemini_service.analyze_document_images(
                image_paths=image_paths,
                document_id=request.document_id,
                document_title=document.get("title", ""),
            )

        # Create tree record
        now = datetime.utcnow()
        tree_data = {
            "document_id": request.document_id,
            "title": tree_analysis["tree_data"].get(
                "title", document.get("title", "Unknown Document")
            ),
            "description": tree_analysis["tree_data"].get("description", ""),
            "status": TreeStatus.completed,
            "tree_data": tree_analysis["tree_data"],
            "generation_metadata": tree_analysis,
            "created_at": now,
            "updated_at": now,
            "version": 1,
        }

        # Insert or update tree
        if existing_tree:
            # Update existing tree
            tree_data["version"] = existing_tree.get("version", 1) + 1
            await trees_collection.replace_one({"_id": existing_tree["_id"]}, tree_data)
            tree_id = str(existing_tree["_id"])
        else:
            # Insert new tree
            result = await trees_collection.insert_one(tree_data)
            tree_id = str(result.inserted_id)

        # Get final tree
        final_tree = await trees_collection.find_one({"_id": ObjectId(tree_id)})
        final_tree["id"] = str(final_tree["_id"])
        del final_tree["_id"]

        logger.info(
            "Tree generation completed",
            tree_id=tree_id,
            document_id=request.document_id,
            node_count=len(tree_analysis["tree_data"].get("nodes", [])),
        )

        return TreeGenerationResponse(
            tree_id=tree_id,
            document_id=request.document_id,
            status="completed",
            message="Tree generated successfully",
            tree_data=final_tree,
            processing_time=tree_analysis.get("processing_time", 0),
        )

    except (NotFoundError, ValidationError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Tree generation failed", document_id=request.document_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Tree generation failed")


@router.get("/stats", response_model=TreeStats)
async def get_tree_stats():
    """Get tree statistics."""
    try:
        trees_collection = get_trees_collection()

        # Get counts by status
        status_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        status_counts = {}
        async for result in trees_collection.aggregate(status_pipeline):
            status_counts[result["_id"]] = result["count"]

        # Get total trees count
        total_trees = await trees_collection.count_documents({})

        # Get average node count (using node_count field from Tree model)
        node_count_pipeline = [
            {"$match": {"node_count": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_nodes": {"$avg": "$node_count"}}}
        ]
        node_count_result = await trees_collection.aggregate(node_count_pipeline).to_list(1)
        average_node_count = node_count_result[0]["avg_nodes"] if node_count_result else 0.0

        # Get average depth (using max_depth field from Tree model)
        depth_pipeline = [
            {"$match": {"max_depth": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_depth": {"$avg": "$max_depth"}}}
        ]
        depth_result = await trees_collection.aggregate(depth_pipeline).to_list(1)
        average_depth = depth_result[0]["avg_depth"] if depth_result else 0.0

        # Get recent generations (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        recent_generations = await trees_collection.count_documents({
            "created_at": {"$gte": twenty_four_hours_ago}
        })

        # Get average generation time
        generation_time_pipeline = [
            {"$match": {"generation_time": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_time": {"$avg": "$generation_time"}}}
        ]
        generation_time_result = await trees_collection.aggregate(generation_time_pipeline).to_list(1)
        average_generation_time = generation_time_result[0]["avg_time"] if generation_time_result else None

        return TreeStats(
            total_trees=total_trees,
            trees_by_status=status_counts,
            average_node_count=average_node_count,
            average_depth=average_depth,
            average_generation_time=average_generation_time,
            recent_generations=recent_generations
        )

    except Exception as e:
        logger.error("Failed to get tree stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get tree statistics")


@router.get("/{tree_id}", response_model=TreeResponse)
async def get_tree(tree_id: str):
    """Get tree by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"_id": object_id})

        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        # Convert ObjectId to string
        tree["id"] = str(tree["_id"])
        del tree["_id"]

        return TreeResponse(**tree)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except Exception as e:
        logger.error("Failed to get tree", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve tree")


@router.get("/", response_model=TreeListResponse)
async def list_trees(
    document_id: Optional[str] = Query(None),
    status: Optional[TreeStatus] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List trees with optional filtering."""
    try:
        trees_collection = get_trees_collection()

        # Build filter
        filter_dict = {}
        if document_id:
            filter_dict["document_id"] = document_id
        if status:
            filter_dict["status"] = status

        # Get total count
        total = await trees_collection.count_documents(filter_dict)

        # Get trees
        cursor = (
            trees_collection.find(filter_dict)
            .skip(skip)
            .limit(limit)
            .sort("created_at", -1)
        )
        trees = await cursor.to_list(length=limit)

        # Convert ObjectIds to strings
        for tree in trees:
            tree["id"] = str(tree["_id"])
            del tree["_id"]

        # Calculate pagination
        page = (skip // limit) + 1
        pages = (total + limit - 1) // limit

        return TreeListResponse(
            trees=[TreeResponse(**tree) for tree in trees],
            total=total,
            page=page,
            size=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error("Failed to list trees", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list trees")


@router.patch("/{tree_id}", response_model=TreePatchResponse)
async def apply_tree_patches(tree_id: str, patch_request: TreePatchRequest):
    """
    Apply JSON Patch operations to a tree.

    Args:
        tree_id: Tree ID
        patch_request: Patch operations to apply

    Returns:
        Patch application result
    """
    try:
        logger.info(
            "Applying tree patches",
            tree_id=tree_id,
            operation_count=len(patch_request.operations),
        )

        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        # Get tree
        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"_id": object_id})

        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        # Apply patches
        modified_tree_data, operation_log = tree_patch_engine.apply_patches(
            tree_data=tree["tree_data"], patches=patch_request.operations
        )

        # Update tree in database
        update_data = {
            "tree_data": modified_tree_data,
            "updated_at": datetime.utcnow(),
            "version": tree.get("version", 1) + 1,
        }

        await trees_collection.update_one({"_id": object_id}, {"$set": update_data})

        # Get updated tree
        updated_tree = await trees_collection.find_one({"_id": object_id})
        updated_tree["id"] = str(updated_tree["_id"])
        del updated_tree["_id"]

        logger.info(
            "Tree patches applied successfully",
            tree_id=tree_id,
            operations_applied=len(
                [log for log in operation_log if not log.startswith("Patch")]
            ),
        )

        return TreePatchResponse(
            tree_id=tree_id,
            success=True,
            message="Patches applied successfully",
            operations_applied=len(patch_request.operations),
            operation_log=operation_log,
            updated_tree=updated_tree,
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except (ValidationError, ProcessingError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to apply tree patches", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to apply tree patches")


@router.post("/{tree_id}/search", response_model=TreeSearchResponse)
async def search_tree(tree_id: str, search_request: TreeSearchRequest):
    """
    Search within a tree structure.

    Args:
        tree_id: Tree ID
        search_request: Search parameters

    Returns:
        Search results
    """
    try:
        logger.info(
            "Tree search started",
            tree_id=tree_id,
            query=search_request.query[:50] + "..."
            if len(search_request.query) > 50
            else search_request.query,
        )

        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        # Get tree
        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"_id": object_id})

        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        tree_data = tree["tree_data"]
        nodes = tree_data.get("nodes", [])

        # Simple search implementation
        matching_nodes = []
        query_lower = search_request.query.lower()

        for node in nodes:
            score = 0.0

            # Search in title
            if query_lower in node.get("title", "").lower():
                score += 0.4

            # Search in description
            if query_lower in node.get("description", "").lower():
                score += 0.3

            # Search in content summary
            if query_lower in node.get("content_summary", "").lower():
                score += 0.2

            # Search in keywords
            keywords = node.get("keywords", [])
            for keyword in keywords:
                if query_lower in keyword.lower():
                    score += 0.1
                    break

            # Filter by node type if specified
            if search_request.node_types:
                if node.get("node_type") not in [
                    nt.value for nt in search_request.node_types
                ]:
                    continue

            # Filter by minimum level if specified
            if search_request.min_level is not None:
                if node.get("level", 0) < search_request.min_level:
                    continue

            # Filter by maximum level if specified
            if search_request.max_level is not None:
                if node.get("level", 0) > search_request.max_level:
                    continue

            if score > 0:
                matching_nodes.append(
                    {"node": node, "relevance_score": min(score, 1.0)}
                )

        # Sort by relevance score
        matching_nodes.sort(key=lambda x: x["relevance_score"], reverse=True)

        # Apply limit
        if search_request.limit:
            matching_nodes = matching_nodes[: search_request.limit]

        # Prepare results
        results = []
        for match in matching_nodes:
            node = match["node"]
            results.append(
                {
                    "node_id": node["id"],
                    "title": node["title"],
                    "description": node.get("description", ""),
                    "node_type": node["node_type"],
                    "level": node["level"],
                    "relevance_score": match["relevance_score"],
                    "page_numbers": node.get("page_numbers", []),
                    "parent_id": node.get("parent_id"),
                    "children_ids": node.get("children_ids", []),
                }
            )

        logger.info(
            "Tree search completed", tree_id=tree_id, results_found=len(results)
        )

        return TreeSearchResponse(
            tree_id=tree_id,
            query=search_request.query,
            total_results=len(results),
            results=results,
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except Exception as e:
        logger.error("Tree search failed", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Tree search failed")


@router.put("/{tree_id}", response_model=TreeResponse)
async def update_tree(tree_id: str, update_data: TreeUpdate):
    """Update tree metadata."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        trees_collection = get_trees_collection()

        # Check if tree exists
        existing_tree = await trees_collection.find_one({"_id": object_id})
        if not existing_tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        # Prepare update data
        update_dict = update_data.dict(exclude_unset=True)
        if update_dict:
            update_dict["updated_at"] = datetime.utcnow()

            # Update tree
            await trees_collection.update_one({"_id": object_id}, {"$set": update_dict})

        # Get updated tree
        updated_tree = await trees_collection.find_one({"_id": object_id})
        updated_tree["id"] = str(updated_tree["_id"])
        del updated_tree["_id"]

        return TreeResponse(**updated_tree)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except Exception as e:
        logger.error("Failed to update tree", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update tree")


@router.delete("/{tree_id}")
async def delete_tree(tree_id: str):
    """Delete tree."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        trees_collection = get_trees_collection()

        # Check if tree exists
        tree = await trees_collection.find_one({"_id": object_id})
        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        # Delete tree
        await trees_collection.delete_one({"_id": object_id})

        logger.info("Tree deleted", tree_id=tree_id)

        return {"message": "Tree deleted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except Exception as e:
        logger.error("Failed to delete tree", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete tree")


@router.get("/{tree_id}/nodes/{node_id}")
async def get_tree_node(tree_id: str, node_id: str):
    """Get specific node from tree."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"_id": object_id})

        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        # Find node
        nodes = tree["tree_data"].get("nodes", [])
        node = next((n for n in nodes if n["id"] == node_id), None)

        if not node:
            raise NotFoundError(f"Node {node_id} not found in tree {tree_id}")

        return {"tree_id": tree_id, "node": node}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree or node not found")
    except Exception as e:
        logger.error(
            "Failed to get tree node", tree_id=tree_id, node_id=node_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to get tree node")


@router.get("/{tree_id}/export")
async def export_tree(
    tree_id: str, format: str = Query("json", regex="^(json|xml|yaml)$")
):
    """Export tree in different formats."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(tree_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid tree ID format")

        trees_collection = get_trees_collection()
        tree = await trees_collection.find_one({"_id": object_id})

        if not tree:
            raise NotFoundError(f"Tree {tree_id} not found")

        tree_data = tree["tree_data"]

        if format == "json":
            return tree_data
        elif format == "xml":
            # Simple XML export (would need proper XML library)
            return {"message": "XML export not implemented yet"}
        elif format == "yaml":
            # Simple YAML export (would need PyYAML)
            return {"message": "YAML export not implemented yet"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Tree not found")
    except Exception as e:
        logger.error("Failed to export tree", tree_id=tree_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to export tree")
