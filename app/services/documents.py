"""
Document service module for tree operations and document management.
Contains business logic for document tree operations that can be reused across different components.
"""

from typing import Dict, Any, Optional, List, Union
import structlog
from app.core.database import get_tree_collection
from app.core.exceptions import NotFoundError

logger = structlog.get_logger()


def count_nodes_recursive(node: dict) -> dict:
    """
    Recursively count nodes in a tree structure.

    Args:
        node: A tree node dictionary

    Returns:
        Dictionary with counts for total nodes and nodes by type
    """
    counts = {
        "total": 1,  # Count current node
        "L1": 0,
        "L2": 0,
        "L3": 0,
    }

    # Count current node by type
    node_type = node.get("node_type")
    if node_type in ["L1", "L2", "L3"]:
        counts[node_type] = 1

    # Recursively count children
    children = node.get("children", [])
    for child in children:
        child_counts = count_nodes_recursive(child)
        counts["total"] += child_counts["total"]
        counts["L1"] += child_counts["L1"]
        counts["L2"] += child_counts["L2"]
        counts["L3"] += child_counts["L3"]

    return counts


def navigate_to_path(tree: dict, path: str) -> dict:
    """
    Navigate to a specific path in the tree structure.

    Args:
        tree: The root tree node
        path: Path string like "/children/0/children/1" where "/" is root

    Returns:
        The node at the specified path

    Raises:
        ValueError: If path is invalid or node not found
    """
    if path == "/" or path == "":
        return tree

    # Remove leading slash and split path
    path_parts = path.strip("/").split("/")
    current_node = tree

    for i in range(0, len(path_parts), 2):
        if i + 1 >= len(path_parts):
            raise ValueError(f"Invalid path format: {path}")

        # Expect "children" followed by index
        if path_parts[i] != "children":
            raise ValueError(
                f"Invalid path segment: {path_parts[i]}. Expected 'children'"
            )

        try:
            index = int(path_parts[i + 1])
        except ValueError:
            raise ValueError(f"Invalid index: {path_parts[i + 1]}. Must be an integer")

        children = current_node.get("children", [])
        if index < 0 or index >= len(children):
            raise ValueError(
                f"Index {index} out of range. Node has {len(children)} children"
            )

        current_node = children[index]

    return current_node


def serialize_tree_to_list(
    node: dict, current_path: str = "/", result: List[Dict[str, str]] = None
) -> List[Dict[str, str]]:
    """
    Serialize a tree node and its children into a flat list of dictionaries.

    Args:
        node: The tree node to serialize
        current_path: The current path in the tree (e.g., "/children/0")
        result: The accumulating result list

    Returns:
        List of dictionaries with "path", "title", and "summary" keys
    """
    if result is None:
        result = []

    # Add current node to result
    result.append(
        {
            "path": current_path,
            "title": node.get("title", ""),
            "summary": node.get("content_summary", node.get("summary", "")),
            "pages": node.get("pages", []),
        }
    )

    # Process children if they exist
    children = node.get("children", [])
    if isinstance(children, list):
        for i, child in enumerate(children):
            if isinstance(child, dict):
                child_path = f"{current_path.rstrip('/')}/children/{i}"
                serialize_tree_to_list(child, child_path, result)

    return result


def limit_tree_depth(node: dict, max_depth: int, current_depth: int = 0) -> dict:
    """
    Limit the depth of a tree structure.

    Args:
        node: The tree node to limit
        max_depth: Maximum depth to include
        current_depth: Current depth (used for recursion)

    Returns:
        Tree node with limited depth
    """
    # Create a copy of the node without children first
    limited_node = {k: v for k, v in node.items() if k != "children"}

    # If we haven't reached max depth and node has children, include them
    if current_depth < max_depth and "children" in node:
        limited_children = []
        for child in node["children"]:
            limited_child = limit_tree_depth(child, max_depth, current_depth + 1)
            limited_children.append(limited_child)
        limited_node["children"] = limited_children

    return limited_node


async def get_document_tree_data(document_id: str) -> Dict[str, Any]:
    """
    Get the complete document tree with node counts.

    Args:
        document_id: Document ID to get tree for

    Returns:
        Dictionary containing document_id, node_counts, and tree_data

    Raises:
        NotFoundError: If tree not found for document
    """
    try:
        # Get tree from tree collection
        tree_collection = get_tree_collection()

        tree_doc = await tree_collection.find_one(
            {"document_id": document_id}, {"_id": 0}
        )

        if not tree_doc:
            raise NotFoundError(f"Tree not found for document {document_id}")

        tree_data = tree_doc["merged_tree"]
        node_counts = count_nodes_recursive(tree_data)

        return {
            "document_id": document_id,
            "node_counts": {
                "total": node_counts["total"],
                "L1": node_counts["L1"],
                "L2": node_counts["L2"],
                "L3": node_counts["L3"],
            },
            "tree_data": tree_data,
        }
    except Exception as e:
        logger.error(
            "Failed to get document tree data", document_id=document_id, error=str(e)
        )
        raise


async def get_document_tree_stats(document_id: str) -> Dict[str, Any]:
    """
    Get statistics about the document tree including node counts.

    Args:
        document_id: Document ID to get tree statistics for

    Returns:
        Dictionary containing document_id and node_counts

    Raises:
        NotFoundError: If tree not found for document
    """
    try:
        # Get tree from tree collection
        tree_collection = get_tree_collection()

        tree_doc = await tree_collection.find_one(
            {"document_id": document_id}, {"_id": 0}
        )

        if not tree_doc:
            raise NotFoundError(f"Tree not found for document {document_id}")

        # Count nodes in the tree
        tree_data = tree_doc["merged_tree"]
        node_counts = count_nodes_recursive(tree_data)

        return {
            "document_id": document_id,
            "node_counts": {
                "total_nodes": node_counts["total"],
                "L1_nodes": node_counts["L1"],
                "L2_nodes": node_counts["L2"],
                "L3_nodes": node_counts["L3"],
            },
        }
    except Exception as e:
        logger.error(
            "Failed to get document tree statistics",
            document_id=document_id,
            error=str(e),
        )
        raise


async def get_document_tree_from_path(
    document_id: str,
    path: str = "/",
    depth: int = 3,
    serialize: bool = False,
) -> Union[Dict[str, Any], List[Dict[str, str]]]:
    """
    Get a subtree starting from a specific path with depth control.

    Args:
        document_id: Document ID to get tree for
        path: Path to the subtree (e.g., "/children/0/children/1" where "/" is root)
        depth: Maximum depth of the subtree to return
        serialize: If True, return a list of dictionaries with "path", "title", and "summary"

    Returns:
        If serialize=False: Dictionary containing document_id, path, depth, and subtree
        If serialize=True: List of dictionaries with "path", "title", and "summary" keys

    Raises:
        NotFoundError: If tree not found for document
        ValueError: If path is invalid
    """
    try:
        # Get tree from tree collection
        tree_collection = get_tree_collection()

        tree_doc = await tree_collection.find_one(
            {"document_id": document_id}, {"_id": 0}
        )

        if not tree_doc:
            raise NotFoundError(f"Tree not found for document {document_id}")

        # Navigate to the specified path
        subtree_root = navigate_to_path(tree_doc["merged_tree"], path)

        # Limit the depth of the subtree
        limited_subtree = limit_tree_depth(subtree_root, depth)

        # Return serialized list if requested
        if serialize:
            return serialize_tree_to_list(limited_subtree, path)

        return {
            "document_id": document_id,
            "path": path,
            "depth": depth,
            "subtree": limited_subtree,
        }
    except Exception as e:
        logger.error(
            "Failed to get document tree from path",
            document_id=document_id,
            path=path,
            depth=depth,
            error=str(e),
        )
        raise
