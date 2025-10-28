"""
Tree merger service for combining page subtrees into a complete document tree.
Uses Pydantic AI to generate JSON Patch operations for merging similar nodes.
"""

from __future__ import annotations

import json
from typing import List, Dict, Any, Optional
import structlog
from datetime import datetime
from dataclasses import dataclass

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModelSettings

from pydantic_ai.models.google import GoogleModel
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ProcessingError, ValidationError
from app.core.database import get_subtrees_collection, get_tree_collection


@dataclass
class TreeMergerDeps:
    """Dependencies for tree merger agent containing current tree state and next subtree."""

    current_tree: Dict[str, Any]
    next_subtree: Dict[str, Any]


logger = structlog.get_logger()


class MergeOperation(BaseModel):
    """Single operation for combining or connecting nodes."""

    operation_type: str = Field(
        description="Type of operation: 'merge' for combining similar content, 'connect' for structural linking"
    )
    source_path: str = Field(
        description="JSON path to source node (e.g., '/0/children/1')"
    )
    target_path: str = Field(description="JSON path to target node")


class MergePatch(BaseModel):
    """Structured output for operations to integrate tree nodes."""

    merge_operations: List[MergeOperation] = Field(
        description="List of operations to merge or connect nodes"
    )


class NodeReference(BaseModel):
    """Reference to a node in the tree structure."""

    page_number: int
    path: str  # JSON path like "/0/children/1"
    title: str
    summary: str
    node_type: str  # "section", "subject", "topic"


class TreeMergerService:
    """Service for merging page subtrees into complete document trees using Pydantic AI."""

    def __init__(self):
        self.settings = get_settings()
        self.provider = None
        self.model = None
        self._initialize_gemini()

    def _initialize_gemini(self):
        """Initialize Gemini provider and model once."""
        try:
            # Create Google provider
            self.provider = GoogleProvider(api_key=self.settings.GEMINI_API_KEY)

            # Create Google model
            self.model = GoogleModel(
                model_name=self.settings.GEMINI_MODEL,
                provider=self.provider,
            )

            logger.info(
                "Gemini client initialized for tree merging",
                model=self.settings.GEMINI_MODEL,
            )

        except Exception as e:
            logger.error("Failed to initialize Gemini client", error=str(e))
            raise ExternalServiceError(f"Gemini initialization failed: {str(e)}")

    def _create_agent(self, system_prompt: str):
        """Create a new agent with the reused model and system prompt."""
        try:
            if not self.model:
                raise ExternalServiceError("Gemini model not initialized")

            return Agent(
                self.model,
                output_type=MergePatch,
                system_prompt=system_prompt,
            )

        except Exception as e:
            logger.error("Failed to create Gemini agent", error=str(e))
            raise ExternalServiceError(f"Agent creation failed: {str(e)}")

    async def merge_document_tree(self, document_id: str) -> Dict[str, Any]:
        """
        Merge all page subtrees for a document into a complete tree iteratively.

        Process:
        1. Start with empty tree {}
        2. First subtree becomes the initial tree (as-is)
        3. For each subsequent subtree:
           - Inject current tree state as context
           - Generate MergePatch to merge next subtree
           - Execute patch and update tree

        Args:
            document_id: MongoDB ObjectId of the document

        Returns:
            Complete merged document tree
        """
        try:
            logger.info("Starting iterative tree merge", document_id=document_id)

            # Fetch all subtrees for the document
            subtrees = await self._fetch_document_subtrees(document_id)
            if not subtrees:
                raise ValueError(f"No subtrees found for document {document_id}")

            total_token_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

            # Initialize with first subtree and save immediately
            if subtrees:
                first_subtree = subtrees[0].get("page_tree", [])
                if first_subtree:
                    merged_tree = first_subtree[0]  # Get the root node
                    # Add pages attribute to all nodes
                    self._add_pages_attribute_recursive(
                        merged_tree, subtrees[0].get("page_number", 1)
                    )

                    # Save initial tree to database immediately
                    await self._save_merged_tree(
                        document_id, merged_tree, total_token_usage
                    )

                    logger.info(
                        "Initialized and saved tree with first subtree",
                        document_id=document_id,
                        page=subtrees[0].get("page_number", 1),
                    )
                else:
                    raise ValueError("First subtree has no page_tree data")
            else:
                raise ValueError("No subtrees available for merging")

            # Iterative merging for remaining subtrees
            for i, subtree_doc in enumerate(subtrees[1:], 1):
                page_tree = subtree_doc.get("page_tree", [])
                if not page_tree:
                    logger.warning(f"Skipping subtree {i + 1} - no page_tree data")
                    continue

                subtree = page_tree[0]  # Get the root node
                page_number = subtree_doc.get("page_number", i + 1)

                # Add pages attribute to subtree nodes
                self._add_pages_attribute_recursive(subtree, page_number)

                try:
                    # Load current tree from database
                    current_tree = await self._load_current_tree(document_id)
                    if not current_tree:
                        logger.error(
                            f"Failed to load current tree for document {document_id}"
                        )
                        continue

                    logger.info(
                        "Merging subtree",
                        document_id=document_id,
                        page=page_number,
                        current_tree_size=len(str(current_tree)),
                    )

                    # Extract compact node summaries for efficient prompting
                    # Use hierarchical strategy for current tree (L1, L2, endmost L3)
                    current_tree_summaries = self._extract_hierarchical_nodes(
                        current_tree, "current"
                    )
                    # Extract only L1 nodes from new subtree for cost optimization
                    subtree_summaries = self._extract_l1_nodes_only(subtree, "new")

                    current_nodes_text = "\n---\n".join(current_tree_summaries)
                    new_nodes_text = "\n---\n".join(subtree_summaries)

                    system_prompt = f"""You are an expert document tree merger. Your task is to merge new content nodes into an existing document tree structure.

CURRENT TREE HIERARCHICAL NODES (strategic selection for efficient processing):
{current_nodes_text}

NEW SUBTREE L1 NODES TO MERGE (top-level sections only for cost optimization):
{new_nodes_text}

Note: The current tree shows a strategic selection of nodes based on hierarchical levels:
- All L1 nodes (top-level sections)
- All L2 nodes (immediate children of L1)
- Only endmost L3 nodes (deepest specific content)

The new subtree shows ONLY L1 nodes (root and direct children) for cost optimization. This limits the number of operations while still capturing the main structural elements that need integration.

IMPORTANT: You must generate operations ONLY for the L1 nodes provided in the new subtree. Each L1 node represents a major section that should be integrated as a complete unit.

Your goal is to intelligently integrate these L1 nodes into the existing tree by:
1. Identifying semantically similar content between new L1 nodes and existing hierarchical nodes
2. Each operation should target one of the provided L1 nodes as the source
3. When merging/connecting an L1 node, its entire subtree structure will be included automatically
4. Focus on high-level structural decisions rather than granular content merging

You have TWO types of operations available:

**MERGE Operation**: Use when L1 nodes have similar/related content that should be combined
- Combines semantically similar L1 nodes by merging their attributes
- Retains target node's title/summary, combines pages and children
- Best for duplicate or overlapping L1 sections

**CONNECT Operation**: Use when L1 nodes are different but should be structurally linked
- Adds source L1 node as a new child to target node without merging content
- Preserves both nodes' distinct identities
- Best for organizing different but related L1 topics hierarchically

Generate operations with these fields:
- operation_type: "merge" or "connect"
- source_path: JSON path to L1 source node in the new subtree (e.g., "new", "new/child_0", "new/child_1")
- target_path: JSON path to target location in current tree (e.g., "current/child_1")

CONSTRAINT: Only generate operations for the L1 nodes shown in the NEW SUBTREE section above. Each operation will process the entire subtree under that L1 node."""

                    agent = self._create_agent(system_prompt)
                    print(
                        f"Merging next tree ({len(subtree_summaries)}) to current tree ({len(current_tree_summaries)})",
                        flush=True,
                    )
                    # Generate merge patch using AI with context
                    result = await agent.run(
                        "Merge the provided subtree into the current tree structure.",
                        model_settings=GoogleModelSettings(
                            google_thinking_config={"thinkingBudget": 512}
                        ),
                    )
                    print("MERGE PATCH", flush=True)
                    print(result.output, flush=True)
                    # Track token usage if available
                    usage = result.usage()
                    print(f"Usage: {usage}", flush=True)
                    total_token_usage["input_tokens"] += getattr(
                        usage, "input_tokens", 0
                    )
                    total_token_usage["output_tokens"] += getattr(
                        usage, "output_tokens", 0
                    )
                    total_token_usage["total_tokens"] += getattr(
                        usage, "total_tokens", 0
                    )

                    # Execute the merge patch
                    updated_tree = await self._execute_merge_patch(
                        current_tree, subtree, result.output
                    )

                    # Save updated tree to database
                    await self._save_merged_tree(
                        document_id, updated_tree, total_token_usage
                    )

                    logger.info(
                        "Subtree merged and saved successfully",
                        document_id=document_id,
                        page=page_number,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to merge subtree",
                        document_id=document_id,
                        page=page_number,
                        error=str(e),
                    )
                    # Continue with next subtree instead of failing completely
                    continue

            logger.info(
                "Iterative tree merge completed",
                document_id=document_id,
                total_subtrees=len(subtrees),
                total_token_usage=total_token_usage,
            )

            # Return the final merged tree
            final_tree = await self._load_current_tree(document_id)
            return final_tree

        except Exception as e:
            logger.error("Tree merge failed", document_id=document_id, error=str(e))
            raise

    async def _execute_merge_patch(
        self,
        current_tree: Dict[str, Any],
        subtree: Dict[str, Any],
        merge_patch: MergePatch,
    ) -> Dict[str, Any]:
        """
        Execute a single merge patch to combine subtree into current tree.

        Args:
            current_tree: Current state of the merged tree
            subtree: Subtree to be merged
            merge_patch: AI-generated merge operations

        Returns:
            Updated tree after applying merge operations
        """
        try:
            # Work with a copy to avoid modifying the original
            result_tree = json.loads(json.dumps(current_tree))

            for operation in merge_patch.merge_operations:
                # Skip connect operations where source and target start with same path prefix
                if operation.operation_type == "connect":
                    source_prefix = (
                        operation.source_path.split("/")[0]
                        if "/" in operation.source_path
                        else operation.source_path
                    )
                    target_prefix = (
                        operation.target_path.split("/")[0]
                        if "/" in operation.target_path
                        else operation.target_path
                    )

                    if source_prefix == target_prefix:
                        logger.info(
                            f"Skipping connect operation with same path prefix: "
                            f"source='{operation.source_path}', target='{operation.target_path}'"
                        )
                        continue

                result_tree = self._apply_merge_operation(
                    result_tree, subtree, operation
                )

            # Recalculate node types after merging to ensure proper L1/L2/L3 classification
            self._recalculate_node_types(result_tree)

            return result_tree

        except Exception as e:
            logger.error("Failed to execute merge patch", error=str(e))
            # Return current tree if patch execution fails
            return current_tree

    async def _fetch_document_subtrees(self, document_id: str) -> List[Dict[str, Any]]:
        """Fetch all subtrees for a document from MongoDB and add node type classification."""
        try:
            subtrees_collection = get_subtrees_collection()

            cursor = subtrees_collection.find({"document_id": document_id}).sort(
                "page_number", 1
            )

            subtrees = await cursor.to_list(length=None)

            # Add node type classification to each subtree
            for subtree in subtrees:
                if "tree" in subtree:
                    self._add_node_types_recursive(subtree["tree"], level=1)

            logger.debug(
                "Fetched subtrees from MongoDB with node types",
                document_id=document_id,
                count=len(subtrees),
            )

            return subtrees

        except Exception as e:
            logger.error(
                "Failed to fetch subtrees",
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to fetch subtrees: {str(e)}")

    def _add_node_types_recursive(self, node: Dict[str, Any], level: int) -> None:
        """
        Add node_type attribute to nodes based on their hierarchical level.

        L1: Top-level nodes (level 1)
        L2: Direct children of L1 (level 2)
        L3: All deeper levels (level 3 and below)
        """
        if not isinstance(node, dict):
            return

        # Assign node type based on level
        if level == 1:
            node["node_type"] = "L1"
        elif level == 2:
            node["node_type"] = "L2"
        else:
            node["node_type"] = "L3"

        # Recursively process children
        if "children" in node and isinstance(node["children"], list):
            for child in node["children"]:
                self._add_node_types_recursive(child, level + 1)

    def _recalculate_node_types(self, tree: Dict[str, Any]) -> None:
        """
        Recalculate node types for the entire tree after merging operations.
        This ensures that L1 nodes that become children of other L1 nodes are properly reclassified as L2, etc.
        """
        if not isinstance(tree, dict):
            return

        # If tree has children, it's the root container
        if "children" in tree and isinstance(tree["children"], list):
            for child in tree["children"]:
                self._add_node_types_recursive(child, level=1)
        else:
            # If tree is a single node, start from level 1
            self._add_node_types_recursive(tree, level=1)

    def _add_pages_attribute_recursive(
        self, node: Dict[str, Any], page_number: int
    ) -> None:
        """Recursively add pages attribute to all nodes in the tree."""
        if isinstance(node, dict):
            # Add or update pages attribute
            if "pages" in node:
                if isinstance(node["pages"], list):
                    if page_number not in node["pages"]:
                        node["pages"].append(page_number)
                        node["pages"].sort()
                else:
                    node["pages"] = [page_number]
            else:
                node["pages"] = [page_number]

            # Recursively process children
            if "children" in node and isinstance(node["children"], list):
                for child in node["children"]:
                    self._add_pages_attribute_recursive(child, page_number)

    def _extract_hierarchical_nodes(
        self, tree: Dict[str, Any], path_prefix: str = ""
    ) -> List[str]:
        """
        Extract nodes based on hierarchical strategy:
        - All L1 nodes (top-level)
        - All L2 nodes (direct children of L1)
        - Only endmost L3 nodes (leaf nodes from deeper levels)

        This provides complete high-level structure while keeping deeper content focused.
        """
        summaries = []
        included_paths = set()

        def traverse_node(node: Dict[str, Any], current_path: str) -> None:
            if not isinstance(node, dict):
                return

            node_type = node.get("node_type", "")

            # Check if this is a leaf node
            has_children = (
                "children" in node
                and isinstance(node["children"], list)
                and len(node["children"]) > 0
            )

            # Include based on node type and leaf status
            if node_type in ["L1", "L2"]:
                # Include all L1 and L2 nodes
                self._add_node_summary(node, current_path, summaries, included_paths)
            elif node_type == "L3" and not has_children:
                # Include only endmost (leaf) L3 nodes
                self._add_node_summary(node, current_path, summaries, included_paths)

            # Continue traversing children
            if has_children:
                for i, child in enumerate(node["children"]):
                    child_path = (
                        f"{current_path}/child_{i}" if current_path else f"child_{i}"
                    )
                    traverse_node(child, child_path)

        # Start traversal from root
        root_path = path_prefix if path_prefix else "root"
        traverse_node(tree, root_path)

        return summaries

    def _extract_node_summaries(
        self, tree: Dict[str, Any], path_prefix: str = "", endmost_only: bool = True
    ) -> List[str]:
        """
        Extract compact node summaries from tree structure for efficient prompting.

        Args:
            tree: The tree structure to extract from
            path_prefix: Prefix for the path (e.g., "current", "new")
            endmost_only: If True, use hierarchical strategy (L1, L2, endmost L3)

        Returns a list of formatted strings with path, title, and summary.
        Format: "path: /section/subsection\ntitle: Node Title\nsummary: Brief description"
        """
        if endmost_only:
            return self._extract_hierarchical_nodes(tree, path_prefix)

        # Fallback to full tree extraction if needed
        summaries = []

        def traverse_node(node: Dict[str, Any], current_path: str) -> None:
            if not isinstance(node, dict):
                return

            # Extract node information
            title = node.get("title", node.get("text", "Untitled"))
            summary = node.get("summary", node.get("content", ""))

            # Truncate summary if too long (keep it concise)
            if len(summary) > 150:
                summary = summary[:147] + "..."

            # Format the node summary
            node_summary = f"path: {current_path}\ntitle: {title}\nsummary: {summary}"
            summaries.append(node_summary)

            # Recursively process children
            if "children" in node and isinstance(node["children"], list):
                for i, child in enumerate(node["children"]):
                    child_path = (
                        f"{current_path}/child_{i}" if current_path else f"child_{i}"
                    )
                    traverse_node(child, child_path)

        # Start traversal from root
        root_path = path_prefix if path_prefix else "root"
        traverse_node(tree, root_path)

        return summaries

    def _extract_l1_nodes_only(
        self, tree: Dict[str, Any], path_prefix: str = ""
    ) -> List[str]:
        """
        Extract only L1 (top-level) nodes from the subtree for cost optimization.
        This limits AI operations to only the main sections of the new subtree.

        Args:
            tree: The tree structure to extract from
            path_prefix: Prefix for the path (e.g., "new")

        Returns:
            List of formatted L1 node summaries
        """
        summaries = []

        # Extract root node (L1)
        if isinstance(tree, dict):
            title = tree.get("title", tree.get("text", "Untitled"))
            summary = tree.get("summary", tree.get("content", ""))

            # Truncate summary if too long
            if len(summary) > 150:
                summary = summary[:147] + "..."

            root_path = path_prefix if path_prefix else "root"
            node_summary = f"path: {root_path}\ntitle: {title}\nsummary: {summary}"
            summaries.append(node_summary)

            # Extract direct children (also L1 level in the context of operations)
            if "children" in tree and isinstance(tree["children"], list):
                for i, child in enumerate(tree["children"]):
                    if isinstance(child, dict):
                        child_title = child.get("title", child.get("text", "Untitled"))
                        child_summary = child.get("summary", child.get("content", ""))

                        # Truncate summary if too long
                        if len(child_summary) > 150:
                            child_summary = child_summary[:147] + "..."

                        child_path = (
                            f"{path_prefix}/child_{i}" if path_prefix else f"child_{i}"
                        )
                        child_node_summary = f"path: {child_path}\ntitle: {child_title}\nsummary: {child_summary}"
                        summaries.append(child_node_summary)

        return summaries

    def _add_node_summary(
        self, node: Dict[str, Any], path: str, summaries: List[str], included_paths: set
    ) -> None:
        """Helper method to add a node summary if not already included."""
        if path not in included_paths:
            title = node.get("title", node.get("text", "Untitled"))
            summary = node.get("summary", node.get("content", ""))
            node_type = node.get("node_type", "")

            # Truncate summary if too long
            if len(summary) > 200:
                summary = summary[:197] + "..."

            # Format the node summary with type information
            node_summary = (
                f"path: {path}\ntype: {node_type}\ntitle: {title}\nsummary: {summary}"
            )
            summaries.append(node_summary)
            included_paths.add(path)

    def _add_page_references_recursive(self, node: Dict[str, Any], page_number: int):
        """Add page references to all nodes recursively."""
        node["source_pages"] = [page_number]

        for child in node.get("children", []):
            self._add_page_references_recursive(child, page_number)

    def _apply_merge_operation(
        self, tree: Dict[str, Any], subtree: Dict[str, Any], operation: MergeOperation
    ) -> Dict[str, Any]:
        """Apply a single operation (merge or connect) to the tree."""
        if operation.operation_type == "merge":
            return self._merge_nodes(
                tree, subtree, operation.source_path, operation.target_path
            )
        elif operation.operation_type == "connect":
            return self._connect_nodes(
                tree, subtree, operation.source_path, operation.target_path
            )
        else:
            logger.warning(f"Unknown operation type: {operation.operation_type}")
            return tree

    def _merge_nodes(
        self,
        tree: Dict[str, Any],
        subtree: Dict[str, Any],
        source_path: str,
        target_path: str,
    ) -> Dict[str, Any]:
        """Merge nodes from subtree into tree at specified paths using compact path format."""
        try:
            # Get source node from subtree using compact path format
            source_node = self._get_node_by_compact_path(subtree, source_path)
            if not source_node:
                logger.warning(f"Source node not found at path: {source_path}")
                return tree

            # Get target node from tree using compact path format
            target_node = self._get_node_by_compact_path(tree, target_path)
            if not target_node:
                logger.warning(f"Target node not found at path: {target_path}")
                return tree

            logger.info(
                f"Merging node from {source_path} to {target_path}",
                source_title=source_node.get("title", "Unknown"),
                target_title=target_node.get("title", "Unknown"),
            )

            # Merge attributes
            self._merge_node_attributes(target_node, source_node)

            return tree

        except Exception as e:
            logger.error(f"Error merging nodes: {e}")
            return tree

    def _merge_node_attributes(
        self, target_node: Dict[str, Any], source_node: Dict[str, Any]
    ) -> None:
        """Merge attributes from source node into target node, including pages."""
        # Merge pages attribute
        if "pages" in source_node:
            if "pages" not in target_node:
                target_node["pages"] = source_node["pages"].copy()
            else:
                # Combine pages and remove duplicates
                combined_pages = list(set(target_node["pages"] + source_node["pages"]))
                target_node["pages"] = sorted(combined_pages)

        # Merge other attributes (title, summary, etc.)
        for key, value in source_node.items():
            if key == "children":
                # Handle children merging separately
                if "children" not in target_node:
                    target_node["children"] = (
                        value.copy() if isinstance(value, list) else value
                    )
                elif isinstance(target_node["children"], list) and isinstance(
                    value, list
                ):
                    target_node["children"].extend(value)
            elif key not in target_node:
                target_node[key] = value
            # For existing attributes like title/summary, keep the target's version
            # unless we want to implement more sophisticated merging logic

    def _connect_nodes(
        self,
        tree: Dict[str, Any],
        subtree: Dict[str, Any],
        source_path: str,
        target_path: str,
    ) -> Dict[str, Any]:
        """Connect nodes by adding source node as child to target node without merging content."""
        try:
            # Get source node from subtree using compact path format
            source_node = self._get_node_by_compact_path(subtree, source_path)
            if not source_node:
                logger.warning(f"Source node not found at path: {source_path}")
                return tree

            # Get target node from tree using compact path format
            target_node = self._get_node_by_compact_path(tree, target_path)
            if not target_node:
                logger.warning(f"Target node not found at path: {target_path}")
                return tree

            logger.info(
                f"Connecting node from {source_path} to {target_path}",
                source_title=source_node.get("title", "Unknown"),
                target_title=target_node.get("title", "Unknown"),
            )

            # Create a copy of the source node to avoid modifying the original
            connected_node = source_node.copy()

            # Initialize children list if it doesn't exist
            if "children" not in target_node:
                target_node["children"] = []

            # Add the source node as a child to the target node
            target_node["children"].append(connected_node)

            return tree

        except Exception as e:
            logger.error(f"Error connecting nodes: {e}")
            return tree

    def _get_node_by_path(
        self, tree: List[Dict[str, Any]], path: str
    ) -> Optional[Dict[str, Any]]:
        """Get a node by its JSON path."""
        try:
            path_parts = [int(p) for p in path.strip("/").split("/") if p.isdigit()]

            if not path_parts:
                return None

            current = tree[path_parts[0]]

            for part in path_parts[1:]:
                if part % 2 == 1:  # Odd indices are "children"
                    current = current.get("children", [])[part // 2]
                else:  # Even indices are direct children
                    current = current.get("children", [])[part]

            return current

        except (IndexError, KeyError, ValueError):
            return None

    def _get_node_by_compact_path(
        self, tree: Dict[str, Any], path: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a node by its compact path format (e.g., 'current/child_0', 'new/child_1').

        Args:
            tree: The tree structure (single root node)
            path: Compact path like 'current/child_0' or 'new/child_1'

        Returns:
            The node at the specified path, or None if not found
        """
        try:
            # Split path into parts
            path_parts = path.strip("/").split("/")

            # Skip the prefix (current/new) and start from the tree root
            if len(path_parts) <= 1:
                return tree  # Return root if only prefix

            current = tree

            # Navigate through the path
            for part in path_parts[1:]:  # Skip the prefix
                if part.startswith("child_"):
                    # Extract child index
                    child_index = int(part.split("_")[1])

                    # Navigate to the child
                    if "children" in current and isinstance(current["children"], list):
                        if child_index < len(current["children"]):
                            current = current["children"][child_index]
                        else:
                            return None
                    else:
                        return None
                else:
                    # Handle other path components if needed
                    return None

            return current

        except (IndexError, KeyError, ValueError, AttributeError) as e:
            logger.warning(f"Failed to resolve compact path '{path}': {e}")
            return None

    def _remove_node_by_path(self, tree: List[Dict[str, Any]], path: str):
        """Remove a node by its JSON path."""
        try:
            path_parts = [int(p) for p in path.strip("/").split("/") if p.isdigit()]

            if len(path_parts) == 1:
                # Remove from root level
                if path_parts[0] < len(tree):
                    tree.pop(path_parts[0])
            else:
                # Navigate to parent and remove child
                parent = tree[path_parts[0]]
                for part in path_parts[1:-1]:
                    if part % 2 == 1:
                        parent = parent.get("children", [])[part // 2]
                    else:
                        parent = parent.get("children", [])[part]

                # Remove the final child
                final_index = path_parts[-1]
                if "children" in parent and final_index < len(parent["children"]):
                    parent["children"].pop(final_index)

        except (IndexError, KeyError, ValueError):
            logger.warning(f"Failed to remove node at path: {path}")

    def _calculate_total_token_usage(
        self, merge_patches: List[MergePatch]
    ) -> Dict[str, int]:
        """Calculate total token usage across all merge patches."""
        total_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "total_calls": 0,
        }

        for patch in merge_patches:
            if hasattr(patch, "token_usage"):
                usage = patch.token_usage
                total_usage["input_tokens"] += usage.get("input_tokens", 0)
                total_usage["output_tokens"] += usage.get("output_tokens", 0)
                total_usage["total_tokens"] += usage.get("total_tokens", 0)
                total_usage["total_calls"] += usage.get("total_calls", 0)

        return total_usage

    async def _load_current_tree(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Load the current merged tree from the database and add node type classification."""
        try:
            tree_collection = get_tree_collection()

            tree_doc = await tree_collection.find_one({"document_id": document_id})
            if tree_doc and "merged_tree" in tree_doc:
                current_tree = tree_doc["merged_tree"]

                # Add node type classification to the current tree
                self._add_node_types_recursive(current_tree, level=1)

                logger.info(
                    "Current tree loaded successfully with node types",
                    document_id=document_id,
                )
                return current_tree
            else:
                logger.warning(
                    "No current tree found in database",
                    document_id=document_id,
                )
                return None

        except Exception as e:
            logger.error(f"Failed to load current tree: {e}")
            return None

    async def _save_merged_tree(
        self,
        document_id: str,
        merged_tree: Dict[str, Any],
        total_token_usage: Dict[str, int],
    ) -> None:
        """Save the merged tree to the database."""
        try:
            tree_collection = get_tree_collection()

            tree_document = {
                "document_id": document_id,
                "merged_tree": merged_tree,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                "token_usage": total_token_usage,
                "merge_method": "ai_guided_iterative",
            }

            # Upsert the merged tree
            await tree_collection.replace_one(
                {"document_id": document_id}, tree_document, upsert=True
            )

            logger.info(
                "Merged tree saved successfully",
                document_id=document_id,
                token_usage=total_token_usage,
            )

        except Exception as e:
            logger.error(f"Failed to save merged tree: {e}")
            raise ProcessingError(f"Failed to save merged tree: {str(e)}")

    async def cleanup(self):
        """Clean up resources and close connections."""
        try:
            # Clean up the provider and model references
            if hasattr(self, "provider") and self.provider:
                # Try to close the provider if it has a close method
                if hasattr(self.provider, "aclose"):
                    try:
                        await self.provider.aclose()
                    except (AttributeError, Exception) as e:
                        # Ignore the _async_httpx_client error and other cleanup errors
                        logger.debug(
                            "Tree merger provider cleanup completed with minor issues",
                            error=str(e),
                        )

                # Clear the provider reference
                self.provider = None

            if hasattr(self, "model"):
                self.model = None

            logger.info("Tree merger service cleanup completed successfully")

        except Exception as e:
            # Log but don't raise - we don't want cleanup errors to crash the shutdown
            logger.warning(
                "Tree merger service cleanup encountered issues", error=str(e)
            )


# Service instance
tree_merger_service = TreeMergerService()
