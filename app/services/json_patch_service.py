"""
JSON Patch service for hierarchical topic tree manipulation.
Provides operations for modifying tree structures using JSON Patch operations.
"""

from typing import List, Dict, Any, Tuple

from copy import deepcopy
import structlog
import jsonschema
import jsonpatch

from app.core.exceptions import ValidationError, ProcessingError
from app.models.tree import NodeType, TreePatchOperation


logger = structlog.get_logger()


class TreePatchEngine:
    """JSON Patch engine for topic tree manipulation."""

    def __init__(self):
        self.tree_schema = self._create_tree_schema()
        self.node_schema = self._create_node_schema()

    def _create_tree_schema(self) -> Dict[str, Any]:
        """Create JSON schema for tree validation."""
        return {
            "type": "object",
            "required": ["title", "nodes"],
            "properties": {
                "title": {"type": "string"},
                "description": {"type": "string"},
                "root_node_id": {"type": "string"},
                "nodes": {"type": "array", "items": {"$ref": "#/definitions/node"}},
                "generation_metadata": {"type": "object"},
            },
            "definitions": {
                "node": {
                    "type": "object",
                    "required": ["id", "title", "node_type", "level"],
                    "properties": {
                        "id": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "node_type": {
                            "type": "string",
                            "enum": [t.value for t in NodeType],
                        },
                        "level": {"type": "integer", "minimum": 0},
                        "page_numbers": {"type": "array", "items": {"type": "integer"}},
                        "parent_id": {"type": ["string", "null"]},
                        "children_ids": {"type": "array", "items": {"type": "string"}},
                        "content_summary": {"type": "string"},
                        "confidence_score": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "keywords": {"type": "array", "items": {"type": "string"}},
                    },
                }
            },
        }

    def _create_node_schema(self) -> Dict[str, Any]:
        """Create JSON schema for individual node validation."""
        return self.tree_schema["definitions"]["node"]

    def apply_patches(
        self, tree_data: Dict[str, Any], patches: List[TreePatchOperation]
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Apply JSON Patch operations to a tree.

        Args:
            tree_data: Original tree data
            patches: List of patch operations

        Returns:
            Tuple of (modified_tree, operation_log)
        """
        try:
            logger.info(
                "Applying tree patches",
                patch_count=len(patches),
                tree_title=tree_data.get("title", "Unknown"),
            )

            # Validate original tree
            self.validate_tree(tree_data)

            # Create working copy
            modified_tree = deepcopy(tree_data)
            operation_log = []

            # Apply each patch operation
            for i, patch_op in enumerate(patches):
                try:
                    modified_tree, log_entry = self._apply_single_patch(
                        modified_tree, patch_op, i + 1
                    )
                    operation_log.append(log_entry)

                except Exception as e:
                    error_msg = f"Patch {i + 1} failed: {str(e)}"
                    logger.error(
                        "Patch operation failed",
                        patch_index=i + 1,
                        operation=patch_op.op,
                        path=patch_op.path,
                        error=str(e),
                    )
                    operation_log.append(error_msg)
                    # Continue with other patches or fail based on configuration
                    if patch_op.op in ["remove", "replace"]:
                        # Critical operations should fail the entire process
                        raise ProcessingError(error_msg)

            # Validate modified tree
            self.validate_tree(modified_tree)

            # Update tree integrity
            modified_tree = self._ensure_tree_integrity(modified_tree)

            logger.info(
                "Tree patches applied successfully",
                operations_applied=len(
                    [log for log in operation_log if not log.startswith("Patch")]
                ),
                final_node_count=len(modified_tree.get("nodes", [])),
            )

            return modified_tree, operation_log

        except Exception as e:
            logger.error("Failed to apply tree patches", error=str(e))
            raise ProcessingError(f"Tree patch application failed: {str(e)}")

    def _apply_single_patch(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply a single patch operation."""

        operation_type = patch_op.op.lower()

        if operation_type == "add":
            return self._apply_add_operation(tree_data, patch_op, operation_index)
        elif operation_type == "remove":
            return self._apply_remove_operation(tree_data, patch_op, operation_index)
        elif operation_type == "replace":
            return self._apply_replace_operation(tree_data, patch_op, operation_index)
        elif operation_type == "move":
            return self._apply_move_operation(tree_data, patch_op, operation_index)
        elif operation_type == "copy":
            return self._apply_copy_operation(tree_data, patch_op, operation_index)
        elif operation_type == "test":
            return self._apply_test_operation(tree_data, patch_op, operation_index)
        else:
            raise ValidationError(f"Unsupported patch operation: {operation_type}")

    def _apply_add_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply add operation."""

        path_parts = patch_op.path.strip("/").split("/")

        if path_parts[0] == "nodes":
            if len(path_parts) == 1:
                # Add new node to nodes array
                new_node = patch_op.value
                self._validate_node(new_node)

                # Check for duplicate ID
                existing_ids = {node["id"] for node in tree_data["nodes"]}
                if new_node["id"] in existing_ids:
                    raise ValidationError(f"Node ID already exists: {new_node['id']}")

                tree_data["nodes"].append(new_node)
                log_entry = (
                    f"Operation {operation_index}: Added new node '{new_node['id']}'"
                )

            elif len(path_parts) == 2:
                # Add property to specific node by index
                node_index = int(path_parts[1])
                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                # Replace entire node
                new_node = patch_op.value
                self._validate_node(new_node)
                tree_data["nodes"][node_index] = new_node
                log_entry = f"Operation {operation_index}: Replaced node at index {node_index} with '{new_node['id']}'"

            elif len(path_parts) == 3:
                # Add property to specific node
                node_index = int(path_parts[1])
                property_name = path_parts[2]

                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                tree_data["nodes"][node_index][property_name] = patch_op.value
                node_id = tree_data["nodes"][node_index]["id"]

                log_entry = f"Operation {operation_index}: Added property '{property_name}' to node '{node_id}'"

            else:
                raise ValidationError(f"Invalid add path: {patch_op.path}")

        else:
            # Add property to tree root
            tree_data[path_parts[0]] = patch_op.value
            log_entry = (
                f"Operation {operation_index}: Added tree property '{path_parts[0]}'"
            )

        return tree_data, log_entry

    def _apply_remove_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply remove operation."""

        path_parts = patch_op.path.strip("/").split("/")

        if path_parts[0] == "nodes":
            if len(path_parts) == 2:
                # Remove node by index
                node_index = int(path_parts[1])
                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                removed_node = tree_data["nodes"].pop(node_index)
                node_id = removed_node["id"]

                # Remove references to this node
                self._remove_node_references(tree_data, node_id)

                log_entry = f"Operation {operation_index}: Removed node '{node_id}'"

            elif len(path_parts) == 3:
                # Remove property from specific node
                node_index = int(path_parts[1])
                property_name = path_parts[2]

                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                node = tree_data["nodes"][node_index]
                if property_name in node:
                    del node[property_name]

                log_entry = f"Operation {operation_index}: Removed property '{property_name}' from node '{node['id']}'"

            else:
                raise ValidationError(f"Invalid remove path: {patch_op.path}")

        else:
            # Remove property from tree root
            if path_parts[0] in tree_data:
                del tree_data[path_parts[0]]
            log_entry = (
                f"Operation {operation_index}: Removed tree property '{path_parts[0]}'"
            )

        return tree_data, log_entry

    def _apply_replace_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply replace operation."""

        path_parts = patch_op.path.strip("/").split("/")

        if path_parts[0] == "nodes":
            if len(path_parts) == 2:
                # Replace entire node
                node_index = int(path_parts[1])
                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                new_node = patch_op.value
                self._validate_node(new_node)

                old_node_id = tree_data["nodes"][node_index]["id"]
                tree_data["nodes"][node_index] = new_node

                log_entry = f"Operation {operation_index}: Replaced node '{old_node_id}' with '{new_node['id']}'"

            elif len(path_parts) == 3:
                # Replace property in specific node
                node_index = int(path_parts[1])
                property_name = path_parts[2]

                if node_index >= len(tree_data["nodes"]):
                    raise ValidationError(f"Node index out of range: {node_index}")

                tree_data["nodes"][node_index][property_name] = patch_op.value
                node_id = tree_data["nodes"][node_index]["id"]
                log_entry = f"Operation {operation_index}: Replaced property '{property_name}' in node '{node_id}'"

            else:
                raise ValidationError(f"Invalid replace path: {patch_op.path}")

        else:
            # Replace property in tree root
            tree_data[path_parts[0]] = patch_op.value
            log_entry = (
                f"Operation {operation_index}: Replaced tree property '{path_parts[0]}'"
            )

        return tree_data, log_entry

    def _apply_move_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply move operation."""

        # Get value from source path
        source_value = self._get_value_at_path(tree_data, patch_op.from_path)

        # Remove from source
        remove_patch = TreePatchOperation(
            op="remove", path=patch_op.from_path, value=None
        )
        tree_data, _ = self._apply_remove_operation(
            tree_data, remove_patch, operation_index
        )

        # Add to destination
        add_patch = TreePatchOperation(op="add", path=patch_op.path, value=source_value)
        tree_data, _ = self._apply_add_operation(tree_data, add_patch, operation_index)

        log_entry = f"Operation {operation_index}: Moved from '{patch_op.from_path}' to '{patch_op.path}'"

        return tree_data, log_entry

    def _apply_copy_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply copy operation."""

        # Get value from source path
        source_value = self._get_value_at_path(tree_data, patch_op.from_path)

        # Add copy to destination
        add_patch = TreePatchOperation(
            op="add", path=patch_op.path, value=deepcopy(source_value)
        )
        tree_data, _ = self._apply_add_operation(tree_data, add_patch, operation_index)

        log_entry = f"Operation {operation_index}: Copied from '{patch_op.from_path}' to '{patch_op.path}'"

        return tree_data, log_entry

    def _apply_test_operation(
        self,
        tree_data: Dict[str, Any],
        patch_op: TreePatchOperation,
        operation_index: int,
    ) -> Tuple[Dict[str, Any], str]:
        """Apply test operation."""

        current_value = self._get_value_at_path(tree_data, patch_op.path)

        if current_value != patch_op.value:
            raise ValidationError(
                f"Test operation failed: expected {patch_op.value}, got {current_value}"
            )

        log_entry = (
            f"Operation {operation_index}: Test passed for path '{patch_op.path}'"
        )

        return tree_data, log_entry

    def _get_value_at_path(self, tree_data: Dict[str, Any], path: str) -> Any:
        """Get value at specified JSON path."""
        path_parts = path.strip("/").split("/")
        current = tree_data

        for part in path_parts:
            if isinstance(current, list):
                current = current[int(part)]
            elif isinstance(current, dict):
                current = current[part]
            else:
                raise ValidationError(f"Cannot navigate path: {path}")

        return current

    def _remove_node_references(self, tree_data: Dict[str, Any], node_id: str) -> None:
        """Remove all references to a node from other nodes."""
        for node in tree_data["nodes"]:
            # Remove from children_ids
            if "children_ids" in node and node_id in node["children_ids"]:
                node["children_ids"].remove(node_id)

            # Update parent_id if this was the parent
            if node.get("parent_id") == node_id:
                node["parent_id"] = None

    def _validate_node(self, node: Dict[str, Any]) -> None:
        """Validate a single node against schema."""
        try:
            jsonschema.validate(node, self.node_schema)
        except jsonschema.ValidationError as e:
            raise ValidationError(f"Node validation failed: {str(e)}")

    def validate_tree(self, tree_data: Dict[str, Any]) -> None:
        """Validate entire tree structure."""
        try:
            jsonschema.validate(tree_data, self.tree_schema)

            # Additional validation
            self._validate_tree_integrity(tree_data)

        except jsonschema.ValidationError as e:
            raise ValidationError(f"Tree validation failed: {str(e)}")

    def _validate_tree_integrity(self, tree_data: Dict[str, Any]) -> None:
        """Validate tree integrity (relationships, uniqueness, etc.)."""
        nodes = tree_data.get("nodes", [])

        # Check node ID uniqueness
        node_ids = [node["id"] for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValidationError("Duplicate node IDs found")

        # Check parent-child relationships
        node_map = {node["id"]: node for node in nodes}

        for node in nodes:
            # Check parent exists
            parent_id = node.get("parent_id")
            if parent_id and parent_id not in node_map:
                raise ValidationError(
                    f"Parent node '{parent_id}' not found for node '{node['id']}'"
                )

            # Check children exist
            children_ids = node.get("children_ids", [])
            for child_id in children_ids:
                if child_id not in node_map:
                    raise ValidationError(
                        f"Child node '{child_id}' not found for node '{node['id']}'"
                    )

                # Check bidirectional relationship
                child_node = node_map[child_id]
                if child_node.get("parent_id") != node["id"]:
                    raise ValidationError(
                        f"Inconsistent parent-child relationship: {node['id']} <-> {child_id}"
                    )

        # Check for exactly one root node
        root_nodes = [node for node in nodes if node.get("parent_id") is None]
        if len(root_nodes) != 1:
            raise ValidationError(
                f"Tree must have exactly one root node, found {len(root_nodes)}"
            )

    def _ensure_tree_integrity(self, tree_data: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure and fix tree integrity issues."""
        nodes = tree_data.get("nodes", [])
        node_map = {node["id"]: node for node in nodes}

        # Fix parent-child relationships
        for node in nodes:
            children_ids = node.get("children_ids", [])

            # Ensure all children have correct parent_id
            for child_id in children_ids:
                if child_id in node_map:
                    node_map[child_id]["parent_id"] = node["id"]

            # Remove invalid children
            valid_children = [cid for cid in children_ids if cid in node_map]
            node["children_ids"] = valid_children

        # Update levels based on tree structure
        self._update_node_levels(tree_data)

        return tree_data

    def _update_node_levels(self, tree_data: Dict[str, Any]) -> None:
        """Update node levels based on tree hierarchy."""
        nodes = tree_data.get("nodes", [])
        node_map = {node["id"]: node for node in nodes}

        # Find root node
        root_nodes = [node for node in nodes if node.get("parent_id") is None]
        if not root_nodes:
            return

        root_node = root_nodes[0]

        # BFS to update levels
        queue = [(root_node["id"], 0)]
        visited = set()

        while queue:
            node_id, level = queue.pop(0)

            if node_id in visited:
                continue

            visited.add(node_id)
            node = node_map[node_id]
            node["level"] = level

            # Add children to queue
            for child_id in node.get("children_ids", []):
                if child_id in node_map and child_id not in visited:
                    queue.append((child_id, level + 1))

    def create_patch_operations(
        self, original_tree: Dict[str, Any], modified_tree: Dict[str, Any]
    ) -> List[TreePatchOperation]:
        """
        Create patch operations to transform original tree to modified tree.

        Args:
            original_tree: Original tree data
            modified_tree: Target tree data

        Returns:
            List of patch operations
        """
        try:
            # Use jsonpatch to generate operations
            patch = jsonpatch.make_patch(original_tree, modified_tree)

            # Convert to TreePatchOperation objects
            operations = []
            for op in patch:
                operation = TreePatchOperation(
                    op=op["op"],
                    path=op["path"],
                    value=op.get("value"),
                    from_path=op.get("from"),
                )
                operations.append(operation)

            logger.info(
                "Generated patch operations",
                operation_count=len(operations),
                original_nodes=len(original_tree.get("nodes", [])),
                modified_nodes=len(modified_tree.get("nodes", [])),
            )

            return operations

        except Exception as e:
            logger.error("Failed to create patch operations", error=str(e))
            raise ProcessingError(f"Patch generation failed: {str(e)}")


# Service instance
tree_patch_engine = TreePatchEngine()
