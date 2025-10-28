"""
Query service module using Pydantic AI for document querying with streaming capabilities.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, AsyncGenerator, Optional
import structlog
from pydantic_ai import (
    Agent,
    RunContext,
    ToolReturn,
    BinaryContent,
    FunctionToolCallEvent,
    PartStartEvent,
    AgentRunResultEvent,
)
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel

from app.services.documents import get_document_tree_from_path
from app.core.database import get_documents_collection
from app.core.exceptions import NotFoundError

import mlflow
import os
from httpx import AsyncClient

# mlflow.pydantic_ai.autolog()

logger = structlog.get_logger()


@dataclass
class QueryDependencies:
    """Dependencies for the query agent containing document context and query tracking."""

    document_id: str
    query_paths: List[str] = field(
        default_factory=lambda: ["/"]
    )  # Track paths accessed during query


class QueryService:
    """Service class for document querying using Pydantic AI agents."""

    def __init__(self):
        """Initialize the QueryService with a configured Pydantic AI agent."""
        self.custom_http_client = AsyncClient(timeout=30)

        self.provider = GoogleProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
            http_client=self.custom_http_client,
        )
        self.model = GoogleModel(
            "gemini-2.5-flash",
            provider=self.provider,
        )

        self.agent = Agent(
            self.model,
            deps_type=QueryDependencies,
            system_prompt=(
                "You are a helpful document analysis assistant. You can explore document trees "
                "and fetch images to answer user questions about documents. Use the available tools "
                "to navigate through the document structure and retrieve relevant information. "
                "When exploring paths, start with the root and then navigate to specific sections "
                "based on the user's query. Always provide detailed and accurate responses based "
                "on the document content you can access."
            ),
        )

        # Register tools with the agent
        self._register_tools()

    def _register_tools(self):
        """Register tools with the agent."""

        @self.agent.tool
        async def get_subtree_by_paths(
            ctx: RunContext[QueryDependencies], paths: List[str] = ["/"], depth: int = 3
        ) -> ToolReturn:
            """
            Get subtree data from specified paths in the document tree.

            Args:
                paths: List of paths to explore (e.g., ["/", "/children/0/children/1"])
                depth: Maximum depth to retrieve for each path (1-10)

            Returns:
                JSON string containing the subtree data for all requested paths
            """
            return await self._get_subtree_by_paths(ctx, paths, depth)

        @self.agent.tool
        async def fetch_image_by_pages(
            ctx: RunContext[QueryDependencies], pages: List[int] = []
        ) -> ToolReturn:
            """
            Fetch document images for specified page numbers.

            Args:
                pages: List of page numbers to fetch images for (0-indexed)

            Returns:
                JSON string containing image information for the requested pages
            """
            return await self._fetch_image_by_pages(ctx, pages)

    async def _get_subtree_by_paths(
        self,
        ctx: RunContext[QueryDependencies],
        paths: List[str] = ["/"],
        depth: int = 3,
    ) -> ToolReturn:
        """
        Get subtree data from specified paths in the document tree.

        Args:
            paths: List of paths to explore (e.g., ["/", "/children/0/children/1"])
            depth: Maximum depth to retrieve for each path (1-10)

        Returns:
            Serialized Tree starts from paths
        """
        try:
            # Add paths to context for tracking query journey
            for path in paths:
                if path not in ctx.deps.query_paths:
                    ctx.deps.query_paths.append(path)

            results = ""
            for path in paths:
                try:
                    # print(
                    #     f"Fetching subtree from {ctx.deps.document_id} at {path}",
                    #     flush=True,
                    # )
                    subtree_data = await get_document_tree_from_path(
                        ctx.deps.document_id, path, depth, serialize=True
                    )
                    # print(f"Fetched subtree {subtree_data}", flush=True)
                    for subtree in subtree_data:
                        results += f"Path {subtree['path']}\nName {subtree['title']}\nPages {','.join([str(p) for p in subtree['pages']])}\n\n"
                    logger.info(
                        "Retrieved subtree for path",
                        document_id=ctx.deps.document_id,
                        path=path,
                        depth=depth,
                    )
                except Exception as e:
                    logger.error(
                        "Failed to get subtree for path",
                        document_id=ctx.deps.document_id,
                        path=path,
                        error=str(e),
                    )
                    results = "Failed to retrieve path {path}: {str(e)}"
            # print("Fetch subtrees: ", flush=True)
            # print(results, flush=True)
            return ToolReturn(return_value=results)
        except Exception as e:
            logger.error(
                "Error in get_subtree_by_paths tool",
                document_id=ctx.deps.document_id,
                paths=paths,
                error=str(e),
            )
            raise f"Failed to get subtree data: {str(e)}"

    async def _fetch_image_by_pages(
        self, ctx: RunContext[QueryDependencies], pages: List[int] = []
    ) -> ToolReturn:
        """
        Fetch document images for specified page numbers.

        Args:
            pages: List of page numbers to fetch images for (0-indexed)

        Returns:
            Images of the pages
        """
        try:
            return ToolReturn(
                return_value="Not yet implemented. Please ask the user to check the pages manually"
            )
        except Exception as e:
            raise f"Failed to fetch document page images: {str(e)}"

    async def query_doc(
        self, document_id: str, query: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Query a document using Pydantic AI agent with streaming response.

        Args:
            document_id: ID of the document to query
            query: User's query about the document
            user_id: Optional user ID for tracking

        Yields:
            Dictionary containing streaming events and final results
        """
        try:
            # Verify document exists
            from bson import ObjectId

            documents_collection = get_documents_collection()
            document = await documents_collection.find_one(
                {"_id": ObjectId(document_id)},
            )
            # print(f"Found Document Tree : {document}", flush=True)

            if not document:
                yield {
                    "type": "error",
                    "error": f"Document not found: {document_id}",
                    "document_id": document_id,
                }
                return

            # Create dependencies with document context
            deps = QueryDependencies(document_id=document_id)

            logger.info(
                "Starting document query",
                document_id=document_id,
                query=query,
                user_id=user_id,
            )

            # Yield initial status
            yield {
                "type": "start",
                "document_id": document_id,
                "document_title": document.get("title", "Unknown"),
                "query": query,
                "user_id": user_id,
            }

            # Stream the agent response
            with mlflow.start_span(name="Query") as span:
                span.set_inputs({"query": query, "deps": deps.__dict__})
                async for event in self.agent.run_stream_events(query, deps=deps):
                    # print(f"Received Event: {event}", flush=True)
                    # # Convert Pydantic AI events to our format
                    if isinstance(event, PartStartEvent):
                        if hasattr(event, "part"):
                            event_part = event.part
                            if hasattr(event_part, "content") and event_part.content:
                                yield {
                                    "type": "text_delta",
                                    "content": event_part.content,
                                }
                    if isinstance(event, FunctionToolCallEvent):
                        if hasattr(event, "part"):
                            event_part = event.part
                            if hasattr(event_part, "tool_name"):
                                yield {
                                    "type": "tool_call",
                                    "content": f"Calling {event_part.tool_name}",
                                    "tool_args": event_part.args
                                    if event_part.args
                                    else {},
                                }
                    if isinstance(event, AgentRunResultEvent):
                        yield {
                            "type": "final_result",
                            "content": event.result.output,
                            "references": deps.__dict__,
                            "usage": event.result.usage().__dict__,
                        }
                span.set_outputs(
                    {
                        "output": event.result.output,
                        "references": deps.__dict__,
                        "usage": event.result.usage().__dict__,
                    }
                )
            logger.info(
                "Completed document query",
                document_id=document_id,
                query=query,
                query_paths=deps.query_paths,
                user_id=user_id,
            )

        except Exception as e:
            logger.error(
                "Error in document query",
                document_id=document_id,
                query=query,
                error=str(e),
                user_id=user_id,
            )
            yield {
                "type": "error",
                "error": str(e),
                "document_id": document_id,
                "user_id": user_id,
            }
