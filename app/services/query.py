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
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from app.services.documents import get_document_tree_from_path, get_document_page
from app.core.database import get_documents_collection
from app.core.exceptions import NotFoundError

import mlflow
import os
import base64
from httpx import AsyncClient

# mlflow.pydantic_ai.autolog()

logger = structlog.get_logger()


@dataclass
class QueryDependencies:
    """Dependencies for the query agent containing document context and query tracking."""

    document_id: str
    query: str
    tool_usage: dict
    query_paths: List[str] = field(
        default_factory=lambda: ["/"]
    )  # Track paths accessed during query


class QueryService:
    """Service class for document querying using Pydantic AI agents."""

    def __init__(self):
        """Initialize the QueryService with a configured Pydantic AI agent."""

        self.provider = GoogleProvider(
            api_key=os.getenv("GEMINI_API_KEY"),
        )
        self.model = GoogleModel(
            "gemini-2.5-flash",
            provider=self.provider,
        )
        system_prompt = """
        You are a profesional information retriever, you have 2 tools:
        1. get_subtree_by_paths: to navigate document sections
        2. fetch_page_details: get more details from the pages
        You can navigate through the document structure first to know where the information might be in, then use the fetch_page_details tool if needed.
        Always try to accomplish your mission as you literally have all means to find any information from the document.
        """

        self.agent = Agent(
            self.model,
            deps_type=QueryDependencies,
            system_prompt=system_prompt,
        )

        # Register tools with the agent
        self._register_tools()

    def _register_tools(self):
        """Register tools with the agent."""

        @self.agent.tool
        async def get_subtree_by_paths(
            ctx: RunContext[QueryDependencies], paths: List[str] = ["/"]
        ) -> ToolReturn:
            """
            Get subtree data from specified paths in the document tree.

            Args:
                paths: List of paths to explore (e.g., ["/", "/children/0/children/1"])

            Returns:
                JSON string containing the subtree data for all requested paths
            """
            return await self._get_subtree_by_paths(ctx, paths, 2)

        @self.agent.tool
        async def fetch_page_details(
            ctx: RunContext[QueryDependencies], pages: List[int] = []
        ) -> ToolReturn:
            """
            Fetch document details for specified page numbers.

            Args:
                pages: List of page numbers to fetch details for (1-x)

            Returns:
                JSON string containing image information for the requested pages
            """
            return await self._fetch_page_details(ctx, pages)

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
        with mlflow.start_span(name="navigate_subtree", span_type="TOOL") as span:
            span.set_inputs({"paths": paths, "depth": depth})
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
                            results += f"Path: {subtree['path']}\nName: {subtree['title']}\nSummary: {subtree['summary'][:300]}...\nPages: {','.join([str(p) for p in subtree['pages']])}\n\n"
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
                span.set_outputs({"result": results})
                return ToolReturn(return_value=results)
            except Exception as e:
                logger.error(
                    "Error in get_subtree_by_paths tool",
                    document_id=ctx.deps.document_id,
                    paths=paths,
                    error=str(e),
                )
                raise f"Failed to get subtree data: {str(e)}"

    async def _fetch_page_details(
        self, ctx: RunContext[QueryDependencies], pages: List[int] = []
    ) -> ToolReturn:
        """
        Fetch document images for specified page numbers and analyze them with Gemini.

        Args:
            pages: List of page numbers to fetch images for (1-indexed)

        Returns:
            Analysis results from Gemini for the requested pages
        """
        with mlflow.start_span(name="fetch_page_details", span_type="TOOL") as span:
            span.set_inputs({"pages": pages})
            try:
                query = ctx.deps.query
                document_id = ctx.deps.document_id

                if not pages:
                    return ToolReturn(
                        return_value="No pages specified. Please provide page numbers to analyze."
                    )

                logger.info(
                    "Fetching page details for analysis",
                    document_id=document_id,
                    pages=pages,
                    query=query,
                )

                # Get page images using the get_document_page function
                pages_data = await get_document_page(document_id, pages)

                if not pages_data:
                    return ToolReturn(
                        return_value="No page images could be retrieved for the specified pages."
                    )

                # Analyze each page with Gemini
                analysis_results = []
                total_usage = {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "total_calls": 0,
                }
                prompt = f"""
                Analyze this page image from the document and provide detailed information that could help answer the following query: "{query}"
                
                Please provide:
                1. A summary of the main content on this page
                2. Key information, data, or insights relevant to the query
                3. Any specific details that directly relate to the user's question
                4. Text content if readable
                5. Visual elements like charts, graphs, or diagrams if present
                
                Focus on information that would be useful for answering the user
                """
                image_bytes_list = []
                for page_key, base64_image in pages_data.items():
                    if base64_image is None:
                        analysis_results.append(
                            {
                                "page": page_key,
                                "error": "Image not available for this page",
                            }
                        )
                        continue

                    try:
                        # Convert base64 back to bytes for BinaryContent
                        image_bytes_list.append(base64.b64decode(base64_image))
                    except Exception as e:
                        logger.error(
                            "Failed to fetch page image data",
                            document_id=document_id,
                            page=page_key,
                            error=str(e),
                        )

                # Prepare content for Pydantic AI
                image_contents = [
                    BinaryContent(data=i, media_type="image/png")
                    for i in image_bytes_list
                ]
                content = [
                    prompt,
                    *image_contents,
                ]
                # provider = GoogleProvider(api_key=os.getenv("GEMINI_API_KEY"))
                # model = GoogleModel("gemini-2.5-flash", provider=provider)
                # agent = Agent(model)
                # Run analysis with Gemini 2.5 Flash
                logger.info(
                    "Starting page analysis",
                    document_id=document_id,
                    pages=pages,
                    query=query,
                )
                agent = Agent(self.model)
                result = await agent.run(
                    content,
                    deps=ctx.deps,
                    model_settings=GoogleModelSettings(
                        google_thinking_config={"thinkingBudget": 512}
                    ),
                )

                # Track token usage
                usage = result.usage()
                tool_usage = usage.__dict__

                analysis_results.append(
                    {
                        "page": page_key,
                        "analysis": result.output,
                        "usage": tool_usage,
                    }
                )

                logger.info(
                    "Page analysis completed",
                    document_id=document_id,
                    page=page_key,
                    tokens_used=usage.total_tokens,
                )

                # Save total usage to deps.tool_usage
                ctx.deps.tool_usage["fetch_page_details"] = tool_usage

                logger.info(
                    "Page details fetch completed",
                    document_id=document_id,
                    total_pages_analyzed=len(
                        [r for r in analysis_results if "analysis" in r]
                    ),
                    total_usage=total_usage,
                )
                span.set_outputs(
                    {
                        "result": "\n\n".join(
                            [
                                f"Page {pages[i]}\n" + res.get("analysis")
                                for i, res in enumerate(analysis_results)
                            ]
                        ),
                        "token_usage": usage.__dict__,
                    }
                )
                return ToolReturn(return_value=analysis_results)

            except Exception as e:
                logger.error(
                    "Failed to fetch page details",
                    document_id=ctx.deps.document_id,
                    pages=pages,
                    error=str(e),
                )
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
            deps = QueryDependencies(
                document_id=document_id,
                query=query,
                tool_usage={},
            )

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
