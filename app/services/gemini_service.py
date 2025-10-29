"""
Gemini Flash service for visual document analysis with hierarchical content categorization.
Handles page-by-page visual processing to extract Sections (L1), Subjects (L2), and Topics (L3).
"""

from __future__ import annotations

import asyncio
import base64
from typing import List, Dict, Any, Optional
import structlog
import time
import os
import uuid

from pydantic_ai import Agent, RunContext, BinaryContent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.google import GoogleModelSettings

from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ProcessingError, ValidationError
from app.core.storage import download_file_data
from app.core.database import get_subtrees_collection
from datetime import datetime
from bson import ObjectId


logger = structlog.get_logger()


class VisualElement(BaseModel):
    """Visual element detected on a page."""

    element: str = Field(description="Type of visual element (e.g., image, table)")
    title: str = Field(description="Clear, descriptive title")
    summary: str = Field(description="Brief content summary")


# Custom models for content hierarchy
class Topic(BaseModel):
    """L3 - Low level topics for detail focus content main topics."""

    title: str = Field(description="Clear, descriptive title")
    summary: str = Field(description="Brief content summary")


class Subject(BaseModel):
    """L2 - Mid level topics indicating content clustering."""

    title: str = Field(description="Clear, descriptive title")
    summary: str = Field(description="Brief content summary")
    children: List[Topic] = Field(
        description="L3 - Low level topics for detail focus content main topics",
    )


class Section(BaseModel):
    """L1 - High level topics indicating content sections."""

    title: str = Field(description="Clear, descriptive title")
    summary: str = Field(description="Brief content summary")
    children: List[Subject] = Field(
        description="L2 - Mid level topics indicating content clustering",
    )


class PageAnalysisResult(BaseModel):
    """Result from analyzing a single page."""

    page_tree: List[Section] = Field(
        description="Hierarchical tree structure with Sections as root nodes containing Subjects and Topics",
    )
    visual_elements: List[VisualElement] = Field(description="Visual elements detected")


class GeminiVisualAnalyzer:
    """Gemini Flash service for page-by-page visual document analysis using Pydantic AI."""

    def __init__(self):
        self.settings = get_settings()
        self._configure_gemini()

    def _count_tree_nodes(self, page_result: PageAnalysisResult) -> Dict[str, int]:
        """Count nodes in the tree structure."""
        sections_count = len(page_result.page_tree)
        subjects_count = sum(len(section.children) for section in page_result.page_tree)
        topics_count = sum(
            len(subject.children)
            for section in page_result.page_tree
            for subject in section.children
        )

        return {
            "sections": sections_count,
            "subjects": subjects_count,
            "topics": topics_count,
        }

    async def save_page_subtree(
        self,
        document_id: str,
        page_number: int,
        page_tree: List[Section],
        visual_elements: List[VisualElement],
        image_path: str,
        token_usage: Dict[str, int],
    ) -> str:
        """
        Save page subtree to MongoDB trees collection.

        Args:
            document_id: Document ID
            page_number: Page number
            page_tree: List of Section objects representing the page tree
            image_path: Path to the page image

        Returns:
            Inserted document ID as string
        """
        try:
            subtrees_collection = get_subtrees_collection()

            # Convert page_tree to dict format for MongoDB storage
            page_tree_dict = [section.model_dump() for section in page_tree]
            elements = [ve.model_dump() for ve in visual_elements]

            # Create document structure for MongoDB
            subtree_document = {
                "document_id": document_id,
                "page_number": page_number,
                "page_tree": page_tree_dict,
                "visual_elements": elements,
                "image_path": image_path,
                "created_at": datetime.utcnow(),
                "processing_metadata": {
                    "sections_count": len(page_tree),
                    "subjects_count": sum(
                        len(section.children) for section in page_tree
                    ),
                    "topics_count": sum(
                        len(subject.children)
                        for section in page_tree
                        for subject in section.children
                    ),
                    "token_usage": token_usage,
                    "analysis_method": "gemini_visual_page_analysis",
                },
            }

            # Insert into MongoDB
            result = await subtrees_collection.insert_one(subtree_document)

            logger.info(
                "Page subtree saved to MongoDB",
                document_id=document_id,
                page_number=page_number,
                subtree_id=str(result.inserted_id),
                sections_count=subtree_document["processing_metadata"][
                    "sections_count"
                ],
                subjects_count=subtree_document["processing_metadata"][
                    "subjects_count"
                ],
                topics_count=subtree_document["processing_metadata"]["topics_count"],
            )

            return str(result.inserted_id)

        except Exception as e:
            logger.error(
                "Failed to save page subtree to MongoDB",
                document_id=document_id,
                page_number=page_number,
                error=str(e),
            )
            # Don't raise exception to avoid breaking the analysis pipeline
            # Return None to indicate save failure
            return None

    def _configure_gemini(self):
        """Configure Gemini API with Pydantic AI."""
        try:
            # Create Google provider
            self.provider = GoogleProvider(api_key=self.settings.GEMINI_API_KEY)

            # Create Google model
            self.model = GoogleModel(
                model_name=self.settings.GEMINI_MODEL,
                provider=self.provider,
            )

            # Create agent for page analysis
            self.page_agent = Agent(model=self.model)

            logger.info(
                "Gemini Pydantic AI configured for page-by-page analysis",
                model=self.settings.GEMINI_MODEL,
                temperature=self.settings.GEMINI_TEMPERATURE,
            )

        except Exception as e:
            logger.error("Failed to configure Gemini Pydantic AI", error=str(e))
            raise ExternalServiceError(f"Gemini configuration failed: {str(e)}")

    async def analyze_document_images(
        self, image_paths: List[str], document_id: str, document_title: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze document images page-by-page to extract topics.

        Args:
            image_paths: List of image paths in MinIO
            document_id: Document ID
            document_title: Document title for context

        Returns:
            Dictionary containing page analysis results
        """
        try:
            logger.info(
                "Starting page-by-page visual document analysis",
                document_id=document_id,
                image_count=len(image_paths),
                title=document_title,
            )

            start_time = time.time()

            # Process each page sequentially
            page_results = []

            for i, image_path in enumerate(image_paths):
                page_number = i + 1

                logger.info(
                    "Analyzing page",
                    document_id=document_id,
                    page_number=page_number,
                    total_pages=len(image_paths),
                )

                # Analyze single page
                page_result = await self.analyze_single_page(
                    image_path,
                    page_number,
                    document_title,
                    document_id,
                )

                page_results.append(page_result)

                # Count nodes in the tree structure
                node_counts = self._count_tree_nodes(page_result)
                logger.info(
                    "Page analysis completed",
                    document_id=document_id,
                    page_number=page_number,
                    sections_found=node_counts["sections"],
                    subjects_found=node_counts["subjects"],
                    topics_found=node_counts["topics"],
                )

            processing_time = time.time() - start_time

            # Calculate totals across all pages
            total_sections = sum(
                self._count_tree_nodes(result)["sections"] for result in page_results
            )
            total_subjects = sum(
                self._count_tree_nodes(result)["subjects"] for result in page_results
            )
            total_topics = sum(
                self._count_tree_nodes(result)["topics"] for result in page_results
            )

            logger.info(
                "Page-by-page visual analysis completed",
                document_id=document_id,
                processing_time=processing_time,
                pages_processed=len(page_results),
                total_sections=total_sections,
                total_subjects=total_subjects,
                total_topics=total_topics,
            )

            # return {
            #     "document_id": document_id,
            #     "document_title": document_title,
            #     "total_pages": len(image_paths),
            #     "page_results": [result.model_dump() for result in page_results],
            #     "processing_time": processing_time,
            #     "total_sections": total_sections,
            #     "total_subjects": total_subjects,
            #     "total_topics": total_topics,
            #     "status": "completed",
            #     "analysis_method": "gemini_visual_page_by_page",
            # }

        except Exception as e:
            logger.error(
                "Page-by-page visual analysis failed",
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Visual analysis failed: {str(e)}")

    async def analyze_single_page(
        self,
        image_path: str,
        page_number: int,
        document_title: str,
        document_id: str = None,
    ) -> PageAnalysisResult:
        """
        Analyze a single page and extract topics.

        Args:
            image_path: Path to the image in MinIO
            page_number: Page number being analyzed
            document_title: Document title for context
            document_id: Document ID for MongoDB storage (optional)

        Returns:
            PageAnalysisResult with topics found on this page
        """
        try:
            page_start_time = time.time()

            # Download and prepare image
            image_data = await self._prepare_single_image(image_path, page_number)

            # Create prompt for this page
            prompt = self._create_single_page_analysis_prompt(
                document_title, page_number
            )

            # Prepare content for Pydantic AI
            # Convert base64 back to bytes for BinaryContent
            image_bytes = base64.b64decode(image_data["image_data"])

            content = [
                prompt,
                BinaryContent(data=image_bytes, media_type=image_data["mime_type"]),
            ]

            # Run single page analysis with structured output
            result = await self.page_agent.run(
                content,
                output_type=PageAnalysisResult,
                model_settings=GoogleModelSettings(
                    google_thinking_config={"thinkingBudget": 512}
                ),
            )
            usage = {
                "input_tokens": result.usage().input_tokens,
                "output_tokens": result.usage().output_tokens,
                "total_tokens": result.usage().total_tokens,
                "total_calls": result.usage().requests,
            }
            print(f"Usage: {result.usage()}", flush=True)
            page_result = result.output
            processing_time = time.time() - page_start_time

            # Count nodes in the tree structure
            node_counts = self._count_tree_nodes(page_result)

            # Save to MongoDB if document_id is provided
            subtree_id = None
            if document_id:
                try:
                    subtree_id = await self.save_page_subtree(
                        document_id=document_id,
                        page_number=page_number,
                        page_tree=page_result.page_tree,
                        visual_elements=page_result.visual_elements,
                        image_path=image_path,
                        token_usage=usage,
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to save page subtree to MongoDB, continuing with analysis",
                        document_id=document_id,
                        page_number=page_number,
                        error=str(e),
                    )

            logger.debug(
                "Single page analysis completed",
                page_number=page_number,
                sections_found=node_counts["sections"],
                subjects_found=node_counts["subjects"],
                topics_found=node_counts["topics"],
                processing_time=processing_time,
                subtree_id=subtree_id,
            )

            return page_result

        except Exception as e:
            logger.error(
                "Single page analysis failed",
                page_number=page_number,
                image_path=image_path,
                error=str(e),
            )
            # Return empty result instead of failing completely
            return PageAnalysisResult(
                page_tree=[],
                visual_elements={"error": str(e)},
            )

    async def _prepare_single_image(
        self, image_path: str, page_number: int
    ) -> Dict[str, Any]:
        """Prepare a single image for analysis."""
        try:
            # Download image from MinIO
            image_bytes = await download_file_data(
                bucket=self.settings.MINIO_BUCKET_IMAGES, object_name=image_path
            )

            # Encode to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            return {
                "page_number": page_number,
                "image_path": image_path,
                "image_data": image_b64,
                "mime_type": "image/png",
                "size": len(image_bytes),
            }

        except Exception as e:
            logger.error(
                "Failed to prepare image",
                image_path=image_path,
                page_number=page_number,
                error=str(e),
            )
            raise ProcessingError(
                f"Image preparation failed for page {page_number}: {str(e)}"
            )

    def _create_single_page_analysis_prompt(
        self, document_title: str, page_number: int
    ) -> str:
        """Create analysis prompt for a single page."""

        return f"""Analyze page {page_number} from "{document_title or "Unknown"}" and build a hierarchical tree:

• **Sections** (L1): Main headings, chapters, major divisions
• **Subjects** (L2): Subsections, themes, categories  
• **Topics** (L3): Specific points, details, bullet items

Structure: Section → Subject → Topic (each level contains children)

For each node provide:
- title: Clear, descriptive
- summary: Brief content description

Return as page_tree with proper parent-child relationships and visual_elements detected."""

    async def cleanup(self):
        """Clean up resources and close connections."""
        try:
            # Clean up the provider and model references
            if hasattr(self, "provider"):
                # Try to close the provider if it has a close method
                if hasattr(self.provider, "aclose"):
                    try:
                        await self.provider.aclose()
                    except (AttributeError, Exception) as e:
                        # Ignore the _async_httpx_client error and other cleanup errors
                        logger.debug(
                            "Provider cleanup completed with minor issues", error=str(e)
                        )

                # Clear the provider reference
                self.provider = None

            if hasattr(self, "model"):
                self.model = None

            if hasattr(self, "page_agent"):
                self.page_agent = None

            logger.info("Gemini service cleanup completed successfully")

        except Exception as e:
            # Log but don't raise - we don't want cleanup errors to crash the shutdown
            logger.warning("Gemini service cleanup encountered issues", error=str(e))


# Service instance
gemini_service = GeminiVisualAnalyzer()
