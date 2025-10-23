"""
Gemini Flash service for visual document analysis and topic tree generation.
Handles all visual processing and content understanding.
"""

import asyncio
import base64
from typing import List, Dict, Any, Optional
import structlog
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import time

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ProcessingError, ValidationError
from app.core.storage import download_file_data
from app.models.tree import NodeType


logger = structlog.get_logger()


class GeminiVisualAnalyzer:
    """Gemini Flash service for visual document analysis."""

    def __init__(self):
        self.settings = get_settings()
        self._configure_gemini()

    def _configure_gemini(self):
        """Configure Gemini API."""
        try:
            genai.configure(api_key=self.settings.GEMINI_API_KEY)

            # Configure model
            self.model = genai.GenerativeModel(
                model_name=self.settings.GEMINI_MODEL,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.settings.GEMINI_TEMPERATURE,
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                },
            )

            logger.info(
                "Gemini model configured",
                model=self.settings.GEMINI_MODEL,
                temperature=self.settings.GEMINI_TEMPERATURE,
            )

        except Exception as e:
            logger.error("Failed to configure Gemini", error=str(e))
            raise ExternalServiceError(f"Gemini configuration failed: {str(e)}")

    async def analyze_document_images(
        self, image_paths: List[str], document_id: str, document_title: str = ""
    ) -> Dict[str, Any]:
        """
        Analyze document images and generate topic tree structure.

        Args:
            image_paths: List of image paths in MinIO
            document_id: Document ID
            document_title: Document title for context

        Returns:
            Dictionary containing the generated topic tree
        """
        try:
            logger.info(
                "Starting visual document analysis",
                document_id=document_id,
                image_count=len(image_paths),
                title=document_title,
            )

            start_time = time.time()

            # Download and prepare images
            images_data = await self._prepare_images(image_paths)

            # Generate topic tree through visual analysis
            tree_data = await self._generate_topic_tree(
                images_data, document_id, document_title
            )

            processing_time = time.time() - start_time

            logger.info(
                "Visual analysis completed",
                document_id=document_id,
                processing_time=processing_time,
                node_count=len(tree_data.get("nodes", [])),
            )

            return {
                "tree_data": tree_data,
                "processing_time": processing_time,
                "images_analyzed": len(images_data),
                "analysis_method": "gemini_visual",
            }

        except Exception as e:
            logger.error(
                "Visual analysis failed", document_id=document_id, error=str(e)
            )
            raise ProcessingError(f"Visual analysis failed: {str(e)}")

    async def _prepare_images(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """Prepare images for Gemini analysis."""
        images_data = []

        for i, image_path in enumerate(image_paths):
            try:
                # Download image from MinIO
                image_bytes = await download_file_data(
                    bucket=self.settings.MINIO_BUCKET_IMAGES, object_name=image_path
                )

                # Encode to base64
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")

                images_data.append(
                    {
                        "page_number": i + 1,
                        "image_path": image_path,
                        "image_data": image_b64,
                        "mime_type": "image/png",
                    }
                )

                logger.debug(
                    "Image prepared for analysis",
                    page_number=i + 1,
                    image_path=image_path,
                    size=len(image_bytes),
                )

            except Exception as e:
                logger.error(
                    "Failed to prepare image", image_path=image_path, error=str(e)
                )
                # Continue with other images
                continue

        if not images_data:
            raise ProcessingError("No images could be prepared for analysis")

        return images_data

    async def _generate_topic_tree(
        self, images_data: List[Dict[str, Any]], document_id: str, document_title: str
    ) -> Dict[str, Any]:
        """Generate topic tree from visual analysis."""

        # Create the analysis prompt
        prompt = self._create_analysis_prompt(document_title, len(images_data))

        # Prepare images for Gemini
        image_parts = []
        for img_data in images_data:
            image_parts.append(
                {"mime_type": img_data["mime_type"], "data": img_data["image_data"]}
            )

        try:
            # Run analysis in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._analyze_with_gemini, prompt, image_parts
            )

            # Parse and validate response
            tree_data = self._parse_gemini_response(response, images_data)

            return tree_data

        except Exception as e:
            logger.error(
                "Gemini analysis failed", document_id=document_id, error=str(e)
            )
            raise ExternalServiceError(f"Gemini analysis failed: {str(e)}")

    def _create_analysis_prompt(self, document_title: str, page_count: int) -> str:
        """Create the analysis prompt for Gemini."""
        return f"""
You are an expert document analyzer. Analyze the provided document images and create a comprehensive hierarchical topic tree structure.

Document Information:
- Title: {document_title or "Unknown Document"}
- Total Pages: {page_count}

Instructions:
1. Examine each page image carefully to understand the document structure and content
2. Identify the main topics, subtopics, and their hierarchical relationships
3. Create a detailed topic tree that represents the document's organization
4. For each node, provide:
   - A clear, descriptive title
   - Node type (root, chapter, section, subsection, topic, subtopic, leaf)
   - Brief description of the content
   - Page numbers where this topic appears
   - Keywords related to this topic
   - Confidence score for the classification

Output Format:
Return a JSON object with the following structure:
{{
    "title": "Document Topic Tree",
    "description": "Brief description of the document",
    "root_node_id": "root",
    "nodes": [
        {{
            "id": "unique_node_id",
            "title": "Node Title",
            "description": "Node description",
            "node_type": "root|chapter|section|subsection|topic|subtopic|leaf",
            "level": 0,
            "page_numbers": [1, 2, 3],
            "parent_id": null,
            "children_ids": ["child1", "child2"],
            "content_summary": "Summary of content in this section",
            "confidence_score": 0.95,
            "keywords": ["keyword1", "keyword2"]
        }}
    ]
}}

Requirements:
- Create a logical hierarchy with proper parent-child relationships
- Ensure all node IDs are unique
- Use descriptive titles and summaries
- Include accurate page number references
- Maintain consistency in node types and levels
- Provide meaningful keywords for each node
- Assign realistic confidence scores (0.0-1.0)

Focus on creating a comprehensive, accurate representation of the document's structure and content based on visual analysis.
"""

    def _analyze_with_gemini(
        self, prompt: str, image_parts: List[Dict[str, Any]]
    ) -> str:
        """Synchronous Gemini analysis."""
        try:
            # Prepare content for Gemini
            content = [prompt]

            # Add images
            for image_part in image_parts:
                content.append(
                    {"mime_type": image_part["mime_type"], "data": image_part["data"]}
                )

            # Generate response
            response = self.model.generate_content(content, stream=False)

            if not response.text:
                raise ExternalServiceError("Empty response from Gemini")

            return response.text

        except Exception as e:
            logger.error("Gemini API call failed", error=str(e))
            raise ExternalServiceError(f"Gemini API error: {str(e)}")

    def _parse_gemini_response(
        self, response_text: str, images_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Parse and validate Gemini response."""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValidationError("No JSON found in Gemini response")

            json_text = response_text[json_start:json_end]
            tree_data = json.loads(json_text)

            # Validate structure
            self._validate_tree_structure(tree_data)

            # Enhance with metadata
            tree_data["generation_metadata"] = {
                "model_used": self.settings.GEMINI_MODEL,
                "temperature": self.settings.GEMINI_TEMPERATURE,
                "pages_analyzed": len(images_data),
                "generation_timestamp": time.time(),
            }

            return tree_data

        except json.JSONDecodeError as e:
            logger.error("Failed to parse Gemini JSON response", error=str(e))
            raise ValidationError(f"Invalid JSON in Gemini response: {str(e)}")
        except Exception as e:
            logger.error("Failed to process Gemini response", error=str(e))
            raise ProcessingError(f"Response processing failed: {str(e)}")

    def _validate_tree_structure(self, tree_data: Dict[str, Any]) -> None:
        """Validate the generated tree structure."""
        required_fields = ["title", "nodes"]
        for field in required_fields:
            if field not in tree_data:
                raise ValidationError(f"Missing required field: {field}")

        nodes = tree_data["nodes"]
        if not isinstance(nodes, list) or len(nodes) == 0:
            raise ValidationError("Tree must contain at least one node")

        # Validate each node
        node_ids = set()
        for node in nodes:
            self._validate_node(node, node_ids)

        # Check for root node
        root_nodes = [n for n in nodes if n.get("parent_id") is None]
        if len(root_nodes) != 1:
            raise ValidationError("Tree must have exactly one root node")

    def _validate_node(self, node: Dict[str, Any], node_ids: set) -> None:
        """Validate individual node structure."""
        required_fields = ["id", "title", "node_type", "level"]
        for field in required_fields:
            if field not in node:
                raise ValidationError(f"Node missing required field: {field}")

        node_id = node["id"]
        if node_id in node_ids:
            raise ValidationError(f"Duplicate node ID: {node_id}")
        node_ids.add(node_id)

        # Validate node type
        valid_types = [t.value for t in NodeType]
        if node["node_type"] not in valid_types:
            raise ValidationError(f"Invalid node type: {node['node_type']}")

        # Validate level
        if not isinstance(node["level"], int) or node["level"] < 0:
            raise ValidationError(f"Invalid node level: {node['level']}")

    async def analyze_specific_content(
        self, image_paths: List[str], query: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze specific content in document images based on a query.

        Args:
            image_paths: List of image paths to analyze
            query: Specific query or question
            context: Additional context for analysis

        Returns:
            Analysis results
        """
        try:
            logger.info(
                "Starting specific content analysis",
                image_count=len(image_paths),
                query=query[:100] + "..." if len(query) > 100 else query,
            )

            # Prepare images
            images_data = await self._prepare_images(image_paths)

            # Create analysis prompt
            prompt = self._create_content_analysis_prompt(query, context)

            # Prepare images for Gemini
            image_parts = []
            for img_data in images_data:
                image_parts.append(
                    {"mime_type": img_data["mime_type"], "data": img_data["image_data"]}
                )

            # Run analysis
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, self._analyze_with_gemini, prompt, image_parts
            )

            # Parse response
            analysis_result = self._parse_content_analysis_response(response)

            logger.info(
                "Content analysis completed",
                query=query[:50] + "..." if len(query) > 50 else query,
                confidence=analysis_result.get("confidence_score", 0),
            )

            return analysis_result

        except Exception as e:
            logger.error(
                "Content analysis failed",
                query=query[:50] + "..." if len(query) > 50 else query,
                error=str(e),
            )
            raise ProcessingError(f"Content analysis failed: {str(e)}")

    def _create_content_analysis_prompt(
        self, query: str, context: Optional[str]
    ) -> str:
        """Create prompt for specific content analysis."""
        context_text = f"\\nContext: {context}" if context else ""

        return f"""
Analyze the provided document images to answer the following query:

Query: {query}{context_text}

Instructions:
1. Examine all provided images carefully
2. Look for information relevant to the query
3. Provide a comprehensive answer based on visual content
4. Include specific page references where information was found
5. Provide confidence scores for your findings

Output Format:
Return a JSON object with the following structure:
{{
    "answer": "Detailed answer to the query",
    "confidence_score": 0.95,
    "evidence": [
        {{
            "page_number": 1,
            "content_description": "Description of relevant content found",
            "relevance_score": 0.9
        }}
    ],
    "reasoning": "Explanation of how the answer was derived",
    "additional_notes": "Any additional relevant information"
}}

Focus on accuracy and provide specific references to support your answer.
"""

    def _parse_content_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse content analysis response."""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValidationError("No JSON found in analysis response")

            json_text = response_text[json_start:json_end]
            analysis_data = json.loads(json_text)

            # Validate required fields
            required_fields = ["answer", "confidence_score"]
            for field in required_fields:
                if field not in analysis_data:
                    raise ValidationError(f"Missing required field: {field}")

            return analysis_data

        except json.JSONDecodeError as e:
            logger.error("Failed to parse analysis JSON response", error=str(e))
            raise ValidationError(f"Invalid JSON in analysis response: {str(e)}")
        except Exception as e:
            logger.error("Failed to process analysis response", error=str(e))
            raise ProcessingError(f"Analysis response processing failed: {str(e)}")


# Service instance
gemini_service = GeminiVisualAnalyzer()
