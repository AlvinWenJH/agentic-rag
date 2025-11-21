"""
Analysis service module using Pydantic AI for generating analysis drafts and processing document analysis.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any
import structlog
from datetime import datetime
from bson import ObjectId

from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel, GoogleModelSettings

from app.core.config import get_settings
from app.core.exceptions import ExternalServiceError, ProcessingError
from app.core.database import (
    get_analysis_collection,
    get_analysis_results_collection,
    get_documents_collection,
)
from app.models.analysis import (
    DraftAnalysisResponse,
    AnalysisResultItem,
    AnalysisResultStatus,
    AnalysisStatsResponse,
    EvaluationResult,
)
from app.services.query import QueryService


logger = structlog.get_logger()


class AnalysisService:
    """Service class for analysis using Pydantic AI agents."""

    def __init__(self):
        """Initialize the AnalysisService with a configured Pydantic AI agent."""
        self.settings = get_settings()
        self._configure_agent()

    def _configure_agent(self):
        """Configure Pydantic AI agent for analysis draft generation."""
        try:
            # Create Google provider
            self.provider = GoogleProvider(api_key=self.settings.GEMINI_API_KEY)

            # Create Google model
            self.model = GoogleModel(
                model_name=self.settings.GEMINI_MODEL,
                provider=self.provider,
            )

            # Create agent for draft analysis generation
            self.draft_agent = Agent(model=self.model)

            # Create agent for evaluation
            self.evaluation_agent = Agent(model=self.model)

            logger.info(
                "Analysis service Pydantic AI configured",
                model=self.settings.GEMINI_MODEL,
            )

        except Exception as e:
            logger.error("Failed to configure Analysis Pydantic AI", error=str(e))
            raise ExternalServiceError(f"Analysis AI configuration failed: {str(e)}")

    async def generate_draft_analysis(self, text: str) -> DraftAnalysisResponse:
        """
        Generate draft analysis from free text using structured LLM output.

        Args:
            text: Free text describing analysis needs

        Returns:
            DraftAnalysisResponse with structured title, description, and analysis items
        """
        try:
            logger.info("Generating draft analysis from text", text_length=len(text))
            start_time = time.time()

            # Create prompt for draft analysis generation
            prompt = self._create_draft_analysis_prompt(text)

            # Run agent with structured output
            result = await self.draft_agent.run(
                prompt,
                output_type=DraftAnalysisResponse,
                model_settings=GoogleModelSettings(
                    temperature=0.7,
                ),
            )

            draft_analysis = result.output
            processing_time = time.time() - start_time

            logger.info(
                "Draft analysis generated",
                title=draft_analysis.title,
                items_count=len(draft_analysis.items),
                processing_time=processing_time,
            )

            return draft_analysis

        except Exception as e:
            logger.error("Failed to generate draft analysis", error=str(e))
            raise ProcessingError(f"Draft analysis generation failed: {str(e)}")

    def _create_draft_analysis_prompt(self, text: str) -> str:
        """Create prompt for draft analysis generation."""
        return f"""Based on the following text, generate a comprehensive analysis plan:

Text: "{text}"

Create:
1. A clear, descriptive title for the analysis
2. A detailed description of what the analysis will cover
3. A list of specific analysis items (questions) that should be answered, each with:
   - A clear question
   - Optional context to help answer the question
   - An order number

The analysis items should be thorough and cover all aspects mentioned in the text.
Generate between 3-10 analysis items depending on the complexity of the topic."""

    async def process_document_analysis(
        self,
        document_id: str,
        analysis_id: str,
        user_id: Optional[str] = None,
    ) -> str:
        """
        Process document analysis using analysis items.
        This method starts the async processing and returns the result ID.

        Args:
            document_id: Document ID to analyze
            analysis_id: Analysis ID with items to process
            user_id: Optional user ID

        Returns:
            Analysis result ID
        """
        try:
            logger.info(
                "Starting document analysis",
                document_id=document_id,
                analysis_id=analysis_id,
                user_id=user_id,
            )

            # Validate document exists
            documents_collection = get_documents_collection()
            document = await documents_collection.find_one(
                {"_id": ObjectId(document_id)}
            )
            if not document:
                raise ProcessingError(f"Document {document_id} not found")

            # Validate analysis exists
            analysis_collection = get_analysis_collection()
            analysis = await analysis_collection.find_one(
                {"_id": ObjectId(analysis_id)}
            )
            if not analysis:
                raise ProcessingError(f"Analysis {analysis_id} not found")

            # Create analysis result record
            results_collection = get_analysis_results_collection()
            result_data = {
                "document_id": document_id,
                "analysis_id": analysis_id,
                "analysis_title": analysis.get("title", ""),
                "document_title": document.get("title", ""),
                "status": AnalysisResultStatus.PENDING,
                "results": [],
                "total_items": len(analysis.get("items", [])),
                "completed_items": 0,
                "user_id": user_id,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = await results_collection.insert_one(result_data)
            result_id = str(result.inserted_id)

            # Start background processing
            asyncio.create_task(
                self._process_analysis_background(
                    result_id=result_id,
                    document_id=document_id,
                    analysis=analysis,
                    document=document,
                )
            )

            logger.info(
                "Document analysis processing started",
                result_id=result_id,
                document_id=document_id,
                analysis_id=analysis_id,
            )

            return result_id

        except Exception as e:
            logger.error(
                "Failed to start document analysis",
                document_id=document_id,
                analysis_id=analysis_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to start document analysis: {str(e)}")

    async def _process_analysis_background(
        self,
        result_id: str,
        document_id: str,
        analysis: Dict[str, Any],
        document: Dict[str, Any],
    ):
        """
        Background task to process each analysis item asynchronously.

        Args:
            result_id: Analysis result ID
            document_id: Document ID
            analysis: Analysis document from DB
            document: Document document from DB
        """
        try:
            logger.info("Starting background analysis processing", result_id=result_id)
            start_time = time.time()

            results_collection = get_analysis_results_collection()

            # Update status to processing
            await results_collection.update_one(
                {"_id": ObjectId(result_id)},
                {
                    "$set": {
                        "status": AnalysisResultStatus.PROCESSING,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            # Initialize query service
            query_service = QueryService()

            # Process each analysis item
            analysis_items = analysis.get("items", [])
            results = []

            # Initialize total usage
            total_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

            for item in analysis_items:
                try:
                    logger.info(
                        "Processing analysis item",
                        result_id=result_id,
                        question=item.get("question"),
                    )

                    # Query document using the analysis question
                    answer_text = ""
                    sources = {}
                    item_usage = {}

                    async for event in query_service.query_doc(
                        document_id=document_id,
                        query=item.get("question"),
                        user_id=analysis.get("user_id"),
                    ):
                        if event.get("type") == "text_delta":
                            answer_text += event.get("content", "")
                        elif event.get("type") == "final_result":
                            # Extract sources from references
                            references = event.get("references", {})
                            query_paths = references.get("query_paths", [])
                            retrieved_pages = references.get("retrieved_pages", [])

                            sources = {
                                "query_paths": query_paths,
                                "retrieved_pages": retrieved_pages,
                            }

                            # Accumulate usage
                            if "usage" in event:
                                item_usage = event["usage"]
                                total_usage["input_tokens"] += item_usage.get(
                                    "input_tokens", 0
                                )
                                total_usage["output_tokens"] += item_usage.get(
                                    "output_tokens", 0
                                )
                                total_usage["total_tokens"] += item_usage.get(
                                    "input_tokens", 0
                                ) + item_usage.get("output_tokens", 0)

                    # Evaluate the answer using the evaluation agent
                    evaluation_prompt = f"""
                    Question: {item.get("question")}
                    Context: {item.get("context", "No additional context provided.")}
                    Answer: {answer_text}

                    Evaluate if the answer satisfactorily addresses the question based on the provided context.
                    If the answer is empty or indicates that the information was not found, mark it as failed.
                    Provide a concise reason for your evaluation.
                    Scoring metrics:
                    0: Failed
                    1: There is non explicit evidence
                    2: Partially comply
                    3: Fully comply
                    """

                    evaluation_result = await self.evaluation_agent.run(
                        evaluation_prompt,
                        output_type=EvaluationResult,
                    )

                    # Accumulate evaluation usage
                    eval_usage = evaluation_result.usage()
                    total_usage["input_tokens"] += eval_usage.input_tokens
                    total_usage["output_tokens"] += eval_usage.output_tokens
                    total_usage["total_tokens"] += (
                        eval_usage.input_tokens + eval_usage.output_tokens
                    )

                    score = evaluation_result.output.score
                    reason = evaluation_result.output.reason

                    # Create result item
                    result_item = AnalysisResultItem(
                        question=item.get("question"),
                        score=score,
                        reason=reason,
                        context=item.get("context"),
                        sources=sources,
                    )

                    results.append(result_item.model_dump())

                    # Update progress
                    await results_collection.update_one(
                        {"_id": ObjectId(result_id)},
                        {
                            "$set": {
                                "results": results,
                                "completed_items": len(results),
                                "updated_at": datetime.utcnow(),
                            }
                        },
                    )

                    logger.info(
                        "Analysis item completed",
                        result_id=result_id,
                        question=item.get("question"),
                        completed=len(results),
                        total=len(analysis_items),
                    )
                except Exception as e:
                    logger.error(
                        "Failed to process analysis item",
                        result_id=result_id,
                        question=item.get("question"),
                        error=str(e),
                    )
                    result_item = AnalysisResultItem(
                        question=item.get("question"),
                        score=0,
                        reason=f"Failed to evaluate {e}",
                        context=item.get("context"),
                        sources={},
                    )

                    results.append(result_item.model_dump())
                # break

            processing_time = time.time() - start_time

            # Update final status
            await results_collection.update_one(
                {"_id": ObjectId(result_id)},
                {
                    "$set": {
                        "status": AnalysisResultStatus.COMPLETED,
                        "processing_time": processing_time,
                        "usage": total_usage,
                        "completed_at": datetime.utcnow(),
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            logger.info(
                "Background analysis processing completed",
                result_id=result_id,
                total_items=len(analysis_items),
                completed_items=len(results),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(
                "Background analysis processing failed",
                result_id=result_id,
                error=str(e),
            )

            # Update status to failed
            try:
                await results_collection.update_one(
                    {"_id": ObjectId(result_id)},
                    {
                        "$set": {
                            "status": AnalysisResultStatus.FAILED,
                            "error_message": str(e),
                            "updated_at": datetime.utcnow(),
                        }
                    },
                )
            except Exception:
                pass

    async def get_analysis_results(
        self,
        analysis_id: str,
        skip: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get analysis results for an analysis with pagination and search.

        Args:
            analysis_id: Analysis ID
            skip: Number of items to skip
            limit: Number of items to return
            search: Optional search term for document title

        Returns:
            Dictionary with results and pagination info
        """
        try:
            results_collection = get_analysis_results_collection()

            # Build filter
            filter_dict = {
                "analysis_id": analysis_id,
                "is_deleted": {"$ne": True},
            }

            # Add search
            if search:
                search_pattern = {"$regex": search, "$options": "i"}
                filter_dict["$or"] = [
                    {"document_title": search_pattern},
                    {"results.reason": search_pattern},
                    {"results.answer": search_pattern},  # Support legacy search
                ]

            # Get total count
            total = await results_collection.count_documents(filter_dict)

            # Get results
            cursor = (
                results_collection.find(filter_dict)
                .skip(skip)
                .limit(limit)
                .sort("created_at", -1)
            )
            results = await cursor.to_list(length=limit)

            # Convert ObjectIds to strings and compute score metadata
            for result in results:
                result["id"] = str(result["_id"])
                del result["_id"]

                scores = [
                    int(item.get("score", 0)) for item in result.get("results", [])
                ]
                total_items = int(
                    result.get("total_items", len(result.get("results", [])))
                )
                score_total = sum(scores)
                score_max = total_items * 3
                score_percentage = (
                    (score_total / score_max * 100) if score_max > 0 else 0.0
                )
                result["score_total"] = score_total
                result["score_max"] = score_max
                result["score_percentage"] = round(score_percentage, 2)

                result["results"] = []

            # Calculate pagination
            page = (skip // limit) + 1
            pages = (total + limit - 1) // limit

            return {
                "results": results,
                "total": total,
                "page": page,
                "size": limit,
                "pages": pages,
            }

        except Exception as e:
            logger.error(
                "Failed to get analysis results",
                analysis_id=analysis_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to get analysis results: {str(e)}")

    async def delete_analysis_result(
        self,
        analysis_id: str,
        document_id: str,
    ) -> bool:
        """
        Soft delete an analysis result.

        Args:
            analysis_id: Analysis ID
            document_id: Document ID

        Returns:
            True if deleted, False if not found
        """
        try:
            results_collection = get_analysis_results_collection()

            result = await results_collection.update_one(
                {
                    "analysis_id": analysis_id,
                    "document_id": document_id,
                },
                {
                    "$set": {
                        "is_deleted": True,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

            return result.modified_count > 0

        except Exception as e:
            logger.error(
                "Failed to delete analysis result",
                analysis_id=analysis_id,
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to delete analysis result: {str(e)}")

    async def get_analysis_result_by_ids(
        self,
        analysis_id: str,
        document_id: str,
    ) -> Dict[str, Any]:
        """
        Get analysis result by analysis ID and document ID.

        Args:
            analysis_id: Analysis ID
            document_id: Document ID

        Returns:
            Analysis result dictionary
        """
        try:
            results_collection = get_analysis_results_collection()

            # Find result
            result = await results_collection.find_one(
                {
                    "analysis_id": analysis_id,
                    "document_id": document_id,
                }
            )

            if not result:
                return None

            # Convert ObjectId to string
            result["id"] = str(result["_id"])
            del result["_id"]

            # Normalize results to match new schema
            if "results" in result:
                normalized_items = []
                for item in result["results"]:
                    # Handle legacy items
                    if "answer" in item and "reason" not in item:
                        item["reason"] = item["answer"]
                        # item.pop("answer", None)

                    # Handle legacy sources
                    if isinstance(item.get("sources"), list):
                        item["sources"] = {"legacy": item["sources"]}

                    # Calculate pass if not present (score >= 2 is pass)
                    if "pass" not in item and "score" in item:
                        item["pass"] = item["score"] >= 2
                    elif "pass" not in item:
                        # Default to false if no score or pass
                        item["pass"] = False

                    normalized_items.append(item)
                result["results"] = normalized_items

            return result

        except Exception as e:
            logger.error(
                "Failed to get analysis result by IDs",
                analysis_id=analysis_id,
                document_id=document_id,
                error=str(e),
            )
            raise ProcessingError(f"Failed to get analysis result: {str(e)}")

    async def get_analysis_stats(self) -> AnalysisStatsResponse:
        """
        Get overall analysis statistics across all analysis results.

        Returns:
            AnalysisStatsResponse
        """
        try:
            results_collection = get_analysis_results_collection()

            # Count distinct documents analyzed
            try:
                distinct_doc_ids = await results_collection.distinct("document_id")
                total_documents = len(distinct_doc_ids)
            except Exception:
                # Fallback: count all results (may overcount if multiple runs per document)
                total_documents = await results_collection.count_documents({})

            # Sum token usage and processing time across all results
            total_input = 0
            total_output = 0
            total_time = 0.0

            cursor = results_collection.find({}, {"usage": 1, "processing_time": 1})
            async for doc in cursor:
                usage = doc.get("usage") or {}
                total_input += int(usage.get("input_tokens", 0))
                total_output += int(usage.get("output_tokens", 0))
                try:
                    total_time += float(doc.get("processing_time", 0.0))
                except Exception:
                    pass

            return AnalysisStatsResponse(
                total_documents=total_documents,
                total_input_token_usage=total_input,
                total_output_token_usage=total_output,
                total_analysis_time=total_time,
            )

        except Exception as e:
            logger.error("Failed to get analysis stats", error=str(e))
            raise ProcessingError(f"Failed to get analysis stats: {str(e)}")


# Service instance
analysis_service = AnalysisService()
