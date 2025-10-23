"""
Pydantic AI service for intelligent query processing and response generation.
Handles type-safe query processing with structured outputs.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models import Model

from app.core.config import get_settings
from app.core.exceptions import ProcessingError, ExternalServiceError
from app.models.query import (
    QueryType,
    QueryScope,
    QueryContext,
    QueryEvidence,
    QueryResult,
    QuerySuggestion,
)
from app.services.gemini_service import gemini_service


logger = structlog.get_logger()


class QueryAnalysisResult(BaseModel):
    """Structured result for query analysis."""

    query_type: QueryType
    scope: QueryScope
    intent: str
    keywords: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    suggested_filters: Dict[str, Any] = Field(default_factory=dict)


class DocumentSearchResult(BaseModel):
    """Structured result for document search."""

    relevant_documents: List[str]
    relevance_scores: Dict[str, float]
    search_strategy: str
    total_documents_searched: int


class ContentExtractionResult(BaseModel):
    """Structured result for content extraction."""

    extracted_content: str
    source_pages: List[int]
    confidence_score: float = Field(ge=0.0, le=1.0)
    extraction_method: str
    content_type: str


class AnswerGenerationResult(BaseModel):
    """Structured result for answer generation."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence_used: List[str]
    limitations: Optional[str] = None


class QuerySuggestionResult(BaseModel):
    """Structured result for query suggestions."""

    suggestions: List[QuerySuggestion]
    suggestion_strategy: str
    context_used: List[str]


class PydanticAIQueryProcessor:
    """Pydantic AI-powered query processing engine."""

    def __init__(self):
        self.settings = get_settings()
        self._setup_agents()

    def _setup_agents(self):
        """Setup Pydantic AI agents for different query processing tasks."""
        try:
            # Query Analysis Agent
            self.query_analyzer = Agent(
                model=self._get_model(),
                result_type=QueryAnalysisResult,
                system_prompt="""
                You are an expert query analyzer for a document processing system.
                Analyze user queries to determine their type, scope, intent, and extract relevant keywords.
                
                Query Types:
                - factual: Seeking specific facts or information
                - analytical: Requiring analysis or interpretation
                - comparative: Comparing different concepts or items
                - procedural: Asking about processes or procedures
                - exploratory: Open-ended exploration of topics
                
                Query Scopes:
                - document: Query about a specific document
                - collection: Query across multiple documents
                - global: Query across entire system
                
                Provide structured analysis with high confidence scores for clear queries.
                """,
            )

            # Document Search Agent
            self.document_searcher = Agent(
                model=self._get_model(),
                result_type=DocumentSearchResult,
                system_prompt="""
                You are a document search specialist. Given a query and available documents,
                determine which documents are most relevant and provide relevance scores.
                
                Consider:
                - Query keywords and semantic meaning
                - Document titles and metadata
                - Content relevance
                - User intent
                
                Provide relevance scores between 0.0 and 1.0.
                """,
            )

            # Content Extraction Agent
            self.content_extractor = Agent(
                model=self._get_model(),
                result_type=ContentExtractionResult,
                system_prompt="""
                You are a content extraction specialist. Extract relevant content from documents
                based on user queries. Focus on accuracy and provide confidence scores.
                
                Extraction methods:
                - visual_analysis: Using visual document analysis
                - keyword_matching: Based on keyword relevance
                - semantic_search: Using semantic understanding
                - structural_analysis: Based on document structure
                
                Always specify the extraction method used and provide source page references.
                """,
            )

            # Answer Generation Agent
            self.answer_generator = Agent(
                model=self._get_model(),
                result_type=AnswerGenerationResult,
                system_prompt="""
                You are an expert answer generator. Create comprehensive, accurate answers
                based on extracted content and evidence.
                
                Guidelines:
                - Be factual and precise
                - Cite evidence clearly
                - Acknowledge limitations
                - Provide reasoning for conclusions
                - Maintain appropriate confidence levels
                
                Structure answers clearly and provide supporting evidence.
                """,
            )

            # Query Suggestion Agent
            self.suggestion_generator = Agent(
                model=self._get_model(),
                result_type=QuerySuggestionResult,
                system_prompt="""
                You are a query suggestion specialist. Generate helpful follow-up queries
                and related questions based on user context and document content.
                
                Suggestion strategies:
                - related_topics: Suggest queries about related topics
                - deeper_dive: Suggest more detailed questions
                - comparative: Suggest comparative questions
                - procedural: Suggest how-to questions
                - exploratory: Suggest open-ended exploration
                
                Provide diverse, relevant suggestions that help users explore content effectively.
                """,
            )

            logger.info("Pydantic AI agents initialized successfully")

        except Exception as e:
            logger.error("Failed to setup Pydantic AI agents", error=str(e))
            raise ExternalServiceError(f"Pydantic AI setup failed: {str(e)}")

    def _get_model(self) -> Model:
        """Get the configured AI model."""
        # For now, we'll use a simple model configuration
        # In a real implementation, this would configure the actual model
        return f"gemini/{self.settings.GEMINI_MODEL}"

    async def analyze_query(
        self, query: str, context: Optional[QueryContext] = None
    ) -> QueryAnalysisResult:
        """
        Analyze a user query to understand its type, scope, and intent.

        Args:
            query: User query string
            context: Optional query context

        Returns:
            Structured query analysis result
        """
        try:
            logger.info(
                "Analyzing query",
                query=query[:100] + "..." if len(query) > 100 else query,
                has_context=context is not None,
            )

            # Prepare context information
            context_info = ""
            if context:
                context_info = f"""
                Context Information:
                - Document ID: {context.document_id or "None"}
                - User ID: {context.user_id or "None"}
                - Session ID: {context.session_id or "None"}
                - Previous queries: {len(context.previous_queries)}
                """

            # Run analysis
            prompt = f"""
            Analyze the following query:
            
            Query: "{query}"
            {context_info}
            
            Provide a comprehensive analysis including query type, scope, intent, keywords, and confidence.
            """

            # For now, we'll simulate the Pydantic AI call with a structured response
            # In a real implementation, this would use the actual Pydantic AI agent
            result = await self._simulate_query_analysis(query, context)

            logger.info(
                "Query analysis completed",
                query_type=result.query_type,
                scope=result.scope,
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error("Query analysis failed", query=query[:50], error=str(e))
            raise ProcessingError(f"Query analysis failed: {str(e)}")

    async def search_documents(
        self,
        query: str,
        available_documents: List[Dict[str, Any]],
        analysis_result: QueryAnalysisResult,
    ) -> DocumentSearchResult:
        """
        Search for relevant documents based on query analysis.

        Args:
            query: User query
            available_documents: List of available documents
            analysis_result: Query analysis result

        Returns:
            Document search result
        """
        try:
            logger.info(
                "Searching documents",
                query=query[:50],
                document_count=len(available_documents),
                query_type=analysis_result.query_type,
            )

            # Simulate document search based on query analysis
            result = await self._simulate_document_search(
                query, available_documents, analysis_result
            )

            logger.info(
                "Document search completed",
                relevant_documents=len(result.relevant_documents),
                strategy=result.search_strategy,
            )

            return result

        except Exception as e:
            logger.error("Document search failed", error=str(e))
            raise ProcessingError(f"Document search failed: {str(e)}")

    async def extract_content(
        self, query: str, document_ids: List[str], analysis_result: QueryAnalysisResult
    ) -> ContentExtractionResult:
        """
        Extract relevant content from documents based on query.

        Args:
            query: User query
            document_ids: List of relevant document IDs
            analysis_result: Query analysis result

        Returns:
            Content extraction result
        """
        try:
            logger.info(
                "Extracting content",
                query=query[:50],
                document_count=len(document_ids),
                extraction_method="visual_analysis",
            )

            # Use Gemini service for visual content extraction
            extracted_content = ""
            source_pages = []
            confidence_scores = []

            for doc_id in document_ids:
                try:
                    # Get document images (this would be implemented in document service)
                    image_paths = await self._get_document_images(doc_id)

                    if image_paths:
                        # Use Gemini for content analysis
                        analysis = await gemini_service.analyze_specific_content(
                            image_paths=image_paths,
                            query=query,
                            context=f"Query type: {analysis_result.query_type}, Keywords: {', '.join(analysis_result.keywords)}",
                        )

                        extracted_content += (
                            f"\\n\\nFrom Document {doc_id}:\\n{analysis['answer']}"
                        )

                        # Extract page numbers from evidence
                        for evidence in analysis.get("evidence", []):
                            page_num = evidence.get("page_number")
                            if page_num and page_num not in source_pages:
                                source_pages.append(page_num)

                        confidence_scores.append(analysis.get("confidence_score", 0.5))

                except Exception as e:
                    logger.warning(
                        f"Failed to extract content from document {doc_id}",
                        error=str(e),
                    )
                    continue

            # Calculate overall confidence
            avg_confidence = (
                sum(confidence_scores) / len(confidence_scores)
                if confidence_scores
                else 0.0
            )

            result = ContentExtractionResult(
                extracted_content=extracted_content.strip(),
                source_pages=sorted(source_pages),
                confidence_score=avg_confidence,
                extraction_method="visual_analysis",
                content_type="mixed",
            )

            logger.info(
                "Content extraction completed",
                content_length=len(result.extracted_content),
                source_pages=len(result.source_pages),
                confidence=result.confidence_score,
            )

            return result

        except Exception as e:
            logger.error("Content extraction failed", error=str(e))
            raise ProcessingError(f"Content extraction failed: {str(e)}")

    async def generate_answer(
        self,
        query: str,
        extracted_content: ContentExtractionResult,
        analysis_result: QueryAnalysisResult,
    ) -> AnswerGenerationResult:
        """
        Generate a comprehensive answer based on extracted content.

        Args:
            query: User query
            extracted_content: Extracted content result
            analysis_result: Query analysis result

        Returns:
            Answer generation result
        """
        try:
            logger.info(
                "Generating answer",
                query=query[:50],
                content_length=len(extracted_content.extracted_content),
                query_type=analysis_result.query_type,
            )

            # Simulate answer generation based on content and analysis
            result = await self._simulate_answer_generation(
                query, extracted_content, analysis_result
            )

            logger.info(
                "Answer generation completed",
                answer_length=len(result.answer),
                confidence=result.confidence,
            )

            return result

        except Exception as e:
            logger.error("Answer generation failed", error=str(e))
            raise ProcessingError(f"Answer generation failed: {str(e)}")

    async def generate_suggestions(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        document_context: Optional[List[str]] = None,
    ) -> QuerySuggestionResult:
        """
        Generate query suggestions based on context.

        Args:
            query: Original query
            context: Query context
            document_context: Document context information

        Returns:
            Query suggestion result
        """
        try:
            logger.info(
                "Generating query suggestions",
                query=query[:50],
                has_context=context is not None,
                document_context_count=len(document_context) if document_context else 0,
            )

            # Simulate suggestion generation
            result = await self._simulate_suggestion_generation(
                query, context, document_context
            )

            logger.info(
                "Query suggestions generated",
                suggestion_count=len(result.suggestions),
                strategy=result.suggestion_strategy,
            )

            return result

        except Exception as e:
            logger.error("Suggestion generation failed", error=str(e))
            raise ProcessingError(f"Suggestion generation failed: {str(e)}")

    async def process_complete_query(
        self,
        query: str,
        context: Optional[QueryContext] = None,
        available_documents: Optional[List[Dict[str, Any]]] = None,
    ) -> QueryResult:
        """
        Process a complete query from analysis to answer generation.

        Args:
            query: User query
            context: Query context
            available_documents: Available documents for search

        Returns:
            Complete query result
        """
        try:
            start_time = datetime.utcnow()

            logger.info(
                "Processing complete query",
                query=query[:100] + "..." if len(query) > 100 else query,
                has_context=context is not None,
                document_count=len(available_documents) if available_documents else 0,
            )

            # Step 1: Analyze query
            analysis = await self.analyze_query(query, context)

            # Step 2: Search documents (if available)
            relevant_docs = []
            if available_documents:
                search_result = await self.search_documents(
                    query, available_documents, analysis
                )
                relevant_docs = search_result.relevant_documents

            # Step 3: Extract content
            content_result = None
            if relevant_docs:
                content_result = await self.extract_content(
                    query, relevant_docs, analysis
                )

            # Step 4: Generate answer
            answer_result = None
            if content_result:
                answer_result = await self.generate_answer(
                    query, content_result, analysis
                )

            # Step 5: Generate suggestions
            suggestions = await self.generate_suggestions(query, context, relevant_docs)

            # Compile final result
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            result = QueryResult(
                query=query,
                answer=answer_result.answer
                if answer_result
                else "No relevant content found.",
                confidence=answer_result.confidence if answer_result else 0.0,
                evidence=[
                    QueryEvidence(
                        source_type="document",
                        source_id=doc_id,
                        content_snippet=content_result.extracted_content[:200] + "..."
                        if content_result
                        else "",
                        relevance_score=0.8,
                        page_numbers=content_result.source_pages
                        if content_result
                        else [],
                    )
                    for doc_id in relevant_docs[:3]  # Top 3 documents
                ],
                suggestions=[
                    QuerySuggestion(
                        query=suggestion.query,
                        description=suggestion.description,
                        confidence=suggestion.confidence,
                    )
                    for suggestion in suggestions.suggestions[:5]  # Top 5 suggestions
                ],
                processing_time=processing_time,
                metadata={
                    "query_type": analysis.query_type,
                    "scope": analysis.scope,
                    "keywords": analysis.keywords,
                    "documents_searched": len(available_documents)
                    if available_documents
                    else 0,
                    "relevant_documents": len(relevant_docs),
                    "extraction_method": content_result.extraction_method
                    if content_result
                    else None,
                },
            )

            logger.info(
                "Complete query processing finished",
                processing_time=processing_time,
                confidence=result.confidence,
                evidence_count=len(result.evidence),
            )

            return result

        except Exception as e:
            logger.error("Complete query processing failed", error=str(e))
            raise ProcessingError(f"Query processing failed: {str(e)}")

    # Simulation methods (to be replaced with actual Pydantic AI calls)

    async def _simulate_query_analysis(
        self, query: str, context: Optional[QueryContext]
    ) -> QueryAnalysisResult:
        """Simulate query analysis (placeholder for actual Pydantic AI)."""

        # Simple heuristic-based analysis
        query_lower = query.lower()

        # Determine query type
        if any(
            word in query_lower for word in ["what", "who", "when", "where", "which"]
        ):
            query_type = QueryType.factual
        elif any(word in query_lower for word in ["why", "how", "analyze", "explain"]):
            query_type = QueryType.analytical
        elif any(
            word in query_lower for word in ["compare", "versus", "vs", "difference"]
        ):
            query_type = QueryType.comparative
        elif any(
            word in query_lower for word in ["process", "procedure", "steps", "how to"]
        ):
            query_type = QueryType.procedural
        else:
            query_type = QueryType.exploratory

        # Determine scope
        if context and context.document_id:
            scope = QueryScope.document
        elif any(word in query_lower for word in ["all", "every", "across", "overall"]):
            scope = QueryScope.global_scope
        else:
            scope = QueryScope.collection

        # Extract keywords (simple approach)
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }
        words = query_lower.split()
        keywords = [word for word in words if len(word) > 2 and word not in stop_words][
            :10
        ]

        return QueryAnalysisResult(
            query_type=query_type,
            scope=scope,
            intent=f"User wants to {query_type.value} information",
            keywords=keywords,
            confidence=0.8,
            suggested_filters={},
        )

    async def _simulate_document_search(
        self,
        query: str,
        available_documents: List[Dict[str, Any]],
        analysis_result: QueryAnalysisResult,
    ) -> DocumentSearchResult:
        """Simulate document search."""

        # Simple keyword-based relevance scoring
        relevant_docs = []
        relevance_scores = {}

        for doc in available_documents:
            doc_id = doc.get("id", "")
            title = doc.get("title", "").lower()
            description = doc.get("description", "").lower()

            # Calculate relevance score
            score = 0.0
            for keyword in analysis_result.keywords:
                if keyword in title:
                    score += 0.3
                if keyword in description:
                    score += 0.2

            if score > 0.1:
                relevant_docs.append(doc_id)
                relevance_scores[doc_id] = min(score, 1.0)

        # Sort by relevance
        relevant_docs.sort(key=lambda x: relevance_scores.get(x, 0), reverse=True)

        return DocumentSearchResult(
            relevant_documents=relevant_docs[:10],  # Top 10
            relevance_scores=relevance_scores,
            search_strategy="keyword_matching",
            total_documents_searched=len(available_documents),
        )

    async def _simulate_answer_generation(
        self,
        query: str,
        extracted_content: ContentExtractionResult,
        analysis_result: QueryAnalysisResult,
    ) -> AnswerGenerationResult:
        """Simulate answer generation."""

        # Simple answer generation based on content
        content = extracted_content.extracted_content

        if not content.strip():
            answer = "I couldn't find relevant information to answer your query."
            confidence = 0.1
        else:
            answer = f"Based on the document analysis, {content[:500]}..."
            confidence = min(extracted_content.confidence_score + 0.1, 1.0)

        return AnswerGenerationResult(
            answer=answer,
            confidence=confidence,
            reasoning=f"Answer generated using {extracted_content.extraction_method} from {len(extracted_content.source_pages)} pages",
            evidence_used=[f"Page {p}" for p in extracted_content.source_pages[:3]],
            limitations="Answer based on visual analysis of document images",
        )

    async def _simulate_suggestion_generation(
        self,
        query: str,
        context: Optional[QueryContext],
        document_context: Optional[List[str]],
    ) -> QuerySuggestionResult:
        """Simulate suggestion generation."""

        suggestions = [
            QuerySuggestion(
                query=f"What are the key points about {query.split()[-1] if query.split() else 'this topic'}?",
                description="Get a summary of key points",
                confidence=0.8,
            ),
            QuerySuggestion(
                query=f"How does {query.split()[-1] if query.split() else 'this'} relate to other topics?",
                description="Explore related concepts",
                confidence=0.7,
            ),
            QuerySuggestion(
                query=f"What are the implications of {query.split()[-1] if query.split() else 'this'}?",
                description="Understand broader implications",
                confidence=0.6,
            ),
        ]

        return QuerySuggestionResult(
            suggestions=suggestions,
            suggestion_strategy="related_topics",
            context_used=document_context[:3] if document_context else [],
        )

    async def _get_document_images(self, document_id: str) -> List[str]:
        """Get image paths for a document (placeholder)."""
        # This would be implemented to fetch actual document images
        # For now, return empty list
        return []


# Service instance - lazy initialization
_pydantic_ai_service = None

def get_pydantic_ai_service() -> PydanticAIQueryProcessor:
    """Get the Pydantic AI service instance with lazy initialization."""
    global _pydantic_ai_service
    if _pydantic_ai_service is None:
        _pydantic_ai_service = PydanticAIQueryProcessor()
    return _pydantic_ai_service

class LazyPydanticAIProcessor:
    """Lazy proxy for PydanticAIQueryProcessor to avoid initialization during import."""
    
    def __getattr__(self, name):
        # Delegate all attribute access to the actual service instance
        return getattr(get_pydantic_ai_service(), name)

# For backward compatibility
pydantic_ai_service = None  # Will be initialized when first accessed
pydantic_ai_processor = LazyPydanticAIProcessor()  # Lazy proxy instance
