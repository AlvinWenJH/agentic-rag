"""
Analysis API endpoints.
Handles analysis CRUD operations, draft generation, and document analysis.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Query
import structlog
from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId

from app.models.analysis import (
    AnalysisCreate,
    AnalysisUpdate,
    AnalysisResponse,
    AnalysisListResponse,
    AnalysisStatus,
    DraftAnalysisRequest,
    DraftAnalysisResponse,
    DocumentAnalysisRequest,
    DocumentAnalysisResponse,
    AnalysisResultResponse,
    AnalysisResultStatus,
)
from app.services.analysis import analysis_service
from app.core.database import get_analysis_collection, get_analysis_results_collection
from app.core.exceptions import NotFoundError


logger = structlog.get_logger()
router = APIRouter()


@router.post("/", response_model=AnalysisResponse)
async def create_analysis(analysis_data: AnalysisCreate):
    """
    Create a new analysis.

    Args:
        analysis_data: Analysis creation data

    Returns:
        Created analysis response
    """
    try:
        logger.info(
            "Creating analysis",
            title=analysis_data.title,
            items_count=len(analysis_data.items),
        )

        # Create analysis record
        now = datetime.utcnow()
        analysis_dict = analysis_data.dict()
        analysis_dict.update(
            {
                "status": AnalysisStatus.DRAFT,
                "created_at": now,
                "updated_at": now,
            }
        )

        # Insert into database
        analysis_collection = get_analysis_collection()
        result = await analysis_collection.insert_one(analysis_dict)
        analysis_id = str(result.inserted_id)

        # Fetch created analysis
        created_analysis = await analysis_collection.find_one(
            {"_id": result.inserted_id}
        )
        created_analysis["id"] = str(created_analysis["_id"])
        del created_analysis["_id"]

        logger.info("Analysis created successfully", analysis_id=analysis_id)

        return AnalysisResponse(**created_analysis)

    except Exception as e:
        logger.error("Failed to create analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to create analysis")


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Get analysis by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(analysis_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")

        analysis_collection = get_analysis_collection()
        analysis = await analysis_collection.find_one({"_id": object_id})

        if not analysis:
            raise NotFoundError(f"Analysis {analysis_id} not found")

        # Convert ObjectId to string
        analysis["id"] = str(analysis["_id"])
        del analysis["_id"]

        return AnalysisResponse(**analysis)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        logger.error("Failed to get analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


@router.get("/", response_model=AnalysisListResponse)
async def list_analyses(
    user_id: Optional[str] = Query(None),
    status: Optional[AnalysisStatus] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List analyses with optional filtering."""
    try:
        analysis_collection = get_analysis_collection()

        # Build filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if status:
            filter_dict["status"] = status

        # Get total count
        total = await analysis_collection.count_documents(filter_dict)

        # Get analyses
        cursor = (
            analysis_collection.find(filter_dict)
            .skip(skip)
            .limit(limit)
            .sort("created_at", -1)
        )
        analyses = await cursor.to_list(length=limit)

        # Convert ObjectIds to strings
        for analysis in analyses:
            analysis["id"] = str(analysis["_id"])
            del analysis["_id"]

        # Calculate pagination
        page = (skip // limit) + 1
        pages = (total + limit - 1) // limit

        return AnalysisListResponse(
            analyses=[AnalysisResponse(**a) for a in analyses],
            total=total,
            page=page,
            size=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error("Failed to list analyses", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list analyses")


@router.put("/{analysis_id}", response_model=AnalysisResponse)
async def update_analysis(analysis_id: str, update_data: AnalysisUpdate):
    """Update analysis."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(analysis_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")

        analysis_collection = get_analysis_collection()

        # Check if analysis exists
        existing_analysis = await analysis_collection.find_one({"_id": object_id})
        if not existing_analysis:
            raise NotFoundError(f"Analysis {analysis_id} not found")

        # Prepare update data
        update_dict = update_data.dict(exclude_unset=True)
        if update_dict:
            update_dict["updated_at"] = datetime.utcnow()

            # Update analysis
            await analysis_collection.update_one(
                {"_id": object_id}, {"$set": update_dict}
            )

        # Get updated analysis
        updated_analysis = await analysis_collection.find_one({"_id": object_id})
        updated_analysis["id"] = str(updated_analysis["_id"])
        del updated_analysis["_id"]

        return AnalysisResponse(**updated_analysis)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        logger.error("Failed to update analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update analysis")


@router.delete("/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(analysis_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid analysis ID format")

        analysis_collection = get_analysis_collection()

        # Check if analysis exists
        analysis = await analysis_collection.find_one({"_id": object_id})
        if not analysis:
            raise NotFoundError(f"Analysis {analysis_id} not found")

        # Delete analysis
        await analysis_collection.delete_one({"_id": object_id})

        logger.info("Analysis deleted", analysis_id=analysis_id)

        return {"message": "Analysis deleted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        logger.error("Failed to delete analysis", analysis_id=analysis_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete analysis")


@router.post("/draft", response_model=DraftAnalysisResponse)
async def generate_draft_analysis(request: DraftAnalysisRequest):
    """
    Generate draft analysis from free text using structured LLM output.

    Args:
        request: Draft analysis request with free text

    Returns:
        Structured draft analysis with title, description, and analysis items
    """
    try:
        logger.info(
            "Generating draft analysis",
            text_length=len(request.text),
            user_id=request.user_id,
        )

        # Generate draft analysis using AI service
        draft_analysis = await analysis_service.generate_draft_analysis(request.text)

        logger.info(
            "Draft analysis generated",
            title=draft_analysis.title,
            items_count=len(draft_analysis.items),
        )

        return draft_analysis

    except Exception as e:
        logger.error("Failed to generate draft analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to generate draft analysis")


@router.post("/analyze", response_model=DocumentAnalysisResponse)
async def analyze_document(request: DocumentAnalysisRequest):
    """
    Analyze document using analysis items.

    This endpoint processes each analysis item asynchronously:
    1. Queries the document using the query API for each item
    2. Uses LLM with Pydantic AI structured output to generate answers
    3. Saves results to MongoDB 'analysis_results' collection

    Args:
        request: Document analysis request

    Returns:
        Analysis processing status
    """
    try:
        logger.info(
            "Starting document analysis",
            document_id=request.document_id,
            analysis_id=request.analysis_id,
            user_id=request.user_id,
        )

        # Start analysis processing
        result_id = await analysis_service.process_document_analysis(
            document_id=request.document_id,
            analysis_id=request.analysis_id,
            user_id=request.user_id,
        )

        logger.info(
            "Document analysis started",
            result_id=result_id,
            document_id=request.document_id,
            analysis_id=request.analysis_id,
        )

        return DocumentAnalysisResponse(
            id=result_id,
            document_id=request.document_id,
            analysis_id=request.analysis_id,
            status=AnalysisResultStatus.PENDING,
            message="Document analysis started in background",
            created_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.error("Failed to start document analysis", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to start document analysis")


@router.get("/results/{result_id}", response_model=AnalysisResultResponse)
async def get_analysis_result(result_id: str):
    """Get analysis result by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(result_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid result ID format")

        results_collection = get_analysis_results_collection()
        result = await results_collection.find_one({"_id": object_id})

        if not result:
            raise NotFoundError(f"Analysis result {result_id} not found")

        # Convert ObjectId to string
        result["id"] = str(result["_id"])
        del result["_id"]

        return AnalysisResultResponse(**result)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Analysis result not found")
    except Exception as e:
        logger.error("Failed to get analysis result", result_id=result_id, error=str(e))
        raise HTTPException(
            status_code=500, detail="Failed to retrieve analysis result"
        )


@router.get("/results/document/{document_id}")
async def list_document_analysis_results(
    document_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List analysis results for a document."""
    try:
        results_collection = get_analysis_results_collection()

        # Build filter
        filter_dict = {"document_id": document_id}

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

        # Convert ObjectIds to strings
        for result in results:
            result["id"] = str(result["_id"])
            del result["_id"]

        # Calculate pagination
        page = (skip // limit) + 1
        pages = (total + limit - 1) // limit

        return {
            "results": [AnalysisResultResponse(**r) for r in results],
            "total": total,
            "page": page,
            "size": limit,
            "pages": pages,
        }

    except Exception as e:
        logger.error(
            "Failed to list analysis results",
            document_id=document_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to list analysis results")
