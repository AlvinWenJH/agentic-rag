"""
Document management API endpoints.
Handles document upload, processing, and retrieval.
"""

from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
import structlog
from datetime import datetime, timedelta
import io
from bson import ObjectId
from bson.errors import InvalidId

from app.models.document import (
    DocumentUpdate,
    DocumentResponse,
    DocumentListResponse,
    DocumentUploadResponse,
    DocumentProcessingStatus,
    DocumentStats,
    DocumentStatus,
    DocumentType,
)
from app.services.pdf_service import pdf_service
from app.services.gemini_service import gemini_service
from app.services.tree_merger_service import tree_merger_service
from app.core.database import get_documents_collection, get_tree_collection
from app.core.storage import upload_file_data, download_file_data
from app.core.exceptions import NotFoundError, ValidationError, ProcessingError
from app.core.config import get_settings


logger = structlog.get_logger()
router = APIRouter()
settings = get_settings()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    description: Optional[str] = None,
    user_id: Optional[str] = None,
):
    """
    Upload a document for processing.

    Args:
        file: PDF file to upload
        title: Optional document title
        description: Optional document description
        user_id: Optional user ID

    Returns:
        Document upload response with processing status
    """
    try:
        logger.info(
            "Document upload started",
            filename=file.filename,
            content_type=file.content_type,
            user_id=user_id,
        )

        # Validate file
        if not file.filename.lower().endswith(".pdf"):
            raise ValidationError("Only PDF files are supported")

        if file.content_type != "application/pdf":
            raise ValidationError("Invalid file type. Expected PDF.")

        # Read file content
        file_content = await file.read()

        # Validate PDF and get metadata
        pdf_metadata = await pdf_service.validate_pdf(file_content)

        # Create document record
        now = datetime.utcnow()
        document_data = {
            "title": title or file.filename,
            "description": description or "",
            "filename": file.filename,
            "content_type": "application/pdf",
            "file_size": len(file_content),
            "document_type": DocumentType.PDF,
            "status": DocumentStatus.UPLOADED,
            "user_id": user_id,
            "storage_path": "",  # Will be set after document_id is available
            "page_count": pdf_metadata["page_count"],
            "metadata": pdf_metadata,
            "tags": [],
            "created_at": now,
            "updated_at": now,
        }

        # Insert into database
        documents_collection = get_documents_collection()
        result = await documents_collection.insert_one(document_data)
        document_id = str(result.inserted_id)

        # Upload original PDF to storage
        pdf_path = f"documents/{document_id}/original.pdf"
        await upload_file_data(
            bucket=settings.MINIO_BUCKET_DOCUMENTS,
            object_name=pdf_path,
            file_data=file_content,
            content_type="application/pdf",
        )

        # Update document with storage path
        await documents_collection.update_one(
            {"_id": result.inserted_id}, {"$set": {"storage_path": pdf_path}}
        )

        # Schedule background processing
        background_tasks.add_task(
            process_document_background, document_id, file_content
        )

        logger.info(
            "Document uploaded successfully",
            document_id=document_id,
            filename=file.filename,
            page_count=pdf_metadata["page_count"],
        )

        return DocumentUploadResponse(
            id=document_id,
            title=title or file.filename,
            filename=file.filename,
            status=DocumentStatus.UPLOADED,
            message="Document uploaded successfully. Processing started.",
        )

    except Exception as e:
        logger.error("Document upload failed", filename=file.filename, error=str(e))
        if isinstance(e, (ValidationError, ProcessingError)):
            raise HTTPException(status_code=400, detail=str(e))
        raise HTTPException(status_code=500, detail="Document upload failed")


async def process_document_background(document_id: str, file_content: bytes):
    """Background task for document processing."""
    try:
        logger.info("Starting background document processing", document_id=document_id)
        start_time = datetime.utcnow()
        documents_collection = get_documents_collection()

        # Update status to processing
        await documents_collection.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "status": DocumentStatus.PROCESSING,
                    "processing_started": datetime.utcnow(),
                }
            },
        )

        # Convert PDF to images
        image_paths, page_count = await pdf_service.convert_pdf_to_images(
            pdf_data=file_content, document_id=document_id
        )

        # Update with image paths
        await documents_collection.update_one(
            {"_id": document_id}, {"$set": {"image_paths": image_paths}}
        )

        # Generate topic tree using Gemini
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise ProcessingError(f"Invalid document ID format: {document_id}")

        document = await documents_collection.find_one({"_id": object_id})
        if not document:
            raise ProcessingError(f"Document {document_id} not found during processing")

        tree_analysis = await gemini_service.analyze_document_images(
            image_paths=image_paths,
            document_id=document_id,
            document_title=document.get("title", ""),
        )

        # # Update document with processing results
        # await documents_collection.update_one(
        #     {"_id": document_id},
        #     {
        #         "$set": {
        #             "status": DocumentStatus.PROCESSED,
        #             "processing_completed": datetime.utcnow(),
        #             "processing_time": tree_analysis["processing_time"],
        #             "tree_data": tree_analysis["tree_data"],
        #             "analysis_metadata": tree_analysis,
        #         }
        #     },
        # )

        logger.info(
            "Document processing completed",
            document_id=document_id,
            processing_time=(datetime.utcnow() - start_time).total_seconds(),
        )

    except Exception as e:
        logger.error(
            "Document processing failed", document_id=document_id, error=str(e)
        )

        # Update status to failed
        documents_collection = get_documents_collection()
        await documents_collection.update_one(
            {"_id": document_id},
            {
                "$set": {
                    "status": DocumentStatus.FAILED,
                    "error_message": str(e),
                    "processing_completed": datetime.utcnow(),
                }
            },
        )


@router.get("/stats", response_model=DocumentStats)
async def get_document_stats():
    """Get document statistics."""
    try:
        documents_collection = get_documents_collection()

        # Get counts by status
        status_pipeline = [{"$group": {"_id": "$status", "count": {"$sum": 1}}}]
        status_counts = {}
        async for result in documents_collection.aggregate(status_pipeline):
            status_counts[result["_id"]] = result["count"]

        # Get counts by document type
        type_pipeline = [{"$group": {"_id": "$document_type", "count": {"$sum": 1}}}]
        type_counts = {}
        async for result in documents_collection.aggregate(type_pipeline):
            type_counts[result["_id"]] = result["count"]

        # Get total file size
        file_size_pipeline = [
            {"$group": {"_id": None, "total_size": {"$sum": "$file_size"}}}
        ]
        file_size_result = await documents_collection.aggregate(
            file_size_pipeline
        ).to_list(1)
        total_file_size = file_size_result[0]["total_size"] if file_size_result else 0

        # Get recent uploads (last 24 hours)
        twenty_four_hours_ago = datetime.utcnow() - timedelta(hours=24)
        recent_uploads = await documents_collection.count_documents(
            {"created_at": {"$gte": twenty_four_hours_ago}}
        )

        # Get average processing time
        processing_time_pipeline = [
            {"$match": {"processing_time": {"$exists": True, "$ne": None}}},
            {"$group": {"_id": None, "avg_time": {"$avg": "$processing_time"}}},
        ]
        processing_time_result = await documents_collection.aggregate(
            processing_time_pipeline
        ).to_list(1)
        average_processing_time = (
            processing_time_result[0]["avg_time"] if processing_time_result else None
        )

        # Get total documents count
        total_documents = await documents_collection.count_documents({})

        return DocumentStats(
            total_documents=total_documents,
            documents_by_status=status_counts,
            documents_by_type=type_counts,
            total_file_size=total_file_size,
            average_processing_time=average_processing_time,
            recent_uploads=recent_uploads,
        )

    except Exception as e:
        logger.error("Failed to get document stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get document statistics")


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get document by ID."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": object_id})

        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        # Convert ObjectId to string
        document["id"] = str(document["_id"])
        del document["_id"]

        return DocumentResponse(**document)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error("Failed to get document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve document")


@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    user_id: Optional[str] = Query(None),
    status: Optional[DocumentStatus] = Query(None),
    document_type: Optional[DocumentType] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List documents with optional filtering."""
    try:
        documents_collection = get_documents_collection()

        # Build filter
        filter_dict = {}
        if user_id:
            filter_dict["user_id"] = user_id
        if status:
            filter_dict["status"] = status
        if document_type:
            filter_dict["document_type"] = document_type

        # Get total count
        total = await documents_collection.count_documents(filter_dict)

        # Get documents
        cursor = (
            documents_collection.find(filter_dict)
            .skip(skip)
            .limit(limit)
            .sort("created_at", -1)
        )
        documents = await cursor.to_list(length=limit)

        # Convert ObjectIds to strings
        for doc in documents:
            doc["id"] = str(doc["_id"])
            del doc["_id"]

        # Calculate pagination
        page = (skip // limit) + 1
        pages = (total + limit - 1) // limit

        return DocumentListResponse(
            documents=[DocumentResponse(**doc) for doc in documents],
            total=total,
            page=page,
            size=limit,
            pages=pages,
        )

    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to list documents")


@router.get("/{document_id}/status", response_model=DocumentProcessingStatus)
async def get_document_status(document_id: str):
    """Get document processing status."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()
        document = await documents_collection.find_one(
            {"_id": object_id},
            {
                "status",
                "processing_started",
                "processing_completed",
                "error_message",
                "processing_time",
            },
        )

        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        return DocumentProcessingStatus(
            id=document_id,
            status=document["status"],
            error_message=document.get("error_message"),
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(
            "Failed to get document status", document_id=document_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to get document status")


@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(document_id: str, update_data: DocumentUpdate):
    """Update document metadata."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()

        # Check if document exists
        existing_doc = await documents_collection.find_one({"_id": object_id})
        if not existing_doc:
            raise NotFoundError(f"Document {document_id} not found")

        # Prepare update data
        update_dict = update_data.dict(exclude_unset=True)
        if update_dict:
            update_dict["updated_at"] = datetime.utcnow()

            # Update document
            await documents_collection.update_one(
                {"_id": object_id}, {"$set": update_dict}
            )

        # Get updated document
        updated_doc = await documents_collection.find_one({"_id": object_id})
        updated_doc["id"] = str(updated_doc["_id"])
        del updated_doc["_id"]

        return DocumentResponse(**updated_doc)

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error("Failed to update document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update document")


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete document and associated files."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()

        # Get document info
        document = await documents_collection.find_one({"_id": object_id})
        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        # Delete files from storage
        try:
            # Delete original PDF
            if document.get("file_path"):
                # Note: MinIO delete would be implemented here
                pass

            # Delete images
            if document.get("image_paths"):
                # Note: MinIO delete would be implemented here
                pass
        except Exception as e:
            logger.warning(
                "Failed to delete some files", document_id=document_id, error=str(e)
            )

        # Delete document record
        await documents_collection.delete_one({"_id": object_id})

        logger.info("Document deleted", document_id=document_id)

        return {"message": "Document deleted successfully"}

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error("Failed to delete document", document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to delete document")


@router.get("/{document_id}/download")
async def download_document(document_id: str):
    """Download original document file."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": object_id})

        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        file_path = document.get("file_path")
        if not file_path:
            raise NotFoundError("Document file not found")

        # Download file from storage
        file_data = await download_file_data(
            bucket=settings.MINIO_BUCKET_DOCUMENTS, object_name=file_path
        )

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={document['filename']}"
            },
        )

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document or file not found")
    except Exception as e:
        logger.error(
            "Failed to download document", document_id=document_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to download document")


@router.post("/{document_id}/merge-tree")
async def merge_document_tree(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    """
    Merge page subtrees into a complete document tree using Pydantic AI.

    This endpoint triggers the tree merging process that:
    1. Fetches all page subtrees for the document
    2. Uses Pydantic AI to generate JSON Patch operations for merging similar nodes
    3. Executes the merge patches to create a unified tree structure
    4. Saves the merged tree to MongoDB 'tree' collection

    Args:
        document_id: Document ID to merge subtrees for

    Returns:
        Tree merging status and metadata
    """
    try:
        logger.info(
            "Tree merging requested",
            document_id=document_id,
        )

        # Convert string ID to ObjectId and validate document exists
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": object_id})

        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        # Check if document has been processed (has subtrees)
        # if document.get("status") != DocumentStatus.PROCESSED:
        #     raise HTTPException(
        #         status_code=400,
        #         detail="Document must be processed before tree merging. Current status: " + str(document.get("status"))
        #     )

        # Schedule background tree merging
        background_tasks.add_task(merge_tree_background, document_id)

        logger.info(
            "Tree merging scheduled",
            document_id=document_id,
        )

        return {
            "document_id": document_id,
            "status": "merging_scheduled",
            "message": "Tree merging process started in background",
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(
            "Failed to start tree merging", document_id=document_id, error=str(e)
        )
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Failed to start tree merging")


async def merge_tree_background(document_id: str):
    """Background task for tree merging."""
    try:
        logger.info("Starting background tree merging", document_id=document_id)

        documents_collection = get_documents_collection()

        # Update document status to indicate tree merging in progress
        await documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "in_progress",
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        # Execute tree merging
        merge_result = await tree_merger_service.merge_document_tree(document_id)

        # Update document with tree merging results
        await documents_collection.update_one(
            {"_id": ObjectId(document_id)},
            {
                "$set": {
                    "status": "completed",
                    "updated_at": datetime.utcnow(),
                }
            },
        )

        logger.info(
            "Background tree merging completed",
            document_id=document_id,
        )

    except Exception as e:
        logger.error(
            "Background tree merging failed",
            document_id=document_id,
            error=str(e),
        )

        # Update document status to indicate failure
        try:
            await documents_collection.update_one(
                {"_id": ObjectId(document_id)},
                {
                    "$set": {
                        "status": "failed",
                        "updated_at": datetime.utcnow(),
                        "tree_merging_error": str(e),
                    }
                },
            )
        except Exception as update_error:
            logger.error(
                "Failed to update document with tree merging failure",
                document_id=document_id,
                error=str(update_error),
            )


@router.get("/{document_id}/tree")
async def get_document_tree(document_id: str):
    """
    Get the merged document tree.

    Args:
        document_id: Document ID to get tree for

    Returns:
        Merged document tree structure
    """
    try:
        from app.services.documents import get_document_tree_data

        return await get_document_tree_data(document_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document tree not found")
    except Exception as e:
        logger.error(
            "Failed to get document tree", document_id=document_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to get document tree")


@router.get("/{document_id}/tree/path")
async def get_document_tree_from_path(
    document_id: str,
    path: str = "/",
    depth: int = Query(3, ge=1, le=10),
    serialize: bool = False,
):
    """
    Get a subtree starting from a specific path with depth control.

    Args:
        document_id: Document ID to get tree for
        path: Path to the subtree (e.g., "/children/0/children/1" where "/" is root)
        depth: Maximum depth of the subtree to return (1-10)

    Returns:
        Subtree structure starting from the specified path
    """
    try:
        from app.services.documents import (
            get_document_tree_from_path as service_get_tree_from_path,
        )

        return await service_get_tree_from_path(document_id, path, depth, serialize)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document tree not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get document tree from path",
            document_id=document_id,
            path=path,
            depth=depth,
            error=str(e),
        )
        raise HTTPException(
            status_code=500, detail="Failed to get document tree from path"
        )


@router.get("/{document_id}/tree/stats")
async def get_document_tree_stats(document_id: str):
    """
    Get statistics about the document tree structure.

    Args:
        document_id: Document ID to get tree stats for

    Returns:
        Tree statistics including total nodes and counts by type (L1, L2, L3)
    """
    try:
        from app.services.documents import (
            get_document_tree_stats as service_get_tree_stats,
        )

        return await service_get_tree_stats(document_id)
    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document tree not found")
    except Exception as e:
        logger.error(
            "Failed to get document tree stats",
            document_id=document_id,
            error=str(e),
        )
        raise HTTPException(status_code=500, detail="Failed to get document tree stats")


@router.get("/{document_id}/images")
async def list_document_images(document_id: str):
    """List document page images."""
    try:
        # Convert string ID to ObjectId
        try:
            object_id = ObjectId(document_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid document ID format")

        documents_collection = get_documents_collection()
        document = await documents_collection.find_one({"_id": object_id})

        if not document:
            raise NotFoundError(f"Document {document_id} not found")

        image_paths = document.get("image_paths", [])

        return {
            "document_id": document_id,
            "image_count": len(image_paths),
            "image_paths": image_paths,
        }

    except NotFoundError:
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        logger.error(
            "Failed to list document images", document_id=document_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail="Failed to list document images")
