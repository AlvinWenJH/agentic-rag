"""
PyPDF service for PDF to image conversion only.
Visual-first approach - no text extraction.
"""

import asyncio
from typing import List, Tuple, Optional
import structlog
import fitz  # PyMuPDF
from PIL import Image
import io

from app.core.config import get_settings
from app.core.exceptions import ProcessingError, ValidationError
from app.core.storage import upload_file_data


logger = structlog.get_logger()


class PDFImageConverter:
    """PDF to image conversion service using PyMuPDF."""

    def __init__(self):
        self.settings = get_settings()

    async def convert_pdf_to_images(
        self,
        pdf_data: bytes,
        document_id: str,
        dpi: Optional[int] = None,
        image_format: str = "PNG",
    ) -> Tuple[List[str], int]:
        """
        Convert PDF to images and upload to MinIO.

        Args:
            pdf_data: PDF file data
            document_id: Document ID for storage path
            dpi: Image resolution (default from settings)
            image_format: Output image format

        Returns:
            Tuple of (image_paths, page_count)
        """
        dpi = dpi or self.settings.PDF_DPI

        try:
            logger.info(
                "Starting PDF to image conversion",
                document_id=document_id,
                dpi=dpi,
                format=image_format,
            )

            # Run conversion in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            image_paths, page_count = await loop.run_in_executor(
                None, self._convert_pdf_sync, pdf_data, document_id, dpi, image_format
            )

            logger.info(
                "PDF conversion completed",
                document_id=document_id,
                page_count=page_count,
                images_generated=len(image_paths),
            )

            return image_paths, page_count

        except Exception as e:
            logger.error("PDF conversion failed", document_id=document_id, error=str(e))
            raise ProcessingError(f"Failed to convert PDF to images: {str(e)}")

    def _convert_pdf_sync(
        self, pdf_data: bytes, document_id: str, dpi: int, image_format: str
    ) -> Tuple[List[str], int]:
        """Synchronous PDF conversion."""
        image_paths = []

        try:
            # Open PDF document
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            page_count = pdf_document.page_count

            logger.info(
                "PDF opened successfully",
                document_id=document_id,
                page_count=page_count,
            )

            # Convert each page to image
            for page_num in range(page_count):
                try:
                    page = pdf_document[page_num]

                    # Create transformation matrix for DPI
                    zoom = dpi / 72.0  # 72 DPI is default
                    mat = fitz.Matrix(zoom, zoom)

                    # Render page to pixmap
                    pix = page.get_pixmap(matrix=mat)

                    # Convert to PIL Image
                    img_data = pix.tobytes("png")
                    image = Image.open(io.BytesIO(img_data))

                    # Optimize image if needed
                    if image_format.upper() == "JPEG":
                        # Convert RGBA to RGB for JPEG
                        if image.mode == "RGBA":
                            background = Image.new("RGB", image.size, (255, 255, 255))
                            background.paste(image, mask=image.split()[-1])
                            image = background

                    # Save image to bytes
                    img_buffer = io.BytesIO()
                    image.save(
                        img_buffer,
                        format=image_format,
                        quality=self.settings.IMAGE_QUALITY
                        if image_format.upper() == "JPEG"
                        else None,
                        optimize=True,
                    )
                    img_bytes = img_buffer.getvalue()

                    # Upload to MinIO
                    image_path = (
                        f"{document_id}/page_{page_num + 1:04d}.{image_format.lower()}"
                    )

                    # Use asyncio to upload (this is still in thread pool)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    try:
                        loop.run_until_complete(
                            upload_file_data(
                                bucket=self.settings.MINIO_BUCKET_IMAGES,
                                object_name=image_path,
                                file_data=img_bytes,
                                content_type=f"image/{image_format.lower()}",
                            )
                        )
                        image_paths.append(image_path)

                        logger.debug(
                            "Page converted successfully",
                            document_id=document_id,
                            page_num=page_num + 1,
                            image_path=image_path,
                            image_size=len(img_bytes),
                        )

                    finally:
                        loop.close()

                except Exception as e:
                    logger.error(
                        "Failed to convert page",
                        document_id=document_id,
                        page_num=page_num + 1,
                        error=str(e),
                    )
                    # Continue with other pages
                    continue

            pdf_document.close()

            if not image_paths:
                raise ProcessingError("No pages could be converted to images")

            return image_paths, page_count

        except Exception as e:
            logger.error("PDF conversion error", document_id=document_id, error=str(e))
            raise

    async def validate_pdf(self, pdf_data: bytes) -> dict:
        """
        Validate PDF file and extract basic metadata.

        Args:
            pdf_data: PDF file data

        Returns:
            Dictionary with PDF metadata
        """
        try:
            loop = asyncio.get_event_loop()
            metadata = await loop.run_in_executor(
                None, self._validate_pdf_sync, pdf_data
            )

            return metadata

        except Exception as e:
            logger.error("PDF validation failed", error=str(e))
            raise ValidationError(f"Invalid PDF file: {str(e)}")

    def _validate_pdf_sync(self, pdf_data: bytes) -> dict:
        """Synchronous PDF validation."""
        try:
            # Try to open PDF
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            # Extract metadata
            metadata = {
                "page_count": pdf_document.page_count,
                "is_encrypted": pdf_document.needs_pass,
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "creator": pdf_document.metadata.get("creator", ""),
                "producer": pdf_document.metadata.get("producer", ""),
                "creation_date": pdf_document.metadata.get("creationDate", ""),
                "modification_date": pdf_document.metadata.get("modDate", ""),
                "file_size": len(pdf_data),
            }

            # Check if PDF is password protected
            if pdf_document.needs_pass:
                pdf_document.close()
                raise ValidationError("Password-protected PDFs are not supported")

            # Validate page count
            if pdf_document.page_count == 0:
                pdf_document.close()
                raise ValidationError("PDF contains no pages")

            # Check for very large documents
            max_pages = 500  # Configurable limit
            if pdf_document.page_count > max_pages:
                pdf_document.close()
                raise ValidationError(f"PDF has too many pages (max: {max_pages})")

            pdf_document.close()

            logger.info(
                "PDF validation successful",
                page_count=metadata["page_count"],
                file_size=metadata["file_size"],
                title=metadata["title"],
            )

            return metadata

        except fitz.FileDataError as e:
            raise ValidationError(f"Corrupted or invalid PDF file: {str(e)}")
        except Exception as e:
            raise ValidationError(f"PDF validation error: {str(e)}")

    async def get_page_dimensions(
        self, pdf_data: bytes, page_num: int = 0
    ) -> Tuple[float, float]:
        """
        Get dimensions of a specific page.

        Args:
            pdf_data: PDF file data
            page_num: Page number (0-indexed)

        Returns:
            Tuple of (width, height) in points
        """
        try:
            loop = asyncio.get_event_loop()
            dimensions = await loop.run_in_executor(
                None, self._get_page_dimensions_sync, pdf_data, page_num
            )

            return dimensions

        except Exception as e:
            logger.error(
                "Failed to get page dimensions", page_num=page_num, error=str(e)
            )
            raise ProcessingError(f"Failed to get page dimensions: {str(e)}")

    def _get_page_dimensions_sync(
        self, pdf_data: bytes, page_num: int
    ) -> Tuple[float, float]:
        """Synchronous page dimension extraction."""
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")

            if page_num >= pdf_document.page_count:
                pdf_document.close()
                raise ValidationError(f"Page {page_num} does not exist")

            page = pdf_document[page_num]
            rect = page.rect

            pdf_document.close()

            return rect.width, rect.height

        except Exception as e:
            raise ProcessingError(f"Error getting page dimensions: {str(e)}")


# Service instance
pdf_service = PDFImageConverter()
