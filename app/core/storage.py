"""
MinIO object storage connection and management.
"""

from typing import Optional, List
import structlog
from minio import Minio
from minio.error import S3Error, InvalidResponseError
import asyncio
from functools import partial
import base64

from app.core.config import get_settings
from app.core.exceptions import StorageError, ConfigurationError


logger = structlog.get_logger()

# Global MinIO client instance
_client: Optional[Minio] = None


async def init_storage() -> None:
    """Initialize MinIO storage connection."""
    global _client

    settings = get_settings()

    try:
        # Create MinIO client
        _client = Minio(
            settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )

        # Test connection by listing buckets
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _client.list_buckets)

        # Ensure required buckets exist
        buckets = [
            settings.MINIO_BUCKET_DOCUMENTS,
            settings.MINIO_BUCKET_IMAGES,
            settings.MINIO_BUCKET_PROCESSED,
        ]

        for bucket in buckets:
            bucket_exists = await loop.run_in_executor(
                None, _client.bucket_exists, bucket
            )
            if not bucket_exists:
                await loop.run_in_executor(None, _client.make_bucket, bucket)
                logger.info("Created MinIO bucket", bucket=bucket)

        logger.info(
            "Storage connection established",
            endpoint=settings.MINIO_ENDPOINT,
            buckets=buckets,
        )

    except (S3Error, InvalidResponseError) as e:
        logger.error("Failed to connect to MinIO", error=str(e))
        raise StorageError(f"Failed to connect to storage: {str(e)}")
    except Exception as e:
        logger.error("Unexpected storage initialization error", error=str(e))
        raise ConfigurationError(f"Storage configuration error: {str(e)}")


def get_storage_client() -> Minio:
    """Get MinIO client instance."""
    if _client is None:
        raise StorageError("Storage not initialized. Call init_storage() first.")
    return _client


async def upload_file_data(
    bucket: str,
    object_name: str,
    file_data: bytes,
    content_type: str = "application/octet-stream",
) -> str:
    """Upload file data to MinIO storage."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        from io import BytesIO

        file_stream = BytesIO(file_data)

        # Upload file data
        await loop.run_in_executor(
            None,
            partial(
                client.put_object,
                bucket,
                object_name,
                file_stream,
                len(file_data),
                content_type=content_type,
            ),
        )

        logger.info(
            "File data uploaded successfully",
            bucket=bucket,
            object_name=object_name,
            size=len(file_data),
        )

        return object_name

    except S3Error as e:
        logger.error(
            "Failed to upload file data",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise StorageError(f"Failed to upload file data: {str(e)}")


async def download_file(bucket: str, object_name: str, file_path: str) -> str:
    """Download a file from MinIO storage."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        # Download file
        await loop.run_in_executor(
            None, client.fget_object, bucket, object_name, file_path
        )

        logger.info(
            "File downloaded successfully",
            bucket=bucket,
            object_name=object_name,
            file_path=file_path,
        )

        return file_path

    except S3Error as e:
        logger.error(
            "Failed to download file",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise StorageError(f"Failed to download file: {str(e)}")


async def download_file_data(bucket: str, object_name: str) -> bytes:
    """Download file data from MinIO storage."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        # Download file data
        response = await loop.run_in_executor(
            None, client.get_object, bucket, object_name
        )

        file_data = response.read()
        response.close()
        response.release_conn()

        logger.info(
            "File data downloaded successfully",
            bucket=bucket,
            object_name=object_name,
            size=len(file_data),
        )

        return file_data

    except S3Error as e:
        logger.error(
            "Failed to download file data",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise StorageError(f"Failed to download file data: {str(e)}")


async def delete_file(bucket: str, object_name: str) -> bool:
    """Delete a file from MinIO storage."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        # Delete file
        await loop.run_in_executor(None, client.remove_object, bucket, object_name)

        logger.info("File deleted successfully", bucket=bucket, object_name=object_name)

        return True

    except S3Error as e:
        logger.error(
            "Failed to delete file",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise StorageError(f"Failed to delete file: {str(e)}")


async def list_files(bucket: str, prefix: str = "") -> List[str]:
    """List files in MinIO storage bucket."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        # List objects
        objects = await loop.run_in_executor(
            None, lambda: list(client.list_objects(bucket, prefix=prefix))
        )

        file_names = [obj.object_name for obj in objects]

        logger.info(
            "Files listed successfully",
            bucket=bucket,
            prefix=prefix,
            count=len(file_names),
        )

        return file_names

    except S3Error as e:
        logger.error("Failed to list files", bucket=bucket, prefix=prefix, error=str(e))
        raise StorageError(f"Failed to list files: {str(e)}")


async def file_exists(bucket: str, object_name: str) -> bool:
    """Check if a file exists in MinIO storage."""
    try:
        client = get_storage_client()
        loop = asyncio.get_event_loop()

        # Check if object exists
        await loop.run_in_executor(None, client.stat_object, bucket, object_name)

        return True

    except S3Error:
        return False
    except Exception as e:
        logger.error(
            "Error checking file existence",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        raise StorageError(f"Error checking file existence: {str(e)}")


async def get_page_image(bucket: str, object_name: str) -> Optional[str]:
    """Get image from MinIO and return base64 encoded string."""
    try:
        # Download the image data
        logger.info(f"Fetching document page at {bucket} and path {object_name}")
        image_data = await download_file_data(bucket, object_name)

        # Encode to base64
        base64_encoded = base64.b64encode(image_data).decode("utf-8")

        logger.info(
            "Image retrieved and encoded successfully",
            bucket=bucket,
            object_name=object_name,
            size=len(image_data),
        )

        return base64_encoded

    except StorageError as e:
        logger.error(
            "Failed to retrieve page image",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        return None
    except Exception as e:
        logger.error(
            "Unexpected error retrieving page image",
            bucket=bucket,
            object_name=object_name,
            error=str(e),
        )
        return None
