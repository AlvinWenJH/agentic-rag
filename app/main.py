"""
FastAPI main application for Vectorless RAG system.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
import structlog
import logging
import sys
import time

from app.core.config import get_settings
from app.core.database import init_database, close_database
from app.core.storage import init_storage
from app.core.cache import init_cache, close_cache
from app.api.router import api_router
from app.core.exceptions import VectorlessRAGException


def configure_logging():
    """Configure structured logging with proper level handling."""
    settings = get_settings()

    # Set up Python's standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    )

    # Configure processors based on log format
    if settings.LOG_FORMAT.lower() == "json":
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        # Console format for development
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=False),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Configure logging on import
configure_logging()

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    settings = get_settings()

    # Startup
    logger.info("Starting Vectorless RAG application", version="0.1.0")

    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized successfully")

        # Initialize storage
        await init_storage()
        logger.info("Storage initialized successfully")

        # Initialize cache
        await init_cache()
        logger.info("Cache initialized successfully")

        logger.info("Application startup completed")

    except Exception as e:
        logger.error("Failed to initialize application", error=str(e))
        raise

    yield

    # Shutdown
    logger.info("Shutting down Vectorless RAG application")

    try:
        await close_cache()
        await close_database()
        logger.info("Application shutdown completed")
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Vectorless RAG API",
        description="A visual-first document processing and querying system without traditional vector embeddings",
        version="0.1.0",
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        lifespan=lifespan,
    )

    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_HOSTS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

    # Add request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
        )

        response = await call_next(request)

        # Log response
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            process_time=round(process_time, 4),
        )

        return response

    # Add exception handler
    @app.exception_handler(VectorlessRAGException)
    async def vectorless_rag_exception_handler(
        request: Request, exc: VectorlessRAGException
    ):
        logger.error(
            "Application error",
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            status_code=exc.status_code,
            url=str(request.url),
        )

        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.__class__.__name__,
                "message": str(exc),
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled error",
            error_type=exc.__class__.__name__,
            error_message=str(exc),
            url=str(request.url),
        )

        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An unexpected error occurred",
            },
        )

    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "vectorless-rag", "version": "0.1.0"}

    # Include API router
    app.include_router(api_router, prefix="/api/v1")

    return app


# Create app instance
app = create_app()
