"""
Main API router that includes all endpoint routers.
"""

from fastapi import APIRouter

from app.api.v1.documents import router as documents_router
from app.api.v1.query import router as query_router

# from app.api.v1.trees import router as trees_router
# from app.api.v1.queries import router as queries_router
from app.api.v1.users import router as users_router


# Create main API router
api_router = APIRouter()

# Include all routers
api_router.include_router(documents_router, prefix="/documents", tags=["documents"])
api_router.include_router(query_router, prefix="/query", tags=["query"])

# api_router.include_router(
#     trees_router,
#     prefix="/trees",
#     tags=["trees"]
# )

# api_router.include_router(
#     queries_router,
#     prefix="/queries",
#     tags=["queries"]
# )

api_router.include_router(users_router, prefix="/users", tags=["users"])
