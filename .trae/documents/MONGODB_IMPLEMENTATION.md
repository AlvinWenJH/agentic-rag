# MongoDB Implementation Guide

## Overview

This document provides comprehensive implementation guidance for MongoDB integration in the Vectorless RAG system. Based on **Section 5.1** and **Section 7** of the PRD, MongoDB serves as the primary database for storing document metadata, hierarchical topic trees, and system data with high performance and scalability.

### Purpose and Role

MongoDB integration serves as:
- **Document Store**: Store document metadata, processing status, and configurations
- **Tree Repository**: Persist hierarchical topic trees with complex nested structures
- **Query Engine**: Support efficient tree traversal and search operations
- **Audit System**: Maintain change history and patch operations
- **Session Manager**: Handle user sessions and authentication data

## Technical Specifications

### Core Requirements

Based on **Section 5.1** and **Section 7** deployment architecture:
- **Version**: MongoDB 7.0+ for optimal performance
- **Driver**: Motor (async MongoDB driver) for Python
- **Replication**: MongoDB replica set for production high availability
- **Indexing**: Optimized indexes for tree traversal and document queries
- **Aggregation**: Advanced aggregation pipelines for complex queries
- **Transactions**: ACID transactions for data consistency

### Database Architecture

```mermaid
graph TD
    A[FastAPI Application] --> B[Motor Driver]
    B --> C[MongoDB Primary]
    C --> D[MongoDB Secondary 1]
    C --> E[MongoDB Secondary 2]
    
    F[Document Collection] --> G[Tree Collection]
    G --> H[User Collection]
    H --> I[Patch History Collection]
    I --> J[Session Collection]
    
    subgraph "MongoDB Replica Set"
        C
        D
        E
    end
    
    subgraph "Database Collections"
        F
        G
        H
        I
        J
    end
    
    K[Indexes] --> L[Tree Traversal Index]
    L --> M[Document Search Index]
    M --> N[User Lookup Index]
    N --> O[Temporal Index]
```

## Implementation Details

### 1. MongoDB Service Configuration

**app/services/mongodb_service.py**
```python
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import asyncio

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING, TEXT
from pymongo.errors import DuplicateKeyError, ConnectionFailure
from bson import ObjectId
from bson.errors import InvalidId

from app.core.config import settings
from app.models.document import Document, DocumentStatus
from app.models.tree import TreeStructure, TreeNode
from app.models.user import User

logger = logging.getLogger(__name__)

class MongoDBService:
    """
    MongoDB service for the Vectorless RAG system
    as specified in Section 5.1 and Section 7 of the PRD.
    """
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.collections: Dict[str, AsyncIOMotorCollection] = {}
        
        # Collection names
        self.DOCUMENTS_COLLECTION = "documents"
        self.TREES_COLLECTION = "trees"
        self.USERS_COLLECTION = "users"
        self.PATCH_HISTORY_COLLECTION = "patch_history"
        self.SESSIONS_COLLECTION = "sessions"
        self.QUERY_LOGS_COLLECTION = "query_logs"
    
    async def connect(self) -> None:
        """Initialize MongoDB connection and setup collections."""
        try:
            # Create MongoDB client
            self.client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=settings.MONGODB_MAX_CONNECTIONS,
                minPoolSize=settings.MONGODB_MIN_CONNECTIONS,
                maxIdleTimeMS=settings.MONGODB_MAX_IDLE_TIME,
                serverSelectionTimeoutMS=settings.MONGODB_TIMEOUT
            )
            
            # Get database
            self.database = self.client[settings.MONGODB_DATABASE]
            
            # Setup collections
            await self._setup_collections()
            
            # Create indexes
            await self._create_indexes()
            
            # Verify connection
            await self.client.admin.command('ping')
            
            logger.info("Successfully connected to MongoDB")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error initializing MongoDB: {str(e)}")
            raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")
    
    async def _setup_collections(self) -> None:
        """Setup MongoDB collections."""
        collection_names = [
            self.DOCUMENTS_COLLECTION,
            self.TREES_COLLECTION,
            self.USERS_COLLECTION,
            self.PATCH_HISTORY_COLLECTION,
            self.SESSIONS_COLLECTION,
            self.QUERY_LOGS_COLLECTION
        ]
        
        for collection_name in collection_names:
            self.collections[collection_name] = self.database[collection_name]
        
        logger.info("MongoDB collections setup completed")
    
    async def _create_indexes(self) -> None:
        """Create optimized indexes for efficient queries."""
        try:
            # Documents collection indexes
            await self.collections[self.DOCUMENTS_COLLECTION].create_indexes([
                IndexModel([("document_id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("status", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("filename", TEXT)]),
                IndexModel([("user_id", ASCENDING), ("created_at", DESCENDING)])
            ])
            
            # Trees collection indexes
            await self.collections[self.TREES_COLLECTION].create_indexes([
                IndexModel([("document_id", ASCENDING)], unique=True),
                IndexModel([("root_topics.id", ASCENDING)]),
                IndexModel([("root_topics.title", TEXT)]),
                IndexModel([("root_topics.keywords", ASCENDING)]),
                IndexModel([("root_topics.level", ASCENDING)]),
                IndexModel([("updated_at", DESCENDING)]),
                # Compound index for tree traversal
                IndexModel([
                    ("document_id", ASCENDING),
                    ("root_topics.level", ASCENDING),
                    ("root_topics.relevance_score", DESCENDING)
                ])
            ])
            
            # Users collection indexes
            await self.collections[self.USERS_COLLECTION].create_indexes([
                IndexModel([("user_id", ASCENDING)], unique=True),
                IndexModel([("email", ASCENDING)], unique=True),
                IndexModel([("created_at", DESCENDING)])
            ])
            
            # Patch history collection indexes
            await self.collections[self.PATCH_HISTORY_COLLECTION].create_indexes([
                IndexModel([("patch_id", ASCENDING)], unique=True),
                IndexModel([("document_id", ASCENDING)]),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("document_id", ASCENDING), ("created_at", DESCENDING)])
            ])
            
            # Sessions collection indexes
            await self.collections[self.SESSIONS_COLLECTION].create_indexes([
                IndexModel([("session_id", ASCENDING)], unique=True),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("expires_at", ASCENDING)], expireAfterSeconds=0)
            ])
            
            # Query logs collection indexes
            await self.collections[self.QUERY_LOGS_COLLECTION].create_indexes([
                IndexModel([("query_id", ASCENDING)], unique=True),
                IndexModel([("document_id", ASCENDING)]),
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("query_text", TEXT)])
            ])
            
            logger.info("MongoDB indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            raise
    
    # Document Management Methods
    
    async def create_document(self, document_data: Dict[str, Any]) -> str:
        """Create a new document record."""
        try:
            document_data["created_at"] = datetime.utcnow()
            document_data["updated_at"] = datetime.utcnow()
            
            result = await self.collections[self.DOCUMENTS_COLLECTION].insert_one(document_data)
            
            logger.info(f"Created document: {document_data['document_id']}")
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            logger.error(f"Document already exists: {document_data['document_id']}")
            raise ValueError(f"Document {document_data['document_id']} already exists")
        except Exception as e:
            logger.error(f"Error creating document: {str(e)}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by ID."""
        try:
            document = await self.collections[self.DOCUMENTS_COLLECTION].find_one(
                {"document_id": document_id}
            )
            
            if document:
                document["_id"] = str(document["_id"])
            
            return document
            
        except Exception as e:
            logger.error(f"Error retrieving document {document_id}: {str(e)}")
            raise
    
    async def update_document_status(
        self, 
        document_id: str, 
        status: DocumentStatus,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update document processing status."""
        try:
            update_data = {
                "status": status.value,
                "updated_at": datetime.utcnow()
            }
            
            if additional_data:
                update_data.update(additional_data)
            
            result = await self.collections[self.DOCUMENTS_COLLECTION].update_one(
                {"document_id": document_id},
                {"$set": update_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated document {document_id} status to {status.value}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating document status: {str(e)}")
            raise
    
    async def list_user_documents(
        self, 
        user_id: str, 
        limit: int = 50, 
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List documents for a specific user."""
        try:
            cursor = self.collections[self.DOCUMENTS_COLLECTION].find(
                {"user_id": user_id}
            ).sort("created_at", DESCENDING).skip(offset).limit(limit)
            
            documents = []
            async for doc in cursor:
                doc["_id"] = str(doc["_id"])
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error listing user documents: {str(e)}")
            raise
    
    # Tree Management Methods
    
    async def store_document_tree(self, tree_data: Dict[str, Any]) -> str:
        """Store hierarchical topic tree for a document."""
        try:
            tree_data["created_at"] = datetime.utcnow()
            tree_data["updated_at"] = datetime.utcnow()
            
            # Upsert operation to handle updates
            result = await self.collections[self.TREES_COLLECTION].replace_one(
                {"document_id": tree_data["document_id"]},
                tree_data,
                upsert=True
            )
            
            logger.info(f"Stored tree for document: {tree_data['document_id']}")
            return str(result.upserted_id) if result.upserted_id else "updated"
            
        except Exception as e:
            logger.error(f"Error storing document tree: {str(e)}")
            raise
    
    async def get_document_tree(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve hierarchical topic tree for a document."""
        try:
            tree = await self.collections[self.TREES_COLLECTION].find_one(
                {"document_id": document_id}
            )
            
            if tree:
                tree["_id"] = str(tree["_id"])
            
            return tree
            
        except Exception as e:
            logger.error(f"Error retrieving document tree: {str(e)}")
            raise
    
    async def update_document_tree(
        self, 
        document_id: str, 
        tree_data: Dict[str, Any]
    ) -> bool:
        """Update existing document tree."""
        try:
            tree_data["updated_at"] = datetime.utcnow()
            
            result = await self.collections[self.TREES_COLLECTION].update_one(
                {"document_id": document_id},
                {"$set": tree_data}
            )
            
            if result.modified_count > 0:
                logger.info(f"Updated tree for document: {document_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating document tree: {str(e)}")
            raise
    
    async def search_trees_by_keywords(
        self, 
        keywords: List[str], 
        user_id: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search trees by keywords in topics."""
        try:
            # Build search query
            search_query = {
                "$or": [
                    {"root_topics.keywords": {"$in": keywords}},
                    {"root_topics.title": {"$regex": "|".join(keywords), "$options": "i"}},
                    {"root_topics.summary": {"$regex": "|".join(keywords), "$options": "i"}}
                ]
            }
            
            # Add user filter if specified
            if user_id:
                # Get user's documents first
                user_docs = await self.collections[self.DOCUMENTS_COLLECTION].find(
                    {"user_id": user_id}, {"document_id": 1}
                ).to_list(None)
                
                doc_ids = [doc["document_id"] for doc in user_docs]
                search_query["document_id"] = {"$in": doc_ids}
            
            cursor = self.collections[self.TREES_COLLECTION].find(search_query).limit(limit)
            
            trees = []
            async for tree in cursor:
                tree["_id"] = str(tree["_id"])
                trees.append(tree)
            
            return trees
            
        except Exception as e:
            logger.error(f"Error searching trees: {str(e)}")
            raise
    
    # Tree Traversal and Query Methods
    
    async def find_topics_by_level(
        self, 
        document_id: str, 
        level: str
    ) -> List[Dict[str, Any]]:
        """Find all topics at a specific level in the tree."""
        try:
            pipeline = [
                {"$match": {"document_id": document_id}},
                {"$unwind": "$root_topics"},
                {"$match": {"root_topics.level": level}},
                {"$replaceRoot": {"newRoot": "$root_topics"}}
            ]
            
            cursor = self.collections[self.TREES_COLLECTION].aggregate(pipeline)
            
            topics = []
            async for topic in cursor:
                topics.append(topic)
            
            return topics
            
        except Exception as e:
            logger.error(f"Error finding topics by level: {str(e)}")
            raise
    
    async def find_topic_path(
        self, 
        document_id: str, 
        topic_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Find the path from root to a specific topic."""
        try:
            tree = await self.get_document_tree(document_id)
            if not tree:
                return None
            
            def find_path_recursive(topics: List[Dict[str, Any]], target_id: str, path: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
                for topic in topics:
                    current_path = path + [topic]
                    
                    if topic["id"] == target_id:
                        return current_path
                    
                    if "children" in topic:
                        result = find_path_recursive(topic["children"], target_id, current_path)
                        if result:
                            return result
                
                return None
            
            return find_path_recursive(tree["root_topics"], topic_id, [])
            
        except Exception as e:
            logger.error(f"Error finding topic path: {str(e)}")
            raise
    
    async def get_topic_children(
        self, 
        document_id: str, 
        topic_id: str
    ) -> List[Dict[str, Any]]:
        """Get all children of a specific topic."""
        try:
            pipeline = [
                {"$match": {"document_id": document_id}},
                {"$unwind": "$root_topics"},
                {
                    "$graphLookup": {
                        "from": self.TREES_COLLECTION,
                        "startWith": "$root_topics",
                        "connectFromField": "children",
                        "connectToField": "id",
                        "as": "all_topics"
                    }
                },
                {"$unwind": "$all_topics"},
                {"$match": {"all_topics.id": topic_id}},
                {"$replaceRoot": {"newRoot": "$all_topics.children"}}
            ]
            
            cursor = self.collections[self.TREES_COLLECTION].aggregate(pipeline)
            
            children = []
            async for child_list in cursor:
                if isinstance(child_list, list):
                    children.extend(child_list)
                else:
                    children.append(child_list)
            
            return children
            
        except Exception as e:
            logger.error(f"Error getting topic children: {str(e)}")
            raise
    
    # Patch History Management
    
    async def store_patch_history(self, patch_data: Dict[str, Any]) -> str:
        """Store patch operation history for audit trail."""
        try:
            result = await self.collections[self.PATCH_HISTORY_COLLECTION].insert_one(patch_data)
            
            logger.info(f"Stored patch history: {patch_data['patch_id']}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error storing patch history: {str(e)}")
            raise
    
    async def get_patch_history(self, patch_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve patch history by ID."""
        try:
            patch = await self.collections[self.PATCH_HISTORY_COLLECTION].find_one(
                {"patch_id": patch_id}
            )
            
            if patch:
                patch["_id"] = str(patch["_id"])
            
            return patch
            
        except Exception as e:
            logger.error(f"Error retrieving patch history: {str(e)}")
            raise
    
    async def get_document_patch_history(
        self, 
        document_id: str, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get patch history for a specific document."""
        try:
            cursor = self.collections[self.PATCH_HISTORY_COLLECTION].find(
                {"document_id": document_id}
            ).sort("created_at", DESCENDING).limit(limit)
            
            patches = []
            async for patch in cursor:
                patch["_id"] = str(patch["_id"])
                patches.append(patch)
            
            return patches
            
        except Exception as e:
            logger.error(f"Error retrieving document patch history: {str(e)}")
            raise
    
    # User Management
    
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create a new user record."""
        try:
            user_data["created_at"] = datetime.utcnow()
            user_data["updated_at"] = datetime.utcnow()
            
            result = await self.collections[self.USERS_COLLECTION].insert_one(user_data)
            
            logger.info(f"Created user: {user_data['user_id']}")
            return str(result.inserted_id)
            
        except DuplicateKeyError:
            logger.error(f"User already exists: {user_data['user_id']}")
            raise ValueError(f"User {user_data['user_id']} already exists")
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}")
            raise
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by ID."""
        try:
            user = await self.collections[self.USERS_COLLECTION].find_one(
                {"user_id": user_id}
            )
            
            if user:
                user["_id"] = str(user["_id"])
            
            return user
            
        except Exception as e:
            logger.error(f"Error retrieving user: {str(e)}")
            raise
    
    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user by email."""
        try:
            user = await self.collections[self.USERS_COLLECTION].find_one(
                {"email": email}
            )
            
            if user:
                user["_id"] = str(user["_id"])
            
            return user
            
        except Exception as e:
            logger.error(f"Error retrieving user by email: {str(e)}")
            raise
    
    # Session Management
    
    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new user session."""
        try:
            result = await self.collections[self.SESSIONS_COLLECTION].insert_one(session_data)
            
            logger.info(f"Created session: {session_data['session_id']}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session by ID."""
        try:
            session = await self.collections[self.SESSIONS_COLLECTION].find_one(
                {"session_id": session_id}
            )
            
            if session:
                session["_id"] = str(session["_id"])
            
            return session
            
        except Exception as e:
            logger.error(f"Error retrieving session: {str(e)}")
            raise
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            result = await self.collections[self.SESSIONS_COLLECTION].delete_one(
                {"session_id": session_id}
            )
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            raise
    
    # Query Logging
    
    async def log_query(self, query_data: Dict[str, Any]) -> str:
        """Log query for analytics and debugging."""
        try:
            query_data["created_at"] = datetime.utcnow()
            
            result = await self.collections[self.QUERY_LOGS_COLLECTION].insert_one(query_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            raise
    
    # Health and Maintenance
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform MongoDB health check."""
        try:
            # Check connection
            await self.client.admin.command('ping')
            
            # Get database stats
            stats = await self.database.command("dbStats")
            
            # Check collection counts
            collection_counts = {}
            for collection_name in self.collections.keys():
                count = await self.collections[collection_name].count_documents({})
                collection_counts[collection_name] = count
            
            return {
                "status": "healthy",
                "database_size": stats.get("dataSize", 0),
                "collections": collection_counts,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow()
            }
    
    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        try:
            result = await self.collections[self.SESSIONS_COLLECTION].delete_many(
                {"expires_at": {"$lt": datetime.utcnow()}}
            )
            
            logger.info(f"Cleaned up {result.deleted_count} expired sessions")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up sessions: {str(e)}")
            raise

# Global instance
mongodb_service = MongoDBService()
```

### 2. Database Models

**app/models/document.py**
```python
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(BaseModel):
    document_id: str = Field(..., description="Unique document identifier")
    user_id: str = Field(..., description="User who uploaded the document")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    content_type: str = Field(..., description="MIME type")
    status: DocumentStatus = Field(default=DocumentStatus.UPLOADED)
    total_pages: Optional[int] = Field(None, description="Total number of pages")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if processing failed")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

## Integration Points

### 1. FastAPI Dependency
```python
# app/core/dependencies.py
from app.services.mongodb_service import mongodb_service

async def get_mongodb():
    return mongodb_service
```

### 2. Application Startup
```python
# app/main.py
@app.on_event("startup")
async def startup_event():
    await mongodb_service.connect()

@app.on_event("shutdown")
async def shutdown_event():
    await mongodb_service.disconnect()
```

## Development Tasks

### Phase 1: Core Setup (Week 1)
- [ ] Setup MongoDB connection and configuration
- [ ] Implement basic CRUD operations
- [ ] Create database indexes
- [ ] Add connection pooling and error handling
- [ ] Implement health checks

### Phase 2: Advanced Features (Week 2)
- [ ] Implement tree traversal methods
- [ ] Add aggregation pipelines for complex queries
- [ ] Create patch history management
- [ ] Add session management
- [ ] Implement query logging

### Phase 3: Optimization (Week 3)
- [ ] Performance optimization and indexing
- [ ] Add caching strategies
- [ ] Implement backup and recovery
- [ ] Add monitoring and metrics
- [ ] Production deployment setup

## Testing Strategy

### Unit Tests
```python
import pytest
from app.services.mongodb_service import mongodb_service

@pytest.mark.asyncio
async def test_document_creation():
    document_data = {
        "document_id": "test_doc_1",
        "user_id": "test_user",
        "filename": "test.pdf",
        "file_size": 1024,
        "content_type": "application/pdf"
    }
    
    result = await mongodb_service.create_document(document_data)
    assert result is not None
    
    # Verify document exists
    doc = await mongodb_service.get_document("test_doc_1")
    assert doc["document_id"] == "test_doc_1"

@pytest.mark.asyncio
async def test_tree_storage():
    tree_data = {
        "document_id": "test_doc_1",
        "root_topics": [
            {
                "id": "topic_1",
                "title": "Test Topic",
                "level": "L1",
                "summary": "Test summary",
                "keywords": ["test"],
                "relevance_score": 0.9
            }
        ],
        "total_pages": 5
    }
    
    result = await mongodb_service.store_document_tree(tree_data)
    assert result is not None
```

## Performance Considerations

### Indexing Strategy
- Compound indexes for common query patterns
- Text indexes for full-text search
- Sparse indexes for optional fields
- TTL indexes for session expiration

### Query Optimization
- Use aggregation pipelines for complex operations
- Implement proper pagination
- Cache frequently accessed data
- Monitor slow queries

### Connection Management
- Connection pooling configuration
- Proper connection lifecycle management
- Retry logic for transient failures
- Circuit breaker pattern

## Security Requirements

### Authentication & Authorization
- User-based data isolation
- Role-based access control
- Session management
- API key validation

### Data Protection
- Encryption at rest (MongoDB feature)
- Secure connection (TLS)
- Input validation and sanitization
- Audit logging

## Deployment Configuration

### Docker Compose Integration
```yaml
# From Section 7 of PRD
mongodb:
  image: mongo:7.0
  container_name: vectorless-rag-mongodb
  environment:
    MONGO_INITDB_ROOT_USERNAME: ${MONGO_ROOT_USERNAME}
    MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ROOT_PASSWORD}
    MONGO_INITDB_DATABASE: ${MONGO_DATABASE}
  volumes:
    - mongodb_data:/data/db
    - ./mongodb/init:/docker-entrypoint-initdb.d
  ports:
    - "27017:27017"
  networks:
    - vectorless-rag-network
```

### Environment Variables
```bash
MONGODB_URL=mongodb://username:password@mongodb:27017/vectorless_rag
MONGODB_DATABASE=vectorless_rag
MONGODB_MAX_CONNECTIONS=100
MONGODB_MIN_CONNECTIONS=10
MONGODB_MAX_IDLE_TIME=30000
MONGODB_TIMEOUT=5000
```

## Monitoring & Logging

### Metrics to Track
- Connection pool usage
- Query performance
- Document and tree counts
- Error rates
- Session activity

### Logging Strategy
- Query logging for debugging
- Performance monitoring
- Error tracking and alerting
- Audit trail maintenance

### Health Monitoring
```python
@router.get("/health/mongodb")
async def mongodb_health():
    health_status = await mongodb_service.health_check()
    return health_status
```

---

*This MongoDB implementation guide provides the foundation for robust data storage and retrieval in the vectorless RAG system. Follow the implementation phases and integrate with other system components as specified in the PRD.*