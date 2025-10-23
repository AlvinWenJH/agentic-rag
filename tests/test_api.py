"""
Test suite for API endpoints.
Tests all REST API endpoints for documents, trees, queries, and users.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime
from io import BytesIO

from app.main import app
from app.models.document import DocumentStatus, DocumentType
from app.models.tree import TreeStatus, NodeType
from app.models.query import QueryStatus, QueryType, QueryScope


client = TestClient(app)


class TestDocumentAPI:
    """Test document management API endpoints."""
    
    @pytest.fixture
    def sample_pdf_file(self):
        """Create a sample PDF file for testing."""
        pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
        return BytesIO(pdf_content)
    
    @patch('app.api.v1.documents.get_documents_collection')
    @patch('app.api.v1.documents.pdf_service')
    @patch('app.api.v1.documents.gemini_service')
    def test_upload_document(self, mock_gemini, mock_pdf, mock_collection):
        """Test document upload endpoint."""
        # Mock database collection
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.insert_one = AsyncMock(return_value=Mock(inserted_id="doc123"))
        mock_collection.return_value.find_one = AsyncMock(return_value={
            "_id": "doc123",
            "title": "test.pdf",
            "status": DocumentStatus.processing,
            "created_timestamp": datetime.utcnow()
        })
        
        # Mock PDF service
        mock_pdf.validate_pdf = AsyncMock(return_value={
            "is_valid": True,
            "page_count": 5,
            "file_size": 1024
        })
        
        # Test upload
        with open("test.pdf", "wb") as f:
            f.write(b"fake_pdf_content")
        
        with open("test.pdf", "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_id"] == "doc123"
        assert data["status"] == "processing"
    
    @patch('app.api.v1.documents.get_documents_collection')
    def test_get_document(self, mock_collection):
        """Test get document endpoint."""
        # Mock database response
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value={
            "_id": "doc123",
            "title": "test.pdf",
            "status": DocumentStatus.completed,
            "document_type": DocumentType.pdf,
            "file_size": 1024,
            "page_count": 5,
            "created_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow()
        })
        
        response = client.get("/api/v1/documents/doc123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "doc123"
        assert data["title"] == "test.pdf"
        assert data["status"] == DocumentStatus.completed
    
    @patch('app.api.v1.documents.get_documents_collection')
    def test_list_documents(self, mock_collection):
        """Test list documents endpoint."""
        # Mock database response
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.count_documents = AsyncMock(return_value=2)
        mock_collection.return_value.find.return_value.skip.return_value.limit.return_value.sort.return_value.to_list = AsyncMock(return_value=[
            {
                "_id": "doc1",
                "title": "document1.pdf",
                "status": DocumentStatus.completed,
                "created_timestamp": datetime.utcnow()
            },
            {
                "_id": "doc2",
                "title": "document2.pdf",
                "status": DocumentStatus.processing,
                "created_timestamp": datetime.utcnow()
            }
        ])
        
        response = client.get("/api/v1/documents/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["id"] == "doc1"
    
    @patch('app.api.v1.documents.get_documents_collection')
    def test_delete_document(self, mock_collection):
        """Test delete document endpoint."""
        # Mock database response
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value={
            "_id": "doc123",
            "title": "test.pdf",
            "file_path": "documents/test.pdf",
            "image_paths": ["images/page1.png", "images/page2.png"]
        })
        mock_collection.return_value.delete_one = AsyncMock()
        
        with patch('app.api.v1.documents.storage_manager') as mock_storage:
            mock_storage.delete_file = AsyncMock()
            
            response = client.delete("/api/v1/documents/doc123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Document deleted successfully"


class TestTreeAPI:
    """Test tree management API endpoints."""
    
    @pytest.fixture
    def sample_tree_data(self):
        """Sample tree data for testing."""
        return {
            "_id": "tree123",
            "document_id": "doc123",
            "title": "Test Tree",
            "description": "A test tree",
            "status": TreeStatus.completed,
            "tree_data": {
                "title": "Test Tree",
                "nodes": [
                    {
                        "id": "root",
                        "title": "Root Node",
                        "node_type": "document",
                        "level": 0,
                        "parent_id": None,
                        "children_ids": ["child1"]
                    },
                    {
                        "id": "child1",
                        "title": "Child Node",
                        "node_type": "section",
                        "level": 1,
                        "parent_id": "root",
                        "children_ids": []
                    }
                ]
            },
            "created_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow()
        }
    
    @patch('app.api.v1.trees.get_trees_collection')
    @patch('app.api.v1.trees.get_documents_collection')
    @patch('app.api.v1.trees.gemini_service')
    def test_generate_tree(self, mock_gemini, mock_docs_collection, mock_trees_collection, sample_tree_data):
        """Test tree generation endpoint."""
        # Mock document exists
        mock_docs_collection.return_value = AsyncMock()
        mock_docs_collection.return_value.find_one = AsyncMock(return_value={
            "_id": "doc123",
            "title": "Test Document",
            "image_paths": ["image1.png", "image2.png"]
        })
        
        # Mock no existing tree
        mock_trees_collection.return_value = AsyncMock()
        mock_trees_collection.return_value.find_one = AsyncMock(return_value=None)
        mock_trees_collection.return_value.insert_one = AsyncMock(return_value=Mock(inserted_id="tree123"))
        mock_trees_collection.return_value.find_one.side_effect = [None, sample_tree_data]
        
        # Mock Gemini service
        mock_gemini.analyze_document_images = AsyncMock(return_value={
            "tree_data": sample_tree_data["tree_data"],
            "processing_time": 5.2
        })
        
        response = client.post("/api/v1/trees/generate", json={
            "document_id": "doc123",
            "regenerate": False
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["tree_id"] == "tree123"
        assert data["status"] == "completed"
        assert data["document_id"] == "doc123"
    
    @patch('app.api.v1.trees.get_trees_collection')
    def test_get_tree(self, mock_collection, sample_tree_data):
        """Test get tree endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=sample_tree_data)
        
        response = client.get("/api/v1/trees/tree123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "tree123"
        assert data["title"] == "Test Tree"
    
    @patch('app.api.v1.trees.get_trees_collection')
    @patch('app.api.v1.trees.tree_patch_engine')
    def test_apply_tree_patches(self, mock_patch_engine, mock_collection, sample_tree_data):
        """Test tree patch application endpoint."""
        # Mock tree exists
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one.side_effect = [
            sample_tree_data,  # First call to get tree
            {**sample_tree_data, "id": "tree123"}  # Second call after update
        ]
        mock_collection.return_value.update_one = AsyncMock()
        
        # Mock patch engine
        modified_tree_data = sample_tree_data["tree_data"].copy()
        modified_tree_data["title"] = "Modified Tree"
        mock_patch_engine.apply_patches.return_value = (modified_tree_data, ["Patch applied successfully"])
        
        patches = [
            {
                "op": "replace",
                "path": "/title",
                "value": "Modified Tree"
            }
        ]
        
        response = client.patch("/api/v1/trees/tree123", json={
            "operations": patches
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["operations_applied"] == 1
    
    @patch('app.api.v1.trees.get_trees_collection')
    def test_search_tree(self, mock_collection, sample_tree_data):
        """Test tree search endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=sample_tree_data)
        
        response = client.post("/api/v1/trees/tree123/search", json={
            "query": "root",
            "limit": 10
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["tree_id"] == "tree123"
        assert data["query"] == "root"
        assert len(data["results"]) > 0


class TestQueryAPI:
    """Test query processing API endpoints."""
    
    @pytest.fixture
    def sample_query_data(self):
        """Sample query data for testing."""
        return {
            "_id": "query123",
            "query_text": "What is document analysis?",
            "query_type": QueryType.search,
            "status": QueryStatus.completed,
            "scope": QueryScope.document,
            "document_ids": ["doc123"],
            "context": {},
            "result": {
                "answer": "Document analysis is the process of examining documents.",
                "confidence": 0.85,
                "sources": ["doc123"]
            },
            "created_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow(),
            "processing_time": 2.5,
            "confidence_score": 0.85
        }
    
    @patch('app.api.v1.queries.get_queries_collection')
    @patch('app.api.v1.queries.get_documents_collection')
    @patch('app.api.v1.queries.pydantic_ai_processor')
    def test_execute_query(self, mock_processor, mock_docs_collection, mock_queries_collection, sample_query_data):
        """Test query execution endpoint."""
        # Mock document exists
        mock_docs_collection.return_value = AsyncMock()
        mock_docs_collection.return_value.find_one = AsyncMock(return_value={"_id": "doc123"})
        
        # Mock query storage
        mock_queries_collection.return_value = AsyncMock()
        mock_queries_collection.return_value.insert_one = AsyncMock(return_value=Mock(inserted_id="query123"))
        
        # Mock Pydantic AI processor
        mock_processor.process_complete_query = AsyncMock(return_value={
            "final_answer": {"answer": "Document analysis is the process of examining documents.", "confidence": 0.85},
            "processing_time": 2.5,
            "confidence_score": 0.85,
            "suggestions": []
        })
        
        response = client.post("/api/v1/queries/execute", json={
            "query": "What is document analysis?",
            "scope": "document",
            "document_ids": ["doc123"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["query_id"] == "query123"
        assert data["status"] == "completed"
        assert data["confidence_score"] == 0.85
    
    @patch('app.api.v1.queries.get_queries_collection')
    def test_get_query(self, mock_collection, sample_query_data):
        """Test get query endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=sample_query_data)
        
        response = client.get("/api/v1/queries/query123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "query123"
        assert data["query_text"] == "What is document analysis?"
    
    @patch('app.api.v1.queries.get_queries_collection')
    def test_list_queries(self, mock_collection, sample_query_data):
        """Test list queries endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.count_documents = AsyncMock(return_value=1)
        mock_collection.return_value.find.return_value.skip.return_value.limit.return_value.sort.return_value.to_list = AsyncMock(return_value=[sample_query_data])
        
        response = client.get("/api/v1/queries/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["queries"]) == 1
        assert data["queries"][0]["id"] == "query123"
    
    @patch('app.api.v1.queries.pydantic_ai_processor')
    def test_get_query_suggestions(self, mock_processor):
        """Test query suggestions endpoint."""
        mock_processor.generate_suggestions = AsyncMock(return_value={
            "suggestions": [
                {"text": "What is document structure?", "confidence": 0.8},
                {"text": "How to analyze content?", "confidence": 0.7}
            ],
            "categories": ["analysis", "structure"],
            "related_topics": ["document processing", "content analysis"]
        })
        
        response = client.post("/api/v1/queries/suggestions", json={
            "context_type": "document",
            "document_ids": ["doc123"]
        })
        
        assert response.status_code == 200
        data = response.json()
        assert len(data["suggestions"]) == 2
        assert "analysis" in data["categories"]


class TestUserAPI:
    """Test user management API endpoints."""
    
    @pytest.fixture
    def sample_user_data(self):
        """Sample user data for testing."""
        return {
            "_id": "user123",
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "is_active": True,
            "created_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow(),
            "last_login": None,
            "query_count": 5,
            "document_count": 3
        }
    
    @patch('app.api.v1.users.get_users_collection')
    def test_create_user(self, mock_collection):
        """Test user creation endpoint."""
        # Mock no existing user
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one.side_effect = [None, {
            "_id": "user123",
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "is_active": True,
            "created_timestamp": datetime.utcnow(),
            "updated_timestamp": datetime.utcnow(),
            "query_count": 0,
            "document_count": 0
        }]
        mock_collection.return_value.insert_one = AsyncMock(return_value=Mock(inserted_id="user123"))
        
        response = client.post("/api/v1/users/", json={
            "email": "test@example.com",
            "username": "testuser",
            "full_name": "Test User",
            "password": "testpassword"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "user123"
        assert data["email"] == "test@example.com"
        assert data["username"] == "testuser"
    
    @patch('app.api.v1.users.get_users_collection')
    def test_get_user(self, mock_collection, sample_user_data):
        """Test get user endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=sample_user_data)
        
        response = client.get("/api/v1/users/user123")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "user123"
        assert data["email"] == "test@example.com"
    
    @patch('app.api.v1.users.get_users_collection')
    def test_list_users(self, mock_collection, sample_user_data):
        """Test list users endpoint."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.count_documents = AsyncMock(return_value=1)
        mock_collection.return_value.find.return_value.skip.return_value.limit.return_value.sort.return_value.to_list = AsyncMock(return_value=[sample_user_data])
        
        response = client.get("/api/v1/users/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["users"]) == 1
        assert data["users"][0]["id"] == "user123"
    
    @patch('app.api.v1.users.get_users_collection')
    def test_update_user(self, mock_collection, sample_user_data):
        """Test update user endpoint."""
        updated_user = sample_user_data.copy()
        updated_user["full_name"] = "Updated Test User"
        
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one.side_effect = [
            sample_user_data,  # Check if user exists
            None,  # Check for conflicts
            updated_user  # Return updated user
        ]
        mock_collection.return_value.update_one = AsyncMock()
        
        response = client.put("/api/v1/users/user123", json={
            "full_name": "Updated Test User"
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["full_name"] == "Updated Test User"


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data


class TestErrorHandling:
    """Test error handling across endpoints."""
    
    @patch('app.api.v1.documents.get_documents_collection')
    def test_document_not_found(self, mock_collection):
        """Test document not found error."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/documents/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('app.api.v1.trees.get_trees_collection')
    def test_tree_not_found(self, mock_collection):
        """Test tree not found error."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/trees/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('app.api.v1.queries.get_queries_collection')
    def test_query_not_found(self, mock_collection):
        """Test query not found error."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/queries/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()
    
    @patch('app.api.v1.users.get_users_collection')
    def test_user_not_found(self, mock_collection):
        """Test user not found error."""
        mock_collection.return_value = AsyncMock()
        mock_collection.return_value.find_one = AsyncMock(return_value=None)
        
        response = client.get("/api/v1/users/nonexistent")
        
        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])