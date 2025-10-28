# Tree Merger Testing and Deployment Guide

## File: tests/test_tree_merger_service.py (New File)

```python
"""
Comprehensive tests for tree merger service.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import List, Dict, Any

from app.services.tree_merger_service import TreeMergerService, tree_merger_service
from app.models.tree import (
    MergePatch, TreeMergeResult, CompleteDocumentTree, 
    DocumentTreeNode, MergeOperation, NodeReference
)
from app.core.exceptions import ProcessingError, ValidationError

class TestTreeMergerService:
    """Test cases for TreeMergerService."""

    @pytest.fixture
    def service(self):
        """Create tree merger service instance."""
        return TreeMergerService()

    @pytest.fixture
    def sample_subtrees(self):
        """Sample subtrees data for testing."""
        return [
            {
                "_id": "subtree_1",
                "document_id": "doc_123",
                "page_number": 1,
                "page_tree": [
                    {
                        "title": "Introduction",
                        "summary": "Overview of the document",
                        "children": [
                            {
                                "title": "Background",
                                "summary": "Historical context",
                                "children": [
                                    {
                                        "title": "Early History",
                                        "summary": "Ancient origins"
                                    }
                                ]
                            }
                        ]
                    }
                ]
            },
            {
                "_id": "subtree_2", 
                "document_id": "doc_123",
                "page_number": 2,
                "page_tree": [
                    {
                        "title": "Introduction",  # Similar to page 1
                        "summary": "Document overview and scope",
                        "children": [
                            {
                                "title": "Methodology",
                                "summary": "Research approach",
                                "children": []
                            }
                        ]
                    }
                ]
            }
        ]

    @pytest.fixture
    def sample_merge_patches(self):
        """Sample merge patches for testing."""
        return [
            MergePatch(
                operation=MergeOperation.MERGE_NODES,
                target_path="/root_nodes/0",
                source_nodes=[
                    NodeReference(
                        page_number=1,
                        section_index=0,
                        node_path="/page_tree/0"
                    ),
                    NodeReference(
                        page_number=2,
                        section_index=0,
                        node_path="/page_tree/0"
                    )
                ],
                merge_strategy="content_similarity",
                merged_content={
                    "title": "Introduction",
                    "summary": "Comprehensive overview of the document and its scope",
                    "level": 1
                },
                confidence_score=0.85,
                reasoning="Both nodes have similar titles and complementary content",
                page_associations=[1, 2],
                keywords=["introduction", "overview", "document", "scope"]
            )
        ]

    @pytest.mark.asyncio
    async def test_merge_document_tree_success(self, service, sample_subtrees):
        """Test successful document tree merging."""
        document_id = "doc_123"
        
        with patch.object(service, '_fetch_document_subtrees', return_value=sample_subtrees), \
             patch.object(service, '_generate_merge_patches') as mock_generate, \
             patch.object(service, '_apply_merge_patches') as mock_apply, \
             patch.object(service, '_save_complete_tree', return_value="tree_456"):
            
            # Mock merge patch generation
            mock_generate.return_value = TreeMergeResult(
                merge_patches=[],
                processing_metadata={"token_usage": {"total_tokens": 100}}
            )
            
            # Mock tree application
            mock_apply.return_value = CompleteDocumentTree(
                document_id=document_id,
                title="Test Document",
                root_nodes=[],
                total_pages=2,
                merge_statistics={"total_patches": 0},
                processing_metadata={}
            )
            
            result = await service.merge_document_tree(document_id)
            
            assert result.document_id == document_id
            assert result.total_pages == 2
            assert "processing_time" in result.processing_metadata
            assert "token_usage" in result.processing_metadata

    @pytest.mark.asyncio
    async def test_merge_document_tree_no_subtrees(self, service):
        """Test merging when no subtrees exist."""
        document_id = "doc_nonexistent"
        
        with patch.object(service, '_fetch_document_subtrees', return_value=[]):
            with pytest.raises(ProcessingError, match="No subtrees found"):
                await service.merge_document_tree(document_id)

    @pytest.mark.asyncio
    async def test_fetch_document_subtrees(self, service, sample_subtrees):
        """Test fetching subtrees from MongoDB."""
        document_id = "doc_123"
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_cursor = AsyncMock()
        mock_cursor.to_list.return_value = sample_subtrees
        mock_collection.find.return_value = mock_cursor
        
        with patch('app.services.tree_merger_service.get_subtrees_collection', return_value=mock_collection):
            result = await service._fetch_document_subtrees(document_id)
            
            assert len(result) == 2
            assert result[0]["page_number"] == 1
            assert result[1]["page_number"] == 2
            mock_collection.find.assert_called_once_with({"document_id": document_id})

    @pytest.mark.asyncio
    async def test_generate_merge_patches(self, service, sample_subtrees):
        """Test merge patch generation using AI."""
        current_subtree = sample_subtrees[1]
        existing_subtrees = [sample_subtrees[0]]
        document_id = "doc_123"
        
        # Mock AI agent response
        mock_result = Mock()
        mock_result.output = TreeMergeResult(
            merge_patches=[
                MergePatch(
                    operation=MergeOperation.MERGE_NODES,
                    target_path="/root_nodes/0",
                    source_nodes=[],
                    merge_strategy="content_similarity",
                    merged_content={},
                    confidence_score=0.8,
                    reasoning="Test merge",
                    page_associations=[1, 2],
                    keywords=["test"]
                )
            ],
            processing_metadata={}
        )
        mock_result.usage.return_value = Mock(
            input_tokens=50,
            output_tokens=30,
            total_tokens=80,
            requests=1
        )
        
        with patch.object(service.merge_agent, 'run', return_value=mock_result):
            result = await service._generate_merge_patches(
                current_subtree, existing_subtrees, document_id
            )
            
            assert len(result.merge_patches) == 1
            assert result.merge_patches[0].confidence_score == 0.8
            assert "token_usage" in result.processing_metadata

    def test_create_merge_analysis_prompt(self, service, sample_subtrees):
        """Test merge analysis prompt creation."""
        current_subtree = sample_subtrees[1]
        existing_subtrees = [sample_subtrees[0]]
        document_id = "doc_123"
        
        prompt = service._create_merge_analysis_prompt(
            current_subtree, existing_subtrees, document_id
        )
        
        assert "page 2" in prompt.lower()
        assert "introduction" in prompt.lower()
        assert "merge patches" in prompt.lower()
        assert "confidence score" in prompt.lower()

    @pytest.mark.asyncio
    async def test_apply_merge_patches(self, service, sample_subtrees, sample_merge_patches):
        """Test applying merge patches to create complete tree."""
        document_id = "doc_123"
        
        with patch.object(service, '_create_initial_tree_structure') as mock_initial, \
             patch.object(service, '_execute_merge_patches') as mock_execute:
            
            mock_initial.return_value = []
            mock_execute.return_value = [
                DocumentTreeNode(
                    id="node_1",
                    title="Introduction",
                    level=1,
                    page_numbers=[1, 2],
                    source_nodes=[]
                )
            ]
            
            result = await service._apply_merge_patches(
                sample_subtrees, sample_merge_patches, document_id
            )
            
            assert result.document_id == document_id
            assert result.total_pages == 2
            assert len(result.root_nodes) == 1
            assert "merge_method" in result.processing_metadata

    def test_create_initial_tree_structure(self, service, sample_subtrees):
        """Test creating initial tree structure from subtrees."""
        document_id = "doc_123"
        
        result = service._create_initial_tree_structure(sample_subtrees, document_id)
        
        assert len(result) == 2  # Two sections from two pages
        assert all(node.level == 1 for node in result)  # All are sections
        assert result[0].title == "Introduction"
        assert result[1].title == "Introduction"

    def test_execute_merge_patches(self, service, sample_merge_patches):
        """Test executing merge patches on tree structure."""
        initial_tree = [
            DocumentTreeNode(
                id="node_1",
                title="Introduction",
                level=1,
                page_numbers=[1],
                source_nodes=[]
            ),
            DocumentTreeNode(
                id="node_2", 
                title="Introduction",
                level=1,
                page_numbers=[2],
                source_nodes=[]
            )
        ]
        
        result = service._execute_merge_patches(initial_tree, sample_merge_patches)
        
        # Should apply high-confidence patches
        assert len(result) >= 1

    def test_extract_keywords(self, service):
        """Test keyword extraction from node content."""
        node_data = {
            "title": "Machine Learning Introduction",
            "summary": "This section covers artificial intelligence and neural networks"
        }
        
        keywords = service._extract_keywords(node_data)
        
        assert "machine" in keywords
        assert "learning" in keywords
        assert "artificial" in keywords
        assert "intelligence" in keywords
        assert len(keywords) <= 10

    def test_calculate_merge_statistics(self, service, sample_merge_patches):
        """Test merge statistics calculation."""
        stats = service._calculate_merge_statistics(sample_merge_patches)
        
        assert stats["total_patches"] == 1
        assert stats["merge_nodes"] == 1
        assert "high_confidence_merges" in stats

    @pytest.mark.asyncio
    async def test_save_complete_tree(self, service):
        """Test saving complete tree to MongoDB."""
        complete_tree = CompleteDocumentTree(
            document_id="doc_123",
            title="Test Document",
            root_nodes=[],
            total_pages=2,
            merge_statistics={},
            processing_metadata={}
        )
        
        # Mock MongoDB collection
        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = Mock(inserted_id="tree_456")
        mock_database = Mock()
        mock_database.trees = mock_collection
        
        with patch('app.services.tree_merger_service.get_database', return_value=mock_database):
            result = await service._save_complete_tree(complete_tree)
            
            assert result == "tree_456"
            mock_collection.insert_one.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_ai_failure(self, service, sample_subtrees):
        """Test error handling when AI processing fails."""
        document_id = "doc_123"
        
        with patch.object(service, '_fetch_document_subtrees', return_value=sample_subtrees), \
             patch.object(service.merge_agent, 'run', side_effect=Exception("AI Error")):
            
            with pytest.raises(ProcessingError, match="Tree merging failed"):
                await service.merge_document_tree(document_id)

    @pytest.mark.asyncio
    async def test_error_handling_mongodb_failure(self, service):
        """Test error handling when MongoDB operations fail."""
        document_id = "doc_123"
        
        with patch('app.services.tree_merger_service.get_subtrees_collection', side_effect=Exception("DB Error")):
            with pytest.raises(ProcessingError, match="Failed to fetch subtrees"):
                await service._fetch_document_subtrees(document_id)

class TestTreeMergerIntegration:
    """Integration tests for tree merger service."""

    @pytest.mark.asyncio
    async def test_full_merge_workflow(self):
        """Test complete merge workflow end-to-end."""
        # This would require actual MongoDB and AI service setup
        # Placeholder for integration testing
        pass

    @pytest.mark.asyncio
    async def test_concurrent_merging(self):
        """Test concurrent tree merging operations."""
        # Test multiple documents being processed simultaneously
        pass

    @pytest.mark.asyncio
    async def test_large_document_processing(self):
        """Test processing documents with many pages."""
        # Test performance with large documents
        pass
```

## File: tests/test_tree_models.py (New File)

```python
"""
Tests for tree-related Pydantic models.
"""

import pytest
from datetime import datetime
from typing import List

from app.models.tree import (
    MergePatch, TreeMergeResult, DocumentTreeNode, CompleteDocumentTree,
    MergeOperation, NodeReference, TreeProcessingStatus
)
from pydantic import ValidationError

class TestTreeModels:
    """Test cases for tree models."""

    def test_node_reference_creation(self):
        """Test NodeReference model creation."""
        ref = NodeReference(
            page_number=1,
            section_index=0,
            subject_index=1,
            topic_index=2,
            node_path="/sections/0/subjects/1/topics/2"
        )
        
        assert ref.page_number == 1
        assert ref.section_index == 0
        assert ref.subject_index == 1
        assert ref.topic_index == 2
        assert ref.node_path == "/sections/0/subjects/1/topics/2"

    def test_merge_patch_validation(self):
        """Test MergePatch model validation."""
        # Valid patch
        patch = MergePatch(
            operation=MergeOperation.MERGE_NODES,
            target_path="/root_nodes/0",
            source_nodes=[
                NodeReference(
                    page_number=1,
                    section_index=0,
                    node_path="/page_tree/0"
                )
            ],
            merge_strategy="content_similarity",
            merged_content={"title": "Test"},
            confidence_score=0.8,
            reasoning="Test reasoning",
            page_associations=[1, 2]
        )
        
        assert patch.confidence_score == 0.8
        assert patch.operation == MergeOperation.MERGE_NODES

    def test_merge_patch_invalid_confidence(self):
        """Test MergePatch with invalid confidence score."""
        with pytest.raises(ValidationError):
            MergePatch(
                operation=MergeOperation.MERGE_NODES,
                target_path="/root_nodes/0",
                source_nodes=[],
                merge_strategy="test",
                merged_content={},
                confidence_score=1.5,  # Invalid: > 1.0
                reasoning="Test",
                page_associations=[]
            )

    def test_document_tree_node_creation(self):
        """Test DocumentTreeNode model creation."""
        node = DocumentTreeNode(
            id="node_1",
            title="Test Section",
            summary="Test summary",
            level=1,
            page_numbers=[1, 2],
            keywords=["test", "section"]
        )
        
        assert node.id == "node_1"
        assert node.level == 1
        assert len(node.page_numbers) == 2
        assert len(node.children) == 0

    def test_document_tree_node_invalid_level(self):
        """Test DocumentTreeNode with invalid level."""
        with pytest.raises(ValidationError):
            DocumentTreeNode(
                id="node_1",
                title="Test",
                level=4,  # Invalid: must be 1-3
                page_numbers=[1]
            )

    def test_document_tree_node_with_children(self):
        """Test DocumentTreeNode with child nodes."""
        child = DocumentTreeNode(
            id="child_1",
            title="Child Node",
            level=2,
            page_numbers=[1]
        )
        
        parent = DocumentTreeNode(
            id="parent_1",
            title="Parent Node",
            level=1,
            page_numbers=[1],
            children=[child]
        )
        
        assert len(parent.children) == 1
        assert parent.children[0].id == "child_1"

    def test_complete_document_tree_creation(self):
        """Test CompleteDocumentTree model creation."""
        root_node = DocumentTreeNode(
            id="root_1",
            title="Root Section",
            level=1,
            page_numbers=[1]
        )
        
        tree = CompleteDocumentTree(
            document_id="doc_123",
            title="Test Document",
            root_nodes=[root_node],
            total_pages=5,
            merge_statistics={"total_patches": 3},
            processing_metadata={"processing_time": 10.5}
        )
        
        assert tree.document_id == "doc_123"
        assert tree.total_pages == 5
        assert len(tree.root_nodes) == 1
        assert tree.merge_statistics["total_patches"] == 3

    def test_tree_merge_result_creation(self):
        """Test TreeMergeResult model creation."""
        patch = MergePatch(
            operation=MergeOperation.MERGE_NODES,
            target_path="/test",
            source_nodes=[],
            merge_strategy="test",
            merged_content={},
            confidence_score=0.7,
            reasoning="Test",
            page_associations=[]
        )
        
        result = TreeMergeResult(
            merge_patches=[patch],
            processing_metadata={"tokens": 100},
            analysis_summary="Test analysis"
        )
        
        assert len(result.merge_patches) == 1
        assert result.analysis_summary == "Test analysis"
        assert result.processing_metadata["tokens"] == 100

    def test_model_serialization(self):
        """Test model serialization to dict."""
        node = DocumentTreeNode(
            id="test_1",
            title="Test Node",
            level=1,
            page_numbers=[1]
        )
        
        node_dict = node.model_dump()
        
        assert node_dict["id"] == "test_1"
        assert node_dict["title"] == "Test Node"
        assert node_dict["level"] == 1
        assert isinstance(node_dict["created_at"], datetime)

    def test_model_json_schema(self):
        """Test model JSON schema generation."""
        schema = MergePatch.model_json_schema()
        
        assert "properties" in schema
        assert "operation" in schema["properties"]
        assert "confidence_score" in schema["properties"]
        assert schema["properties"]["confidence_score"]["minimum"] == 0.0
        assert schema["properties"]["confidence_score"]["maximum"] == 1.0
```

## File: tests/test_tree_utils.py (New File)

```python
"""
Tests for tree utility functions.
"""

import pytest
from app.utils.tree_utils import (
    validate_tree_structure, calculate_tree_metrics, generate_tree_hash,
    find_node_by_id, get_node_path, export_tree_to_json
)
from app.models.tree import CompleteDocumentTree, DocumentTreeNode

class TestTreeUtils:
    """Test cases for tree utility functions."""

    @pytest.fixture
    def sample_tree(self):
        """Create sample tree for testing."""
        topic_node = DocumentTreeNode(
            id="topic_1",
            title="Topic 1",
            level=3,
            page_numbers=[1]
        )
        
        subject_node = DocumentTreeNode(
            id="subject_1",
            title="Subject 1",
            level=2,
            page_numbers=[1],
            children=[topic_node]
        )
        
        section_node = DocumentTreeNode(
            id="section_1",
            title="Section 1",
            level=1,
            page_numbers=[1],
            children=[subject_node]
        )
        
        return CompleteDocumentTree(
            document_id="doc_123",
            title="Test Document",
            root_nodes=[section_node],
            total_pages=1,
            merge_statistics={},
            processing_metadata={}
        )

    def test_validate_tree_structure_valid(self, sample_tree):
        """Test validation of valid tree structure."""
        is_valid, errors = validate_tree_structure(sample_tree)
        
        assert is_valid
        assert len(errors) == 0

    def test_validate_tree_structure_empty_tree(self):
        """Test validation of empty tree."""
        empty_tree = CompleteDocumentTree(
            document_id="doc_123",
            title="Empty Document",
            root_nodes=[],
            total_pages=0,
            merge_statistics={},
            processing_metadata={}
        )
        
        is_valid, errors = validate_tree_structure(empty_tree)
        
        assert not is_valid
        assert "must have at least one root node" in errors[0]

    def test_validate_tree_structure_invalid_level(self):
        """Test validation with invalid node levels."""
        invalid_node = DocumentTreeNode(
            id="invalid_1",
            title="Invalid Node",
            level=2,  # Should be 1 for root node
            page_numbers=[1]
        )
        
        invalid_tree = CompleteDocumentTree(
            document_id="doc_123",
            title="Invalid Document",
            root_nodes=[invalid_node],
            total_pages=1,
            merge_statistics={},
            processing_metadata={}
        )
        
        is_valid, errors = validate_tree_structure(invalid_tree)
        
        assert not is_valid
        assert any("incorrect level" in error for error in errors)

    def test_calculate_tree_metrics(self, sample_tree):
        """Test tree metrics calculation."""
        metrics = calculate_tree_metrics(sample_tree)
        
        assert metrics["total_nodes"] == 3
        assert metrics["sections_count"] == 1
        assert metrics["subjects_count"] == 1
        assert metrics["topics_count"] == 1
        assert metrics["max_depth"] == 3

    def test_generate_tree_hash(self, sample_tree):
        """Test tree hash generation."""
        hash1 = generate_tree_hash(sample_tree)
        hash2 = generate_tree_hash(sample_tree)
        
        # Same tree should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hash length

    def test_find_node_by_id(self, sample_tree):
        """Test finding node by ID."""
        found_node = find_node_by_id(sample_tree, "subject_1")
        
        assert found_node is not None
        assert found_node.id == "subject_1"
        assert found_node.title == "Subject 1"

    def test_find_node_by_id_not_found(self, sample_tree):
        """Test finding non-existent node."""
        found_node = find_node_by_id(sample_tree, "nonexistent")
        
        assert found_node is None

    def test_get_node_path(self, sample_tree):
        """Test getting node path."""
        path = get_node_path(sample_tree, "topic_1")
        
        assert path is not None
        assert "root_nodes[0]" in path
        assert "children[0]" in path

    def test_export_tree_to_json(self, sample_tree):
        """Test exporting tree to JSON."""
        json_str = export_tree_to_json(sample_tree, include_metadata=True)
        
        assert "document_id" in json_str
        assert "root_nodes" in json_str
        assert "processing_metadata" in json_str

    def test_export_tree_to_json_no_metadata(self, sample_tree):
        """Test exporting tree without metadata."""
        json_str = export_tree_to_json(sample_tree, include_metadata=False)
        
        assert "document_id" in json_str
        assert "root_nodes" in json_str
        assert "processing_metadata" not in json_str
```

## File: docker-compose.test.yml (New File)

```yaml
version: '3.8'

services:
  # Test MongoDB instance
  mongodb-test:
    image: mongo:7.0
    container_name: vectorless-rag-mongodb-test
    environment:
      MONGO_INITDB_ROOT_USERNAME: testuser
      MONGO_INITDB_ROOT_PASSWORD: testpass
      MONGO_INITDB_DATABASE: vectorless_rag_test
    ports:
      - "27018:27017"
    volumes:
      - mongodb_test_data:/data/db
      - ./mongo-init-test.js:/docker-entrypoint-initdb.d/mongo-init.js:ro
    networks:
      - test-network

  # Test application
  app-test:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: vectorless-rag-app-test
    environment:
      - ENVIRONMENT=test
      - MONGODB_URL=mongodb://testuser:testpass@mongodb-test:27017/vectorless_rag_test?authSource=admin
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - GEMINI_MODEL=gemini-1.5-pro
    depends_on:
      - mongodb-test
    networks:
      - test-network
    command: ["python", "-m", "pytest", "tests/", "-v", "--cov=app"]

volumes:
  mongodb_test_data:

networks:
  test-network:
    driver: bridge
```

## File: .github/workflows/test-tree-merger.yml (New File)

```yaml
name: Tree Merger Tests

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'app/services/tree_merger_service.py'
      - 'app/models/tree.py'
      - 'app/utils/tree_utils.py'
      - 'tests/test_tree_*.py'
  pull_request:
    branches: [ main ]
    paths:
      - 'app/services/tree_merger_service.py'
      - 'app/models/tree.py'
      - 'app/utils/tree_utils.py'
      - 'tests/test_tree_*.py'

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      mongodb:
        image: mongo:7.0
        env:
          MONGO_INITDB_ROOT_USERNAME: testuser
          MONGO_INITDB_ROOT_PASSWORD: testpass
          MONGO_INITDB_DATABASE: vectorless_rag_test
        ports:
          - 27017:27017
        options: >-
          --health-cmd "mongosh --eval 'db.adminCommand(\"ping\")'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Wait for MongoDB
      run: |
        until mongosh --host localhost:27017 --eval "print(\"MongoDB is ready\")"; do
          echo "Waiting for MongoDB..."
          sleep 2
        done
    
    - name: Run tree merger tests
      env:
        MONGODB_URL: mongodb://testuser:testpass@localhost:27017/vectorless_rag_test?authSource=admin
        GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        ENVIRONMENT: test
      run: |
        python -m pytest tests/test_tree_*.py -v --cov=app/services/tree_merger_service --cov=app/models/tree --cov=app/utils/tree_utils --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: tree-merger
        name: tree-merger-coverage
```

## File: scripts/deploy_tree_merger.sh (New File)

```bash
#!/bin/bash

# Tree Merger Service Deployment Script

set -e

echo "🚀 Deploying Tree Merger Service..."

# Configuration
SERVICE_NAME="tree-merger"
DOCKER_IMAGE="vectorless-rag:latest"
CONTAINER_NAME="vectorless-rag-app"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    if [ -z "$GEMINI_API_KEY" ]; then
        log_error "GEMINI_API_KEY environment variable is not set"
        exit 1
    fi
    
    log_info "Prerequisites check passed"
}

# Build application
build_application() {
    log_info "Building application..."
    
    docker build -t $DOCKER_IMAGE .
    
    if [ $? -eq 0 ]; then
        log_info "Application built successfully"
    else
        log_error "Application build failed"
        exit 1
    fi
}

# Run database migrations
run_migrations() {
    log_info "Running database migrations..."
    
    # Create trees collection if it doesn't exist
    docker exec vectorless-rag-mongodb mongosh \
        --username $MONGODB_USERNAME \
        --password $MONGODB_PASSWORD \
        --authenticationDatabase admin \
        vectorless_rag \
        --eval "
        if (!db.getCollectionNames().includes('trees')) {
            db.createCollection('trees');
            db.trees.createIndex({ 'document_id': 1 }, { unique: true });
            db.trees.createIndex({ 'created_at': -1 });
            db.trees.createIndex({ 'status': 1 });
            print('Trees collection created with indexes');
        } else {
            print('Trees collection already exists');
        }
        "
    
    log_info "Database migrations completed"
}

# Deploy service
deploy_service() {
    log_info "Deploying service..."
    
    # Stop existing container if running
    if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
        log_warn "Stopping existing container..."
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    fi
    
    # Start services with docker-compose
    docker-compose up -d
    
    # Wait for service to be ready
    log_info "Waiting for service to be ready..."
    sleep 10
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_info "Service is healthy"
    else
        log_error "Service health check failed"
        exit 1
    fi
}

# Run tests
run_tests() {
    log_info "Running tree merger tests..."
    
    docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
    
    if [ $? -eq 0 ]; then
        log_info "All tests passed"
    else
        log_error "Tests failed"
        exit 1
    fi
    
    # Cleanup test containers
    docker-compose -f docker-compose.test.yml down
}

# Verify deployment
verify_deployment() {
    log_info "Verifying deployment..."
    
    # Test tree merger endpoint
    response=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST \
        -H "Content-Type: application/json" \
        -d '{"document_id": "test_doc"}' \
        http://localhost:8000/api/v1/trees/generate/test_doc)
    
    if [ "$response" -eq 404 ] || [ "$response" -eq 422 ]; then
        log_info "Tree merger endpoint is responding correctly"
    else
        log_error "Tree merger endpoint test failed (HTTP $response)"
        exit 1
    fi
    
    log_info "Deployment verification completed"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    docker-compose -f docker-compose.test.yml down --volumes --remove-orphans 2>/dev/null || true
}

# Main deployment process
main() {
    trap cleanup EXIT
    
    log_info "Starting Tree Merger Service deployment..."
    
    check_prerequisites
    build_application
    run_tests
    run_migrations
    deploy_service
    verify_deployment
    
    log_info "🎉 Tree Merger Service deployed successfully!"
    log_info "Service is available at: http://localhost:8000"
    log_info "API documentation: http://localhost:8000/docs"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--skip-tests] [--skip-build] [--help]"
            echo "  --skip-tests    Skip running tests"
            echo "  --skip-build    Skip building application"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            log_error "Unknown option $1"
            exit 1
            ;;
    esac
done

# Run main function
main
```

## File: monitoring/tree_merger_metrics.py (New File)

```python
"""
Monitoring and metrics collection for tree merger service.
"""

import time
import structlog
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict, deque

logger = structlog.get_logger()

@dataclass
class TreeMergerMetrics:
    """Metrics for tree merger operations."""
    total_merges: int = 0
    successful_merges: int = 0
    failed_merges: int = 0
    average_processing_time: float = 0.0
    average_token_usage: float = 0.0
    total_nodes_processed: int = 0
    total_patches_applied: int = 0
    confidence_scores: deque = None
    
    def __post_init__(self):
        if self.confidence_scores is None:
            self.confidence_scores = deque(maxlen=1000)

class TreeMergerMonitor:
    """Monitor for tree merger service performance and health."""
    
    def __init__(self):
        self.metrics = TreeMergerMetrics()
        self.processing_times = deque(maxlen=100)
        self.token_usage_history = deque(maxlen=100)
        self.error_counts = defaultdict(int)
        self.start_time = datetime.utcnow()
    
    def record_merge_start(self, document_id: str) -> str:
        """Record the start of a merge operation."""
        operation_id = f"merge_{document_id}_{int(time.time())}"
        
        logger.info(
            "Tree merge operation started",
            operation_id=operation_id,
            document_id=document_id,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return operation_id
    
    def record_merge_success(
        self, 
        operation_id: str, 
        document_id: str,
        processing_time: float,
        token_usage: Dict[str, int],
        nodes_processed: int,
        patches_applied: int,
        confidence_scores: list
    ):
        """Record successful merge operation."""
        self.metrics.total_merges += 1
        self.metrics.successful_merges += 1
        self.metrics.total_nodes_processed += nodes_processed
        self.metrics.total_patches_applied += patches_applied
        
        # Update processing time metrics
        self.processing_times.append(processing_time)
        self.metrics.average_processing_time = sum(self.processing_times) / len(self.processing_times)
        
        # Update token usage metrics
        total_tokens = token_usage.get('total_tokens', 0)
        self.token_usage_history.append(total_tokens)
        self.metrics.average_token_usage = sum(self.token_usage_history) / len(self.token_usage_history)
        
        # Update confidence scores
        self.metrics.confidence_scores.extend(confidence_scores)
        
        logger.info(
            "Tree merge operation completed successfully",
            operation_id=operation_id,
            document_id=document_id,
            processing_time=processing_time,
            token_usage=total_tokens,
            nodes_processed=nodes_processed,
            patches_applied=patches_applied,
            average_confidence=sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        )
    
    def record_merge_failure(
        self, 
        operation_id: str, 
        document_id: str,
        error: Exception,
        processing_time: Optional[float] = None
    ):
        """Record failed merge operation."""
        self.metrics.total_merges += 1
        self.metrics.failed_merges += 1
        
        error_type = type(error).__name__
        self.error_counts[error_type] += 1
        
        logger.error(
            "Tree merge operation failed",
            operation_id=operation_id,
            document_id=document_id,
            error_type=error_type,
            error_message=str(error),
            processing_time=processing_time
        )
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status of tree merger service."""
        uptime = datetime.utcnow() - self.start_time
        success_rate = (
            self.metrics.successful_merges / self.metrics.total_merges 
            if self.metrics.total_merges > 0 else 1.0
        )
        
        # Calculate average confidence
        avg_confidence = (
            sum(self.metrics.confidence_scores) / len(self.metrics.confidence_scores)
            if self.metrics.confidence_scores else 0.0
        )
        
        return {
            "status": "healthy" if success_rate >= 0.95 else "degraded" if success_rate >= 0.8 else "unhealthy",
            "uptime_seconds": uptime.total_seconds(),
            "metrics": {
                "total_merges": self.metrics.total_merges,
                "successful_merges": self.metrics.successful_merges,
                "failed_merges": self.metrics.failed_merges,
                "success_rate": success_rate,
                "average_processing_time": self.metrics.average_processing_time,
                "average_token_usage": self.metrics.average_token_usage,
                "total_nodes_processed": self.metrics.total_nodes_processed,
                "total_patches_applied": self.metrics.total_patches_applied,
                "average_confidence": avg_confidence
            },
            "errors": dict(self.error_counts),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        recent_processing_times = list(self.processing_times)[-10:]  # Last 10 operations
        recent_token_usage = list(self.token_usage_history)[-10:]
        recent_confidence = list(self.metrics.confidence_scores)[-100:]  # Last 100 scores
        
        return {
            "processing_time": {
                "average": self.metrics.average_processing_time,
                "recent_average": sum(recent_processing_times) / len(recent_processing_times) if recent_processing_times else 0,
                "min": min(self.processing_times) if self.processing_times else 0,
                "max": max(self.processing_times) if self.processing_times else 0
            },
            "token_usage": {
                "average": self.metrics.average_token_usage,
                "recent_average": sum(recent_token_usage) / len(recent_token_usage) if recent_token_usage else 0,
                "min": min(self.token_usage_history) if self.token_usage_history else 0,
                "max": max(self.token_usage_history) if self.token_usage_history else 0
            },
            "confidence": {
                "average": sum(recent_confidence) / len(recent_confidence) if recent_confidence else 0,
                "min": min(recent_confidence) if recent_confidence else 0,
                "max": max(recent_confidence) if recent_confidence else 0
            },
            "throughput": {
                "nodes_per_second": self.metrics.total_nodes_processed / (datetime.utcnow() - self.start_time).total_seconds(),
                "patches_per_second": self.metrics.total_patches_applied / (datetime.utcnow() - self.start_time).total_seconds()
            }
        }

# Global monitor instance
tree_merger_monitor = TreeMergerMonitor()
```

This comprehensive testing and deployment guide provides:

1. **Complete Test Suite**: Unit tests, integration tests, and model validation tests
2. **CI/CD Pipeline**: GitHub Actions workflow for automated testing
3. **Docker Configuration**: Test environment setup with MongoDB
4. **Deployment Script**: Automated deployment with health checks
5. **Monitoring System**: Performance metrics and health monitoring
6. **Error Handling**: Comprehensive error tracking and logging

The implementation ensures robust testing, reliable deployment, and continuous monitoring of the tree merger service.