"""
Test suite for core services.
Tests PDF processing, Gemini integration, JSON Patch, and Pydantic AI services.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from io import BytesIO
from PIL import Image
import json

from app.services.pdf_service import pdf_service
from app.services.gemini_service import gemini_service
from app.services.json_patch_service import tree_patch_engine
from app.services.pydantic_ai_service import pydantic_ai_processor


class TestPDFService:
    """Test PDF processing service."""
    
    @pytest.fixture
    def sample_pdf_bytes(self):
        """Create a simple PDF for testing."""
        # This would normally be a real PDF file
        return b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        img = Image.new('RGB', (100, 100), color='white')
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')
        return img_bytes.getvalue()
    
    @patch('app.services.pdf_service.fitz')
    @patch('app.services.pdf_service.storage_manager')
    async def test_convert_pdf_to_images(self, mock_storage, mock_fitz, sample_pdf_bytes):
        """Test PDF to image conversion."""
        # Mock PyMuPDF document
        mock_doc = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.tobytes.return_value = b"fake_image_data"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 2
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        
        # Mock storage upload
        mock_storage.upload_file_data = AsyncMock(return_value="uploaded_path")
        
        # Test conversion
        result = await pdf_service.convert_pdf_to_images(
            pdf_data=sample_pdf_bytes,
            document_id="test_doc_123"
        )
        
        assert result["success"] is True
        assert len(result["image_paths"]) == 2
        assert result["total_pages"] == 2
        assert "processing_time" in result
        
        # Verify storage was called
        assert mock_storage.upload_file_data.call_count == 2
    
    @patch('app.services.pdf_service.fitz')
    async def test_validate_pdf(self, mock_fitz, sample_pdf_bytes):
        """Test PDF validation."""
        # Mock valid PDF
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_doc.metadata = {"title": "Test Document", "author": "Test Author"}
        mock_fitz.open.return_value = mock_doc
        
        result = await pdf_service.validate_pdf(sample_pdf_bytes)
        
        assert result["is_valid"] is True
        assert result["page_count"] == 5
        assert result["metadata"]["title"] == "Test Document"
        assert result["file_size"] == len(sample_pdf_bytes)
    
    @patch('app.services.pdf_service.fitz')
    async def test_validate_invalid_pdf(self, mock_fitz):
        """Test validation of invalid PDF."""
        # Mock invalid PDF
        mock_fitz.open.side_effect = Exception("Invalid PDF")
        
        result = await pdf_service.validate_pdf(b"invalid_pdf_data")
        
        assert result["is_valid"] is False
        assert "error" in result


class TestGeminiService:
    """Test Gemini visual analysis service."""
    
    @pytest.fixture
    def sample_tree_response(self):
        """Sample Gemini response for tree generation."""
        return {
            "title": "Sample Document Analysis",
            "description": "Analysis of a sample document",
            "nodes": [
                {
                    "id": "root",
                    "title": "Document Root",
                    "description": "Main document content",
                    "node_type": "document",
                    "level": 0,
                    "parent_id": None,
                    "children_ids": ["section_1"],
                    "page_numbers": [1],
                    "content_summary": "Overview of document",
                    "keywords": ["document", "analysis"]
                },
                {
                    "id": "section_1",
                    "title": "Introduction",
                    "description": "Introduction section",
                    "node_type": "section",
                    "level": 1,
                    "parent_id": "root",
                    "children_ids": [],
                    "page_numbers": [1],
                    "content_summary": "Introduction content",
                    "keywords": ["introduction", "overview"]
                }
            ]
        }
    
    @patch('app.services.gemini_service.storage_manager')
    @patch('app.services.gemini_service.genai')
    async def test_analyze_document_images(self, mock_genai, mock_storage, sample_tree_response):
        """Test document image analysis."""
        # Mock storage download
        mock_storage.download_file_data = AsyncMock(return_value=b"fake_image_data")
        
        # Mock Gemini response
        mock_response = Mock()
        mock_response.text = json.dumps(sample_tree_response)
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test analysis
        result = await gemini_service.analyze_document_images(
            image_paths=["image1.png", "image2.png"],
            document_id="test_doc_123",
            document_title="Test Document"
        )
        
        assert result["success"] is True
        assert result["tree_data"]["title"] == "Sample Document Analysis"
        assert len(result["tree_data"]["nodes"]) == 2
        assert "processing_time" in result
        
        # Verify storage was called
        assert mock_storage.download_file_data.call_count == 2
    
    @patch('app.services.gemini_service.storage_manager')
    @patch('app.services.gemini_service.genai')
    async def test_analyze_content_query(self, mock_genai, mock_storage):
        """Test content analysis for specific query."""
        # Mock storage and Gemini
        mock_storage.download_file_data = AsyncMock(return_value=b"fake_image_data")
        
        mock_response = Mock()
        mock_response.text = json.dumps({
            "relevant_content": "This section discusses the main topic",
            "confidence_score": 0.85,
            "page_references": [1, 2]
        })
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        result = await gemini_service.analyze_content_for_query(
            image_paths=["image1.png"],
            query="What is the main topic?",
            context={}
        )
        
        assert result["success"] is True
        assert result["analysis"]["confidence_score"] == 0.85
        assert "relevant_content" in result["analysis"]


class TestJSONPatchService:
    """Test JSON Patch tree manipulation service."""
    
    @pytest.fixture
    def sample_tree(self):
        """Sample tree structure for testing."""
        return {
            "title": "Test Tree",
            "description": "A test tree structure",
            "nodes": [
                {
                    "id": "root",
                    "title": "Root Node",
                    "description": "Root description",
                    "node_type": "document",
                    "level": 0,
                    "parent_id": None,
                    "children_ids": ["child1"],
                    "page_numbers": [1],
                    "content_summary": "Root content",
                    "keywords": ["root"]
                },
                {
                    "id": "child1",
                    "title": "Child Node",
                    "description": "Child description",
                    "node_type": "section",
                    "level": 1,
                    "parent_id": "root",
                    "children_ids": [],
                    "page_numbers": [1],
                    "content_summary": "Child content",
                    "keywords": ["child"]
                }
            ]
        }
    
    def test_apply_add_patch(self, sample_tree):
        """Test adding a new node via JSON Patch."""
        patches = [
            {
                "op": "add",
                "path": "/nodes/-",
                "value": {
                    "id": "child2",
                    "title": "New Child",
                    "description": "New child description",
                    "node_type": "section",
                    "level": 1,
                    "parent_id": "root",
                    "children_ids": [],
                    "page_numbers": [2],
                    "content_summary": "New content",
                    "keywords": ["new"]
                }
            },
            {
                "op": "add",
                "path": "/nodes/0/children_ids/-",
                "value": "child2"
            }
        ]
        
        result, log = tree_patch_engine.apply_patches(sample_tree, patches)
        
        assert len(result["nodes"]) == 3
        assert result["nodes"][2]["id"] == "child2"
        assert "child2" in result["nodes"][0]["children_ids"]
    
    def test_apply_remove_patch(self, sample_tree):
        """Test removing a node via JSON Patch."""
        patches = [
            {
                "op": "remove",
                "path": "/nodes/1"
            },
            {
                "op": "remove",
                "path": "/nodes/0/children_ids/0"
            }
        ]
        
        result, log = tree_patch_engine.apply_patches(sample_tree, patches)
        
        assert len(result["nodes"]) == 1
        assert len(result["nodes"][0]["children_ids"]) == 0
    
    def test_apply_replace_patch(self, sample_tree):
        """Test replacing node properties via JSON Patch."""
        patches = [
            {
                "op": "replace",
                "path": "/nodes/1/title",
                "value": "Updated Child Title"
            },
            {
                "op": "replace",
                "path": "/nodes/1/description",
                "value": "Updated description"
            }
        ]
        
        result, log = tree_patch_engine.apply_patches(sample_tree, patches)
        
        assert result["nodes"][1]["title"] == "Updated Child Title"
        assert result["nodes"][1]["description"] == "Updated description"
    
    def test_validate_tree_structure(self, sample_tree):
        """Test tree structure validation."""
        # Valid tree should pass
        is_valid, errors = tree_patch_engine.validate_tree_structure(sample_tree)
        assert is_valid is True
        assert len(errors) == 0
        
        # Invalid tree (missing root) should fail
        invalid_tree = {
            "title": "Invalid Tree",
            "nodes": [
                {
                    "id": "child1",
                    "title": "Child Node",
                    "node_type": "section",
                    "level": 1,
                    "parent_id": "nonexistent",
                    "children_ids": []
                }
            ]
        }
        
        is_valid, errors = tree_patch_engine.validate_tree_structure(invalid_tree)
        assert is_valid is False
        assert len(errors) > 0


class TestPydanticAIService:
    """Test Pydantic AI query processing service."""
    
    @pytest.fixture
    def sample_query_analysis(self):
        """Sample query analysis result."""
        return {
            "intent": "search",
            "entities": ["document", "analysis"],
            "query_type": "factual",
            "scope": "document",
            "confidence": 0.9
        }
    
    @pytest.fixture
    def sample_search_results(self):
        """Sample document search results."""
        return {
            "documents": [
                {
                    "document_id": "doc1",
                    "relevance_score": 0.85,
                    "matching_sections": ["intro", "conclusion"]
                }
            ],
            "total_found": 1
        }
    
    async def test_analyze_query(self, sample_query_analysis):
        """Test query analysis."""
        with patch.object(pydantic_ai_processor, 'simulate_query_analysis', return_value=sample_query_analysis):
            result = await pydantic_ai_processor.analyze_query("What is document analysis?")
            
            assert result["intent"] == "search"
            assert result["confidence"] == 0.9
            assert "document" in result["entities"]
    
    async def test_search_documents(self, sample_search_results):
        """Test document search."""
        with patch.object(pydantic_ai_processor, 'simulate_document_search', return_value=sample_search_results):
            result = await pydantic_ai_processor.search_documents(
                query="document analysis",
                document_ids=["doc1", "doc2"]
            )
            
            assert result["total_found"] == 1
            assert len(result["documents"]) == 1
            assert result["documents"][0]["document_id"] == "doc1"
    
    async def test_generate_answer(self):
        """Test answer generation."""
        sample_answer = {
            "answer": "Document analysis involves examining document structure and content.",
            "confidence": 0.88,
            "sources": ["doc1"],
            "reasoning": "Based on the document content analysis."
        }
        
        with patch.object(pydantic_ai_processor, 'simulate_answer_generation', return_value=sample_answer):
            result = await pydantic_ai_processor.generate_answer(
                query="What is document analysis?",
                context={"documents": ["doc1"]},
                extracted_content={"content": "Document analysis content"}
            )
            
            assert result["confidence"] == 0.88
            assert "Document analysis involves" in result["answer"]
            assert "doc1" in result["sources"]
    
    async def test_process_complete_query(self):
        """Test complete query processing pipeline."""
        # Mock all the individual steps
        with patch.object(pydantic_ai_processor, 'analyze_query') as mock_analyze, \
             patch.object(pydantic_ai_processor, 'search_documents') as mock_search, \
             patch.object(pydantic_ai_processor, 'extract_content') as mock_extract, \
             patch.object(pydantic_ai_processor, 'generate_answer') as mock_answer:
            
            # Setup mocks
            mock_analyze.return_value = {"intent": "search", "confidence": 0.9}
            mock_search.return_value = {"documents": [{"document_id": "doc1"}]}
            mock_extract.return_value = {"content": "Extracted content"}
            mock_answer.return_value = {"answer": "Final answer", "confidence": 0.85}
            
            result = await pydantic_ai_processor.process_complete_query(
                query="What is document analysis?",
                document_ids=["doc1"]
            )
            
            assert "query_analysis" in result
            assert "search_results" in result
            assert "extracted_content" in result
            assert "final_answer" in result
            assert result["final_answer"]["answer"] == "Final answer"
            
            # Verify all steps were called
            mock_analyze.assert_called_once()
            mock_search.assert_called_once()
            mock_extract.assert_called_once()
            mock_answer.assert_called_once()


# Integration test
class TestServiceIntegration:
    """Test integration between services."""
    
    @patch('app.services.pdf_service.storage_manager')
    @patch('app.services.pdf_service.fitz')
    @patch('app.services.gemini_service.storage_manager')
    @patch('app.services.gemini_service.genai')
    async def test_pdf_to_tree_pipeline(self, mock_genai, mock_gemini_storage, mock_fitz, mock_pdf_storage):
        """Test complete pipeline from PDF to tree generation."""
        # Mock PDF processing
        mock_doc = Mock()
        mock_page = Mock()
        mock_pix = Mock()
        mock_pix.tobytes.return_value = b"fake_image_data"
        mock_page.get_pixmap.return_value = mock_pix
        mock_doc.__len__.return_value = 1
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz.open.return_value = mock_doc
        mock_pdf_storage.upload_file_data = AsyncMock(return_value="image_path.png")
        
        # Mock Gemini analysis
        tree_data = {
            "title": "Test Document",
            "nodes": [
                {
                    "id": "root",
                    "title": "Document Root",
                    "node_type": "document",
                    "level": 0,
                    "parent_id": None,
                    "children_ids": []
                }
            ]
        }
        mock_gemini_storage.download_file_data = AsyncMock(return_value=b"fake_image_data")
        mock_response = Mock()
        mock_response.text = json.dumps(tree_data)
        mock_model = Mock()
        mock_model.generate_content = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model
        
        # Test pipeline
        pdf_data = b"fake_pdf_data"
        
        # Step 1: Convert PDF to images
        pdf_result = await pdf_service.convert_pdf_to_images(pdf_data, "test_doc")
        assert pdf_result["success"] is True
        
        # Step 2: Analyze images with Gemini
        gemini_result = await gemini_service.analyze_document_images(
            image_paths=pdf_result["image_paths"],
            document_id="test_doc",
            document_title="Test Document"
        )
        assert gemini_result["success"] is True
        assert gemini_result["tree_data"]["title"] == "Test Document"
        
        # Step 3: Apply JSON patches to modify tree
        patches = [
            {
                "op": "replace",
                "path": "/title",
                "value": "Modified Test Document"
            }
        ]
        
        modified_tree, log = tree_patch_engine.apply_patches(
            gemini_result["tree_data"],
            patches
        )
        assert modified_tree["title"] == "Modified Test Document"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])