# Achivist

A visual-first document processing and query system that uses computer vision and AI for intelligent document analysis without traditional vector embeddings.

## 🚀 Overview

The Achivist system revolutionizes document processing by:

- **Visual-First Architecture**: Converting PDFs to images and using computer vision for analysis
- **Hierarchical Topic Trees**: Organizing document content in structured, navigable trees
- **Intelligent Query Processing**: Using Pydantic AI for type-safe, intelligent responses
- **JSON Patch Manipulation**: Dynamic tree structure modification and updates
- **Scalable Infrastructure**: Docker-based microservices with MongoDB, MinIO, and Redis

## 🏗️ Architecture

### Core Components

1. **FastAPI Backend**: High-performance async API server
2. **MongoDB**: Document and tree storage with indexing
3. **MinIO**: Object storage for PDFs and images
4. **Redis**: Caching and session management
5. **Gemini Flash**: Visual document analysis and content understanding
6. **Pydantic AI**: Type-safe query processing and response generation

### Processing Pipeline

```
PDF Upload → Image Conversion → Visual Analysis → Topic Tree → Query Processing
     ↓              ↓               ↓                ↓                ↓
  Validation    PyMuPDF         Gemini Flash      MongoDB         Pydantic AI
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Python 3.11+
- **Database**: MongoDB with Motor (async driver)
- **Storage**: MinIO (S3-compatible object storage)
- **Cache**: Redis
- **AI/ML**: Google Gemini Flash, Pydantic AI
- **PDF Processing**: PyMuPDF (fitz), Pillow
- **Containerization**: Docker, Docker Compose
- **Testing**: Pytest, AsyncIO testing
- **Monitoring**: Structured logging with structlog

## 📋 Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)
- Google Gemini API key
- 8GB+ RAM recommended
- 10GB+ disk space

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd agentic-rag

# Copy environment template
cp .env.example .env

# Edit .env with your configuration
# Required: GEMINI_API_KEY
```

### 2. Docker Deployment

```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
```

### 3. Verify Installation

```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

## 📚 API Documentation

### Document Management

#### Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/v1/documents/upload" \
     -F "file=@document.pdf" \
     -F "title=My Document" \
     -F "description=Document description" \
     -F "user_id=user123"
```

#### Get Document
```bash
GET /api/v1/documents/{document_id}

curl "http://localhost:8000/api/v1/documents/doc123"
```

#### List Documents
```bash
GET /api/v1/documents/?user_id=user123&status=PROCESSED&limit=10

curl "http://localhost:8000/api/v1/documents/?status=PROCESSED&limit=10"
```

#### Get Document Statistics
```bash
GET /api/v1/documents/stats

curl "http://localhost:8000/api/v1/documents/stats"
```

#### Get Document Processing Status
```bash
GET /api/v1/documents/{document_id}/status

curl "http://localhost:8000/api/v1/documents/doc123/status"
```

#### Update Document
```bash
PUT /api/v1/documents/{document_id}
Content-Type: application/json

curl -X PUT "http://localhost:8000/api/v1/documents/doc123" \
     -H "Content-Type: application/json" \
     -d '{"title": "Updated Title", "description": "Updated description"}'
```

#### Delete Document
```bash
DELETE /api/v1/documents/{document_id}

curl -X DELETE "http://localhost:8000/api/v1/documents/doc123"
```

#### Download Document
```bash
GET /api/v1/documents/{document_id}/download

curl "http://localhost:8000/api/v1/documents/doc123/download" -o document.pdf
```

#### Merge Document Tree
```bash
POST /api/v1/documents/{document_id}/merge-tree

curl -X POST "http://localhost:8000/api/v1/documents/doc123/merge-tree"
```

#### Get Document Tree
```bash
GET /api/v1/documents/{document_id}/tree

curl "http://localhost:8000/api/v1/documents/doc123/tree"
```

#### Get Document Tree from Path
```bash
GET /api/v1/documents/{document_id}/tree/path?path=/&depth=3&serialize=false

curl "http://localhost:8000/api/v1/documents/doc123/tree/path?path=/introduction&depth=2"
```

#### Get Document Tree Statistics
```bash
GET /api/v1/documents/{document_id}/tree/stats

curl "http://localhost:8000/api/v1/documents/doc123/tree/stats"
```

#### List Document Images
```bash
GET /api/v1/documents/{document_id}/visual_elements

curl "http://localhost:8000/api/v1/documents/doc123/visual_elements"
```

### Query Processing

#### Query Document (Streaming)
```bash
POST /api/v1/query/document/{document_id}
Content-Type: application/json

curl -X POST "http://localhost:8000/api/v1/query/document/doc123" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the main topic of this document?",
       "user_id": "user123"
     }' \
     --no-buffer
```

## 🤖 AI Agent Capabilities

The Agentic RAG system features a sophisticated AI agent powered by **Pydantic AI** and **Google Gemini 2.5 Flash** that provides intelligent document querying without traditional vector embeddings. The agent uses a tool-based approach to navigate and analyze documents dynamically.

### 🧠 Core Agent Features

#### **Professional Information Retrieval**
- **Intelligent Navigation**: Dynamically explores document structure based on query context
- **Multi-Modal Analysis**: Combines text, visual elements, and document structure
- **Streaming Responses**: Real-time response generation with tool call visibility
- **Context Tracking**: Maintains query journey and tool usage analytics

#### **Advanced Tool Arsenal**

The AI agent has access to three powerful tools for comprehensive document analysis:

##### 1. **Document Tree Navigation** (`navigate_document`)
- **Purpose**: Navigate hierarchical document structure intelligently
- **Capabilities**:
  - Explore document sections by nodes' path
  - Retrieve structured summaries and page references
  - Path tracking for query journey analysis
- **Use Cases**: Finding specific sections, understanding document organization, locating relevant content areas

##### 2. **Visual Page Analysis** (`fetch_page_details`)
- **Purpose**: Deep analysis of specific document pages using computer vision
- **Capabilities**:
  - Multi-page image analysis with Gemini 2.5 Flash
  - OCR and text extraction from images
  - Chart, graph, and diagram interpretation
  - Context-aware analysis based on user query
  - Token usage tracking and optimization
- **Use Cases**: Analyzing charts, reading tables, extracting specific data, understanding visual content

##### 3. **Visual Elements Search** (`list_detected_document_visual_elements`)
- **Purpose**: Search and filter document visual elements by keywords
- **Capabilities**:
  - Keyword-based filtering across titles, summaries, and element types
  - Support for charts, tables, diagrams, signatures, and other visual elements
  - Pagination support (limit/skip parameters)
  - Case-insensitive partial matching
  - Comprehensive metadata return
- **Use Cases**: Finding specific tables, locating charts about topics, discovering visual content

### 🔍 Query Processing Workflow

#### **Intelligent Query Resolution**
1. **Query Analysis**: Agent analyzes user intent and determines optimal strategy
2. **Structure Exploration**: Uses tree navigation to locate relevant document sections
3. **Content Analysis**: Fetches and analyzes specific pages when detailed information is needed
4. **Visual Search**: Searches visual elements when query involves charts, tables, or diagrams
5. **Response Synthesis**: Combines findings into comprehensive, contextual answers

#### **Example Query Scenarios**

##### **Structural Queries**
```bash
# Query: "What are the main sections of this document?"
# Agent Strategy: Uses get_subtree_by_paths(["/"], depth=2)
# Result: Document outline with section summaries
```

##### **Data-Specific Queries**
```bash
# Query: "Show me the performance metrics table"
# Agent Strategy: 
# 1. fetch_document_visual_elements(keyword="performance")
# 2. fetch_page_details(pages=[found_page_numbers])
# Result: Located table with detailed analysis
```

##### **Content Analysis Queries**
```bash
# Query: "What does the methodology section say about data collection?"
# Agent Strategy:
# 1. get_subtree_by_paths(["/methodology"])
# 2. fetch_page_details(pages=[methodology_pages])
# Result: Detailed methodology analysis focused on data collection
```

### 📊 Advanced Features

#### **Real-Time Streaming**
- **Event Types**: `start`, `tool_call`, `text_delta`, `final_result`
- **Tool Visibility**: Users see which tools are being called and why
- **Progressive Responses**: Answers build incrementally as agent explores

#### **Context Awareness**
- **Query Journey Tracking**: Records paths explored during query resolution
- **Tool Usage Analytics**: Tracks which tools were used and their effectiveness
- **Adaptive Strategy**: Agent learns from document structure to optimize future queries

#### **Performance Optimization**
- **Smart Caching**: Leverages Redis for frequently accessed content
- **Token Management**: Optimizes Gemini API usage with intelligent batching
- **Parallel Processing**: Concurrent tool execution where possible

### 🎯 Query Examples

#### **Finding Specific Information**
```json
{
  "query": "What are the key findings in the results section?",
  "strategy": "Navigate to results → Analyze pages → Extract key points"
}
```

#### **Visual Content Discovery**
```json
{
  "query": "Show me all charts related to revenue growth",
  "strategy": "Search visual elements → Filter by keyword → Analyze chart pages"
}
```

#### **Comparative Analysis**
```json
{
  "query": "Compare the data in tables from different sections",
  "strategy": "Find all tables → Analyze content → Provide comparison"
}
```

#### **Document Understanding**
```json
{
  "query": "Summarize the document's main arguments",
  "strategy": "Explore structure → Identify key sections → Synthesize content"
}
```

### 🔧 Technical Implementation

#### **Pydantic AI Integration**
- **Type Safety**: Fully typed tool parameters and responses
- **Async Processing**: Non-blocking operations for optimal performance
- **Error Handling**: Graceful degradation with informative error messages
- **MLflow Integration**: Comprehensive logging and monitoring

#### **Gemini 2.5 Flash Features**
- **Multi-Modal Input**: Text + multiple images in single requests
- **Thinking Budget**: Configurable reasoning depth (up to 512 tokens)
- **High Performance**: Optimized for speed and accuracy
- **Cost Efficiency**: Smart token usage with detailed tracking

### 📈 Performance Metrics

#### **Response Times**
- **Tree Navigation**: ~200-500ms per path
- **Page Analysis**: ~1-3s per page (depending on complexity)
- **Visual Search**: ~100-300ms for filtering
- **End-to-End Queries**: ~2-10s (depending on complexity)

#### **Accuracy Features**
- **Multi-Tool Validation**: Cross-references information across tools
- **Context Preservation**: Maintains query context throughout exploration
- **Source Attribution**: Provides page numbers and section references
- **Confidence Indicators**: Tool usage analytics indicate information reliability

### User Management

#### Create User
```bash
POST /api/v1/users/
Content-Type: application/json

curl -X POST "http://localhost:8000/api/v1/users/" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "user@example.com",
       "username": "johndoe",
       "full_name": "John Doe",
       "password": "securepassword"
     }'
```

#### Get User Statistics
```bash
GET /api/v1/users/stats

curl "http://localhost:8000/api/v1/users/stats"
```

#### Get User
```bash
GET /api/v1/users/{user_id}

curl "http://localhost:8000/api/v1/users/user123"
```

#### List Users
```bash
GET /api/v1/users/?is_active=true&skip=0&limit=50

curl "http://localhost:8000/api/v1/users/?is_active=true&limit=10"
```

#### Update User
```bash
PUT /api/v1/users/{user_id}
Content-Type: application/json

curl -X PUT "http://localhost:8000/api/v1/users/user123" \
     -H "Content-Type: application/json" \
     -d '{
       "full_name": "John Smith",
       "is_active": true
     }'
```

#### Delete User
```bash
DELETE /api/v1/users/{user_id}

curl -X DELETE "http://localhost:8000/api/v1/users/user123"
```

#### User Login
```bash
POST /api/v1/users/{user_id}/login

curl -X POST "http://localhost:8000/api/v1/users/user123/login"
```

#### Get User Activity
```bash
GET /api/v1/users/{user_id}/activity?days=30

curl "http://localhost:8000/api/v1/users/user123/activity?days=7"
```

## 🔧 Configuration

### Environment Variables

```bash
# Application
APP_NAME=Vectorless RAG
APP_VERSION=1.0.0
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# MongoDB
MONGODB_URL=mongodb://mongodb:27017
MONGODB_DATABASE=vectorless_rag

# Redis
REDIS_URL=redis://redis:6379/0

# MinIO
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Gemini AI
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

# Processing
MAX_FILE_SIZE=50
MAX_PAGES_PER_DOCUMENT=100
SUPPORTED_FORMATS=pdf

# Cache
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# Security
SECRET_KEY=your_secret_key_here
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### Docker Compose Services

- **backend**: FastAPI application (port 8000)
- **mongodb**: MongoDB database (port 27017)
- **redis**: Redis cache (port 6379)
- **minio**: MinIO object storage (port 9000, 9001)
- **minio-client**: MinIO bucket initialization


## 📊 Monitoring and Logging

### Structured Logging

The system uses structured logging with contextual information:

```python
import structlog

logger = structlog.get_logger()
logger.info("Document processed", 
           document_id="doc123", 
           processing_time=5.2,
           page_count=10)
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Storage health
curl http://localhost:8000/health/storage
```

### Metrics Endpoints

```bash
# Document statistics
curl http://localhost:8000/api/v1/documents/stats

# Tree statistics
curl http://localhost:8000/api/v1/trees/stats

# Query analytics
curl http://localhost:8000/api/v1/queries/analytics/stats
```

## 🔍 Development

### Local Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start external services
docker-compose up -d mongodb redis minio

# Run application locally
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```


### Project Structure

```
vectorless-rag/
├── app/
│   ├── api/
│   │   └── v1/
│   │       ├── documents.py
│   │       ├── trees.py
│   │       ├── queries.py
│   │       └── users.py
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── storage.py
│   │   ├── cache.py
│   │   └── exceptions.py
│   ├── models/
│   │   ├── document.py
│   │   ├── tree.py
│   │   ├── query.py
│   │   └── user.py
│   ├── services/
│   │   ├── pdf_service.py
│   │   ├── gemini_service.py
│   │   ├── json_patch_service.py
│   │   └── pydantic_ai_service.py
│   └── main.py
├── tests/
│   ├── test_services.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### Frontend
1. Login page with email and password form and sign up link
2. Sign up page with email, username, fullname, and password and sign up button then redirection to sign in page if signup success
3. Home page with short explanation of the application concept with cards like "Start Uploading" "Start Querying" also sidebar for page navigation and profile card to show the username if clicked show option settings and log out
4. Documents page with overall metrics on top and add new button, below is the table of document list with quick action button to view, download and delete
5. Document view page to view the pdf and Document concept tree, 
6. Chat interface with history management in sidebar, the chat will show the thought process and after result is fully received, there will be button to open a sheet to show the page navigating process, which is the tree navigated (i will give more detail when we are implementing this)
7. User management only to show who are registered to the app

9. Settings page for changing theme and model used in query and indexing (Gemini 2.5 flash, Gemini 2.5 pro, or Gemini 2.5 flash lite)