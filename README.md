# Vectorless RAG System

A visual-first document processing and query system that uses computer vision and AI for intelligent document analysis without traditional vector embeddings.

## 🚀 Overview

The Vectorless RAG system revolutionizes document processing by:

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
     ↓              ↓               ↓              ↓            ↓
  Validation    PyMuPDF        Gemini Flash    MongoDB    Pydantic AI
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
cd vectorless-rag

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
GET /api/v1/documents/{document_id}/images

curl "http://localhost:8000/api/v1/documents/doc123/images"
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

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_services.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual service and function testing
- **Integration Tests**: Service interaction testing
- **API Tests**: Endpoint testing with mocked dependencies
- **Pipeline Tests**: End-to-end workflow testing

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

### Code Quality

```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Lint code
flake8 app/ tests/

# Type checking
mypy app/
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

## 🚨 Troubleshooting

### Common Issues

#### 1. Gemini API Errors
```bash
# Check API key
echo $GEMINI_API_KEY

# Verify API access
curl -H "Authorization: Bearer $GEMINI_API_KEY" \
     https://generativelanguage.googleapis.com/v1/models
```

#### 2. MongoDB Connection Issues
```bash
# Check MongoDB status
docker-compose logs mongodb

# Test connection
docker-compose exec mongodb mongosh --eval "db.adminCommand('ping')"
```

#### 3. MinIO Storage Issues
```bash
# Check MinIO status
docker-compose logs minio

# Access MinIO console
open http://localhost:9001
```

#### 4. Memory Issues
```bash
# Check container memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop → Settings → Resources → Memory
```

### Performance Optimization

1. **Database Indexing**: Ensure proper MongoDB indexes
2. **Caching**: Configure Redis for optimal cache hit rates
3. **Image Processing**: Optimize PDF to image conversion settings
4. **Concurrent Processing**: Adjust worker counts based on CPU cores

## 📈 Scaling

### Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  backend:
    deploy:
      replicas: 3
  
  mongodb:
    deploy:
      replicas: 3
      
  redis:
    deploy:
      replicas: 3
```

### Load Balancing

```nginx
# nginx.conf
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
    }
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run quality checks
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Google Gemini for visual AI capabilities
- Pydantic AI for type-safe AI processing
- FastAPI for high-performance web framework
- MongoDB for flexible document storage
- MinIO for S3-compatible object storage

## 📞 Support

For support and questions:

- Create an issue on GitHub
- Check the documentation
- Review the troubleshooting guide

---

**Built with ❤️ for intelligent document processing**