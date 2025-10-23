# Vectorless RAG Development Guide

## Overview

This guide outlines the Docker-based development workflow for the Vectorless RAG system. All development, testing, and deployment activities are containerized to ensure consistency across different environments.

## Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- Git for version control
- Text editor/IDE of choice

## Architecture Overview

The system consists of the following Docker services:

### Core Services
- **vectorless-rag-api**: FastAPI backend application (Port 8000)
- **mongodb**: MongoDB database for metadata storage (Port 27017)
- **minio**: MinIO object storage for file storage (Port 9000, Console: 9001)
- **redis**: Redis cache for session management (Port 6379)

### Service Dependencies
```
FastAPI ← → MongoDB (metadata)
FastAPI ← → MinIO (file storage)
FastAPI ← → Redis (caching)
FastAPI ← → Gemini API (AI processing)
```

## Development Workflow

### 1. Environment Setup

All containers should already be running. If not, start them with:

```bash
docker-compose up -d
```

### 2. Hot Reload Development

The development environment is configured with:
- **Volume mounts**: Code changes are automatically reflected in containers
- **--reload flag**: FastAPI automatically restarts on code changes
- **Live monitoring**: Container logs show real-time feedback

### 3. Code Change Workflow

1. **Make code changes** in your local IDE
2. **Save files** - changes are immediately synced to containers via volume mounts
3. **Check logs** for automatic reload confirmation:
   ```bash
   docker-compose logs -f vectorless-rag-api
   ```
4. **Test changes** using API endpoints or web interface

### 4. Container Management

#### View Running Containers
```bash
docker-compose ps
```

#### Monitor Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f vectorless-rag-api
docker-compose logs -f mongodb
docker-compose logs -f minio
docker-compose logs -f redis
```

#### Restart Services
```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart vectorless-rag-api
```

## Testing Procedures

### 1. Health Check
Verify all services are running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "vectorless-rag",
  "version": "0.1.0"
}
```

### 2. Document Upload Testing

Upload the sample PDF:
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample.pdf" \
  -F "title=Sample Document" \
  -F "description=Test document for development"
```

### 3. Topic Tree Generation

After document upload, generate topic tree:
```bash
curl -X POST "http://localhost:8000/api/v1/trees/generate" \
  -H "Content-Type: application/json" \
  -d '{"document_id": "YOUR_DOCUMENT_ID"}'
```

### 4. Query Processing

Test intelligent query processing:
```bash
curl -X POST "http://localhost:8000/api/v1/queries/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main topic of this document?",
    "scope": "document",
    "document_ids": ["YOUR_DOCUMENT_ID"]
  }'
```

## Development Best Practices

### 1. Code Changes
- **Save frequently**: Changes are reflected immediately
- **Monitor logs**: Watch for reload confirmations and errors
- **Test incrementally**: Verify each change before proceeding

### 2. Debugging
- **Container logs**: Primary source of debugging information
- **Health endpoints**: Use `/health` to verify service status
- **Database inspection**: Use MongoDB Compass or CLI for data verification

### 3. Error Handling
- **Check logs first**: Most issues are visible in container logs
- **Restart services**: If issues persist, restart specific containers
- **Environment variables**: Verify all required env vars are set

## Common Development Tasks

### Adding New API Endpoints
1. Create endpoint in appropriate router file (`app/api/v1/`)
2. Save file - FastAPI will auto-reload
3. Check logs for successful reload
4. Test endpoint with curl or API client

### Database Schema Changes
1. Modify models in `app/models/`
2. Update database operations in `app/core/database.py`
3. Test with sample data
4. Verify changes in MongoDB

### Adding New Dependencies
1. Update `pyproject.toml`
2. Rebuild container:
   ```bash
   docker-compose build vectorless-rag-api
   docker-compose up -d vectorless-rag-api
   ```

## Monitoring and Logs

### Log Levels
- **INFO**: General application flow
- **WARNING**: Potential issues
- **ERROR**: Application errors
- **DEBUG**: Detailed debugging information

### Key Log Patterns
- `INFO: Application startup complete` - Service ready
- `INFO: Uvicorn running on` - Server started
- `WARNING: Pydantic` - Configuration warnings (usually safe)
- `ERROR:` - Critical issues requiring attention

### Log Commands
```bash
# Follow all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f vectorless-rag-api

# Show last 100 lines
docker-compose logs --tail=100 vectorless-rag-api

# Show logs since timestamp
docker-compose logs --since="2024-01-01T00:00:00" vectorless-rag-api
```

## Troubleshooting

### Common Issues

#### 1. Container Won't Start
```bash
# Check container status
docker-compose ps

# View startup logs
docker-compose logs vectorless-rag-api

# Restart container
docker-compose restart vectorless-rag-api
```

#### 2. Code Changes Not Reflected
- Verify volume mounts in `docker-compose.yml`
- Check if `--reload` flag is enabled
- Restart container if needed

#### 3. Database Connection Issues
```bash
# Check MongoDB logs
docker-compose logs mongodb

# Verify connection string in logs
docker-compose logs vectorless-rag-api | grep -i mongo
```

#### 4. File Upload Issues
```bash
# Check MinIO logs
docker-compose logs minio

# Verify MinIO console access
open http://localhost:9001
```

### Environment Variables

Required environment variables (check `.env.example`):
- `GEMINI_API_KEY`: Google Gemini API key
- `MONGODB_URL`: MongoDB connection string
- `MINIO_ENDPOINT`: MinIO server endpoint
- `MINIO_ACCESS_KEY`: MinIO access credentials
- `MINIO_SECRET_KEY`: MinIO secret credentials

## API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints
- `GET /health` - Health check
- `POST /api/v1/documents/upload` - Upload documents
- `POST /api/v1/trees/generate` - Generate topic trees
- `POST /api/v1/queries/execute` - Process queries

## Development Tips

1. **Use the logs**: Container logs are your best friend for debugging
2. **Test incrementally**: Make small changes and test frequently
3. **Monitor resources**: Keep an eye on Docker resource usage
4. **Backup data**: Regularly backup MongoDB data for important tests
5. **Environment consistency**: Always use Docker for development to match production

## Next Steps

After setting up the development environment:
1. Familiarize yourself with the API documentation
2. Test the complete workflow with sample.pdf
3. Explore the codebase structure
4. Start implementing new features following the established patterns

---

For additional help, check the container logs or refer to the project documentation in the `.trae/documents/` directory.