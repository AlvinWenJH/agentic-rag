# Docker Compose Implementation Guide

## Overview

This document provides comprehensive implementation guidance for Docker Compose containerization in the Vectorless RAG system. Based on **Section 7** of the PRD, Docker Compose orchestrates the multi-service architecture including FastAPI, MongoDB, MinIO, and Redis containers with proper networking, volumes, and environment management.

### Purpose and Role

Docker Compose integration serves as:
- **Service Orchestration**: Coordinate multiple containerized services
- **Environment Management**: Handle development and production configurations
- **Network Configuration**: Establish secure inter-service communication
- **Volume Management**: Persist data across container restarts
- **Scaling Support**: Enable horizontal scaling and high availability

## Technical Specifications

### Core Requirements

Based on **Section 7** deployment architecture from PRD:
- **Version**: Docker Compose 3.8+ for advanced features
- **Services**: FastAPI, MongoDB, MinIO, Redis containers
- **Networking**: Custom bridge networks for service isolation
- **Volumes**: Named volumes for data persistence
- **Health Checks**: Container health monitoring
- **Secrets Management**: Secure handling of sensitive data

### Service Architecture

```mermaid
graph TD
    A[Docker Compose] --> B[FastAPI Container]
    A --> C[MongoDB Container]
    A --> D[MinIO Container]
    A --> E[Redis Container]
    
    B --> F[vectorless-rag-network]
    C --> F
    D --> F
    E --> F
    
    C --> G[mongodb-data Volume]
    D --> H[minio-data Volume]
    E --> I[redis-data Volume]
    
    B --> J[Environment Variables]
    C --> K[MongoDB Config]
    D --> L[MinIO Config]
    E --> M[Redis Config]
    
    subgraph "Container Layer"
        B
        C
        D
        E
    end
    
    subgraph "Network Layer"
        F
    end
    
    subgraph "Storage Layer"
        G
        H
        I
    end
    
    subgraph "Configuration Layer"
        J
        K
        L
        M
    end
```

## Implementation Details

### 1. Main Docker Compose Configuration

**docker-compose.yml**
```yaml
version: '3.8'

services:
  # FastAPI Application Service
  fastapi:
    build:
      context: .
      dockerfile: Dockerfile
      target: ${BUILD_TARGET:-production}
    container_name: vectorless-rag-api
    restart: unless-stopped
    ports:
      - "${API_PORT:-8000}:8000"
    environment:
      - ENVIRONMENT=${ENVIRONMENT:-production}
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=mongodb://mongodb:27017/vectorless_rag
      - MINIO_ENDPOINT=minio:9000
      - MINIO_ACCESS_KEY=${MINIO_ACCESS_KEY}
      - MINIO_SECRET_KEY=${MINIO_SECRET_KEY}
      - REDIS_URL=redis://redis:6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./logs:/app/logs
      - ./uploads:/app/uploads
    networks:
      - vectorless-rag-network
    depends_on:
      mongodb:
        condition: service_healthy
      minio:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # MongoDB Database Service
  mongodb:
    image: mongo:7.0
    container_name: vectorless-rag-mongodb
    restart: unless-stopped
    ports:
      - "${MONGODB_PORT:-27017}:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=${MONGODB_ROOT_USERNAME:-admin}
      - MONGO_INITDB_ROOT_PASSWORD=${MONGODB_ROOT_PASSWORD}
      - MONGO_INITDB_DATABASE=vectorless_rag
    volumes:
      - mongodb-data:/data/db
      - mongodb-config:/data/configdb
      - ./mongodb/init:/docker-entrypoint-initdb.d:ro
      - ./mongodb/mongod.conf:/etc/mongod.conf:ro
    networks:
      - vectorless-rag-network
    command: ["mongod", "--config", "/etc/mongod.conf"]
    healthcheck:
      test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          memory: 2G
          cpus: '1.0'

  # MinIO Object Storage Service
  minio:
    image: minio/minio:RELEASE.2024-01-01T00-00-00Z
    container_name: vectorless-rag-minio
    restart: unless-stopped
    ports:
      - "${MINIO_API_PORT:-9000}:9000"
      - "${MINIO_CONSOLE_PORT:-9001}:9001"
    environment:
      - MINIO_ROOT_USER=${MINIO_ACCESS_KEY}
      - MINIO_ROOT_PASSWORD=${MINIO_SECRET_KEY}
      - MINIO_BROWSER_REDIRECT_URL=${MINIO_BROWSER_REDIRECT_URL:-http://localhost:9001}
    volumes:
      - minio-data:/data
      - ./minio/config:/root/.minio
    networks:
      - vectorless-rag-network
    command: server /data --console-address ":9001"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'

  # Redis Cache Service
  redis:
    image: redis:7.2-alpine
    container_name: vectorless-rag-redis
    restart: unless-stopped
    ports:
      - "${REDIS_PORT:-6379}:6379"
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD}
    volumes:
      - redis-data:/data
      - ./redis/redis.conf:/usr/local/etc/redis/redis.conf:ro
    networks:
      - vectorless-rag-network
    command: ["redis-server", "/usr/local/etc/redis/redis.conf"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '0.5'
        reservations:
          memory: 512M
          cpus: '0.25'

  # Nginx Reverse Proxy (Optional for production)
  nginx:
    image: nginx:1.25-alpine
    container_name: vectorless-rag-nginx
    restart: unless-stopped
    ports:
      - "${NGINX_HTTP_PORT:-80}:80"
      - "${NGINX_HTTPS_PORT:-443}:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/logs:/var/log/nginx
    networks:
      - vectorless-rag-network
    depends_on:
      - fastapi
    profiles:
      - production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
        reservations:
          memory: 256M
          cpus: '0.25'

# Network Configuration
networks:
  vectorless-rag-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
    driver_opts:
      com.docker.network.bridge.name: vectorless-rag-br0

# Volume Configuration
volumes:
  mongodb-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/mongodb
  
  mongodb-config:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/mongodb-config
  
  minio-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/minio
  
  redis-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: ${DATA_PATH:-./data}/redis

# Secrets (for production)
secrets:
  mongodb_root_password:
    file: ./secrets/mongodb_root_password.txt
  minio_secret_key:
    file: ./secrets/minio_secret_key.txt
  redis_password:
    file: ./secrets/redis_password.txt
  openai_api_key:
    file: ./secrets/openai_api_key.txt
  gemini_api_key:
    file: ./secrets/gemini_api_key.txt
```

### 2. Development Configuration

**docker-compose.dev.yml**
```yaml
version: '3.8'

services:
  fastapi:
    build:
      target: development
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
      - RELOAD=true
    volumes:
      - .:/app
      - /app/__pycache__
      - /app/.pytest_cache
    ports:
      - "8000:8000"
      - "5678:5678"  # Debug port
    command: ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  mongodb:
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_ROOT_USERNAME=admin
      - MONGO_INITDB_ROOT_PASSWORD=password123
    volumes:
      - ./dev-data/mongodb:/data/db

  minio:
    ports:
      - "9000:9000"
      - "9001:9001"
    environment:
      - MINIO_ROOT_USER=minioadmin
      - MINIO_ROOT_PASSWORD=minioadmin123
    volumes:
      - ./dev-data/minio:/data

  redis:
    ports:
      - "6379:6379"
    volumes:
      - ./dev-data/redis:/data

  # Development tools
  mongo-express:
    image: mongo-express:1.0.0
    container_name: vectorless-rag-mongo-express
    restart: unless-stopped
    ports:
      - "8081:8081"
    environment:
      - ME_CONFIG_MONGODB_ADMINUSERNAME=admin
      - ME_CONFIG_MONGODB_ADMINPASSWORD=password123
      - ME_CONFIG_MONGODB_URL=mongodb://admin:password123@mongodb:27017/
      - ME_CONFIG_BASICAUTH_USERNAME=admin
      - ME_CONFIG_BASICAUTH_PASSWORD=admin
    networks:
      - vectorless-rag-network
    depends_on:
      - mongodb

  redis-commander:
    image: rediscommander/redis-commander:latest
    container_name: vectorless-rag-redis-commander
    restart: unless-stopped
    ports:
      - "8082:8081"
    environment:
      - REDIS_HOSTS=local:redis:6379
    networks:
      - vectorless-rag-network
    depends_on:
      - redis
```

### 3. Production Configuration

**docker-compose.prod.yml**
```yaml
version: '3.8'

services:
  fastapi:
    build:
      target: production
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    secrets:
      - openai_api_key
      - gemini_api_key
    deploy:
      replicas: 3
      update_config:
        parallelism: 1
        delay: 10s
        order: start-first
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  mongodb:
    secrets:
      - mongodb_root_password
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  minio:
    secrets:
      - minio_secret_key
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.role == manager
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  redis:
    secrets:
      - redis_password
    deploy:
      replicas: 1
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3

  # Production monitoring
  prometheus:
    image: prom/prometheus:latest
    container_name: vectorless-rag-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    networks:
      - vectorless-rag-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    container_name: vectorless-rag-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    networks:
      - vectorless-rag-network
    depends_on:
      - prometheus

volumes:
  prometheus-data:
  grafana-data:
```

### 4. Environment Configuration

**.env.example**
```bash
# Environment
ENVIRONMENT=development
BUILD_TARGET=development

# API Configuration
API_PORT=8000
SECRET_KEY=your-super-secret-key-change-this-in-production

# Database Configuration
MONGODB_PORT=27017
MONGODB_ROOT_USERNAME=admin
MONGODB_ROOT_PASSWORD=your-mongodb-password

# MinIO Configuration
MINIO_API_PORT=9000
MINIO_CONSOLE_PORT=9001
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=your-minio-secret-key
MINIO_BROWSER_REDIRECT_URL=http://localhost:9001

# Redis Configuration
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password

# AI Service API Keys
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
GEMINI_API_KEY=your-gemini-api-key

# Nginx Configuration (Production)
NGINX_HTTP_PORT=80
NGINX_HTTPS_PORT=443

# Data Paths
DATA_PATH=./data

# Monitoring (Production)
GRAFANA_PASSWORD=your-grafana-password
```

### 5. Service Configuration Files

**mongodb/mongod.conf**
```yaml
# MongoDB Configuration
storage:
  dbPath: /data/db
  journal:
    enabled: true

systemLog:
  destination: file
  logAppend: true
  path: /var/log/mongodb/mongod.log
  logRotate: reopen

net:
  port: 27017
  bindIp: 0.0.0.0

processManagement:
  timeZoneInfo: /usr/share/zoneinfo

security:
  authorization: enabled

replication:
  replSetName: rs0

setParameter:
  enableLocalhostAuthBypass: false
```

**redis/redis.conf**
```conf
# Redis Configuration
bind 0.0.0.0
port 6379
protected-mode yes
requirepass ${REDIS_PASSWORD}

# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
```

**nginx/nginx.conf**
```nginx
events {
    worker_connections 1024;
}

http {
    upstream fastapi_backend {
        server fastapi:8000;
    }

    server {
        listen 80;
        server_name localhost;

        client_max_body_size 50M;

        location / {
            proxy_pass http://fastapi_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        location /health {
            proxy_pass http://fastapi_backend/health;
        }
    }
}
```

## Integration Points

### 1. Application Dockerfile

**Dockerfile**
```dockerfile
# Multi-stage build for FastAPI application
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development
RUN pip install --no-cache-dir pytest pytest-asyncio black isort
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Production stage
FROM base as production
COPY . .
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. Initialization Scripts

**mongodb/init/01-init-db.js**
```javascript
// MongoDB initialization script
db = db.getSiblingDB('vectorless_rag');

// Create collections
db.createCollection('documents');
db.createCollection('trees');
db.createCollection('users');
db.createCollection('query_logs');
db.createCollection('patch_history');

// Create indexes
db.documents.createIndex({ "user_id": 1, "created_at": -1 });
db.documents.createIndex({ "status": 1 });
db.trees.createIndex({ "document_id": 1 });
db.trees.createIndex({ "user_id": 1 });
db.query_logs.createIndex({ "user_id": 1, "created_at": -1 });

print('Database initialized successfully');
```

### 3. Health Check Scripts

**scripts/health-check.sh**
```bash
#!/bin/bash
# Health check script for all services

echo "Checking service health..."

# Check FastAPI
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ FastAPI is healthy"
else
    echo "✗ FastAPI is unhealthy"
    exit 1
fi

# Check MongoDB
if docker exec vectorless-rag-mongodb mongosh --eval "db.adminCommand('ping')" > /dev/null 2>&1; then
    echo "✓ MongoDB is healthy"
else
    echo "✗ MongoDB is unhealthy"
    exit 1
fi

# Check MinIO
if curl -f http://localhost:9000/minio/health/live > /dev/null 2>&1; then
    echo "✓ MinIO is healthy"
else
    echo "✗ MinIO is unhealthy"
    exit 1
fi

# Check Redis
if docker exec vectorless-rag-redis redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is healthy"
else
    echo "✗ Redis is unhealthy"
    exit 1
fi

echo "All services are healthy!"
```

## Development Tasks

### Phase 1: Basic Setup (Week 1)
- [ ] Create main docker-compose.yml configuration
- [ ] Setup development environment with docker-compose.dev.yml
- [ ] Configure service networking and volumes
- [ ] Implement basic health checks
- [ ] Create initialization scripts

### Phase 2: Production Ready (Week 2)
- [ ] Create production configuration with secrets management
- [ ] Add Nginx reverse proxy configuration
- [ ] Implement monitoring with Prometheus and Grafana
- [ ] Add backup and restore procedures
- [ ] Configure logging and log rotation

### Phase 3: Scaling & Optimization (Week 3)
- [ ] Implement horizontal scaling configurations
- [ ] Add load balancing and failover
- [ ] Optimize resource allocation and limits
- [ ] Add automated deployment scripts
- [ ] Implement comprehensive monitoring

## Testing Strategy

### Container Testing
```bash
# Test service startup
docker-compose up -d
docker-compose ps

# Test health checks
./scripts/health-check.sh

# Test service communication
docker-compose exec fastapi curl http://mongodb:27017
docker-compose exec fastapi curl http://minio:9000/minio/health/live
```

### Integration Testing
```bash
# Test full stack
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
pytest tests/integration/
```

## Performance Considerations

### Resource Optimization
- Container resource limits and reservations
- Volume mount optimization
- Network performance tuning
- Memory and CPU allocation

### Scaling Strategies
- Horizontal scaling with replicas
- Load balancing configuration
- Database clustering
- Cache optimization

## Security Requirements

### Container Security
- Non-root user execution
- Secrets management
- Network isolation
- Image vulnerability scanning

### Data Protection
- Volume encryption
- Secure inter-service communication
- Access control and authentication
- Regular security updates

## Deployment Configuration

### Development Deployment
```bash
# Start development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# View logs
docker-compose logs -f fastapi

# Stop environment
docker-compose down
```

### Production Deployment
```bash
# Deploy to production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale services
docker-compose up -d --scale fastapi=3

# Update services
docker-compose up -d --no-deps fastapi
```

### Backup and Restore
```bash
# Backup data
./scripts/backup.sh

# Restore data
./scripts/restore.sh backup-2024-01-01.tar.gz
```

## Monitoring & Logging

### Service Monitoring
- Container health checks
- Resource usage monitoring
- Performance metrics collection
- Alert configuration

### Log Management
- Centralized logging with ELK stack
- Log rotation and retention
- Error tracking and alerting
- Performance log analysis

### Metrics Collection
```yaml
# Prometheus configuration
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']
  
  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
```

---

*This Docker Compose implementation guide provides comprehensive containerization for the vectorless RAG system. Follow the deployment phases and configuration guidelines for optimal performance and security.*