# 🐳 Docker Deployment Guide

## 📋 Prerequisites

1. Docker installed and running
2. Docker Compose (optional but recommended)
3. Your trained model files in `saved_model_components/`

## 🚀 Quick Start - Docker

### Option 1: Simple Docker Run

```bash
# Build the Docker image
docker build -t book-recommender .

# Run the container
docker run -d \
  --name book-api \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/saved_model_components:/app/saved_model_components \
  book-recommender

# Check if it's running
docker ps

# View logs
docker logs book-api

# Stop the container
docker stop book-api

# Remove the container
docker rm book-api
```

### Option 2: Using Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up --build -d
```

## 📁 Docker Files

### Dockerfile
```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data saved_model_components

# Set environment variables
ENV PYTHONPATH=/app
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: '3.8'

services:
  book-recommender:
    build: .
    container_name: book-recommendation-api
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DEBUG=false
      - MAX_RECOMMENDATIONS=50
      - DEFAULT_RECOMMENDATIONS=5
    volumes:
      # Mount data and model directories
      - ./data:/app/data:ro
      - ./saved_model_components:/app/saved_model_components:ro
      # Optional: Mount logs directory
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - book-network

  # Optional: Add nginx reverse proxy
  nginx:
    image: nginx:alpine
    container_name: book-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro  # SSL certificates
    depends_on:
      - book-recommender
    restart: unless-stopped
    networks:
      - book-network
    profiles:
      - with-nginx

  # Optional: Add Redis for caching
  redis:
    image: redis:alpine
    container_name: book-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped
    networks:
      - book-network
    profiles:
      - with-cache

volumes:
  redis-data:

networks:
  book-network:
    driver: bridge
```

### .dockerignore
```dockerignore
# Virtual environments
env/
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db

# Git
.git/
.gitignore

# Documentation
README.md
docs/

# Testing
tests/
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/

# Temporary files
tmp/
temp/
```

## 🔧 Docker Management Commands

### Building and Running

```bash
# Build with specific tag
docker build -t book-recommender:v1.0 .

# Build with no cache
docker build --no-cache -t book-recommender .

# Run with environment variables
docker run -d \
  --name book-api \
  -p 8000:8000 \
  -e DEBUG=true \
  -e MAX_RECOMMENDATIONS=20 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/saved_model_components:/app/saved_model_components \
  book-recommender

# Run in interactive mode (for debugging)
docker run -it \
  --name book-api-debug \
  -p 8000:8000 \
  -v $(pwd):/app \
  book-recommender bash
```

### Monitoring and Debugging

```bash
# View container logs
docker logs book-api
docker logs -f book-api  # Follow logs

# Execute commands in running container
docker exec -it book-api bash
docker exec book-api ls -la /app

# Check container stats
docker stats book-api

# Inspect container
docker inspect book-api

# Check health status
docker inspect --format='{{.State.Health.Status}}' book-api
```

### Docker Compose Commands

```bash
# Start services in background
docker-compose up -d

# Start with specific profile
docker-compose --profile with-nginx up -d

# View logs for all services
docker-compose logs

# View logs for specific service
docker-compose logs book-recommender

# Scale services
docker-compose up -d --scale book-recommender=3

# Stop services
docker-compose stop

# Remove everything
docker-compose down -v --remove-orphans

# Rebuild specific service
docker-compose build book-recommender
docker-compose up -d book-recommender
```

## 🌐 Production Docker Setup

### Multi-stage Dockerfile for Production
```dockerfile
# Multi-stage build for smaller production image
FROM python:3.9-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data saved_model_components logs

# Set environment variables
ENV PYTHONPATH=/app
ENV PATH=/root/.local/bin:$PATH
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "api.main:app", \
     "-w", "4", \
     "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--log-level", "info"]
```

### Production docker-compose.yml
```yaml
version: '3.8'

services:
  book-recommender:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: book-api-prod
    ports:
      - "8000:8000"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8000
      - DEBUG=false
      - WORKERS=4
    volumes:
      - ./data:/app/data:ro
      - ./saved_model_components:/app/saved_model_components:ro
      - ./logs:/app/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - book-network

  nginx:
    image: nginx:alpine
    container_name: book-nginx-prod
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - book-recommender
    restart: unless-stopped
    networks:
      - book-network

  redis:
    image: redis:alpine
    container_name: book-redis-prod
    volumes:
      - redis-data:/data
      - ./redis.conf:/usr/local/etc/redis/redis.conf:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    restart: unless-stopped
    networks:
      - book-network

volumes:
  redis-data:
    driver: local

networks:
  book-network:
    driver: bridge
```

## 🚀 Deployment Workflows

### Development Workflow
```bash
# 1. Make changes to code
# 2. Test locally
python api/main.py

# 3. Build and test in Docker
docker-compose up --build

# 4. Test API
curl http://localhost:8000/health
```

### Production Deployment Workflow
```bash
# 1. Tag your release
git tag v1.0.0

# 2. Build production image
docker build -f Dockerfile.prod -t book-recommender:v1.0.0 .

# 3. Push to registry (if using)
docker tag book-recommender:v1.0.0 your-registry.com/book-recommender:v1.0.0
docker push your-registry.com/book-recommender:v1.0.0

# 4. Deploy to production
docker-compose -f docker-compose.prod.yml up -d

# 5. Health check
curl http://your-domain.com/health
```

## 🔐 Security Best Practices

### Docker Security
```bash
# Use non-root user in container
USER app

# Scan for vulnerabilities
docker scan book-recommender

# Use specific base image versions
FROM python:3.9.17-slim

# Limit container resources
docker run --memory="1g" --cpus="1" book-recommender
```

### Environment Security
```bash
# Use secrets for sensitive data
echo "secret_key_here" | docker secret create api_secret -

# Use read-only mounts
-v $(pwd)/data:/app/data:ro

# Network isolation
docker network create --driver bridge book-secure-network
```

## 📊 Monitoring Docker Deployment

### Container Monitoring
```bash
# Monitor resource usage
docker stats

# Export metrics
docker run -d \
  --name cadvisor \
  -p 8080:8080 \
  -v /:/rootfs:ro \
  -v /var/run:/var/run:ro \
  -v /sys:/sys:ro \
  -v /var/lib/docker/:/var/lib/docker:ro \
  google/cadvisor:latest
```

### Log Management
```bash
# Configure logging driver
docker run -d \
  --log-driver=json-file \
  --log-opt max-size=10m \
  --log-opt max-file=3 \
  book-recommender

# Use centralized logging
docker run -d \
  --log-driver=fluentd \
  --log-opt fluentd-address=localhost:24224 \
  book-recommender
```