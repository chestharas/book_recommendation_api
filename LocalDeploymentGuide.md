# 🖥️ Local Deployment Guide

## 📋 Prerequisites

1. Python 3.8+ installed
2. pip package manager
3. Virtual environment (recommended)

## 🚀 Quick Start - Local Setup

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv env

# Activate virtual environment
# Windows:
env\Scripts\activate
# Mac/Linux:
source env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Prepare Data
```bash
# Make sure your books.csv is in the data/ folder
# Your structure should be:
# BOOK_RECOMMENDATION_API/
# ├── data/
# │   └── books.csv  # <-- Your dataset here
# └── ...
```

### Step 3: Train Model
```bash
# Option 1: Using the training script directly
python models/train_model.py

# Option 2: Using the runner script
python scripts/run_training.py
```

### Step 4: Start API Server
```bash
# Option 1: Using the main file
python api/main.py

# Option 2: Using uvicorn directly
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Using the runner script
python scripts/run_api.py
```

### Step 5: Test the API
```bash
# Check health
curl http://localhost:8000/health

# Get recommendations
curl "http://localhost:8000/recommendations/1?num_recommendations=5"

# Visit documentation
# http://localhost:8000/docs
```

## 🔧 Troubleshooting Local Setup

### Common Issues and Solutions

#### Issue 1: Module Import Errors
```bash
# Make sure you're in the project root directory
cd BOOK_RECOMMENDATION_API

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# Windows:
set PYTHONPATH=%PYTHONPATH%;%cd%
```

#### Issue 2: Model Files Not Found
```bash
# Check if model files exist
ls saved_model_components/

# If missing, retrain the model
python models/train_model.py
```

#### Issue 3: scikit-learn Version Conflicts
```bash
# Update scikit-learn
pip install --upgrade scikit-learn

# Or install specific version
pip install scikit-learn==1.6.1
```

#### Issue 4: Port Already in Use
```bash
# Use different port
uvicorn api.main:app --host 0.0.0.0 --port 8001

# Or kill process using port 8000
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Mac/Linux:
lsof -ti:8000 | xargs kill -9
```

## 📝 Environment Variables (.env file)

Create a `.env` file in your project root:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Model Configuration
MAX_RECOMMENDATIONS=50
DEFAULT_RECOMMENDATIONS=5

# Data Paths
DATA_DIR=data
MODEL_DIR=saved_model_components

# Performance
WORKERS=1
```

## 🎯 Development Mode

For development with auto-reload:

```bash
# With auto-reload (recommended for development)
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# With specific settings
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level info \
  --access-log
```

## 🏗️ Production Mode (Local)

For production deployment on local server:

```bash
# Install gunicorn for production
pip install gunicorn

# Run with gunicorn (better for production)
gunicorn api.main:app \
  -w 4 \
  -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --log-level info \
  --access-logfile - \
  --error-logfile -

# Or with specific configuration
gunicorn api.main:app \
  --config gunicorn.conf.py
```

Create `gunicorn.conf.py`:
```python
# Gunicorn configuration file
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 50
preload_app = True
keepalive = 2
timeout = 30
```

## 🧪 Testing Local Setup

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=api --cov-report=html

# Test specific endpoint
python -c "
import requests
response = requests.get('http://localhost:8000/health')
print(response.json())
"
```

## 📊 Monitoring Local Deployment

### Check API Status
```bash
# Health check
curl http://localhost:8000/health

# Get stats
curl http://localhost:8000/debug/stats

# Check all endpoints
curl http://localhost:8000/
```

### Performance Testing
```bash
# Install apache bench
# Ubuntu: sudo apt-get install apache2-utils
# Mac: brew install apache2

# Test performance
ab -n 1000 -c 10 http://localhost:8000/health

# Test recommendations endpoint
ab -n 100 -c 5 "http://localhost:8000/recommendations/1"
```

## 🔄 Updates and Maintenance

### Update Model
```bash
# Retrain model with new data
python models/train_model.py

# Reload model in running API (if reload endpoint available)
curl -X POST http://localhost:8000/admin/reload-model
```

### Update Dependencies
```bash
# Update all packages
pip list --outdated
pip install --upgrade package_name

# Or update from requirements
pip install -r requirements.txt --upgrade
```

### Backup Important Files
```bash
# Backup model files
cp -r saved_model_components/ backup/saved_model_components_$(date +%Y%m%d)/

# Backup data
cp -r data/ backup/data_$(date +%Y%m%d)/
```

## 🚨 Security Considerations (Local Production)

```bash
# Create non-root user for running the service
sudo useradd -r -s /bin/false bookapi

# Set proper file permissions
chmod 644 api/*.py
chmod 600 .env

# Use firewall to restrict access
sudo ufw allow 8000/tcp
sudo ufw enable

# Consider using reverse proxy (nginx)
sudo apt-get install nginx
```

Example nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```