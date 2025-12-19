# CrowdMind AI - Backend

## Overview
FastAPI-based backend for real-time crowd analysis and risk assessment.

## Features
- Real-time video processing using deep learning models
- WebSocket support for streaming analysis results
- REST API for video upload and management
- GPU/CPU automatic detection
- Crowd density estimation using CSRNet
- Risk level classification

## Setup

### 1. Create Virtual Environment
```bash
cd backend
python -m venv venv
```

### 2. Activate Virtual Environment
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Server
```bash
# From backend directory
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation at `http://localhost:8000/docs`

## API Endpoints

### Video Management
- `POST /api/videos/upload` - Upload a video file
- `GET /api/videos/` - List all uploaded videos
- `GET /api/videos/{video_id}` - Get video metadata
- `DELETE /api/videos/{video_id}` - Delete a video

### Analysis
- `WebSocket /api/analysis/ws/{video_id}` - Real-time video analysis stream
- `GET /api/analysis/sample-videos` - Get available sample videos

### Health Check
- `GET /health` - Server health check

## Project Structure
```
backend/
├── api/
│   ├── main.py              # FastAPI application
│   └── routes/
│       ├── video.py         # Video management endpoints
│       └── analysis.py      # Analysis and WebSocket endpoints
├── core/
│   ├── model.py             # Neural network models
│   ├── processor.py         # Video processing engine
│   └── config.py            # Configuration
├── assets/                  # Model weights
├── uploads/                 # Uploaded videos
├── Samples/                 # Sample videos
└── requirements.txt
```

## Models
- **CSRNet**: Crowd density estimation model
- **CrowdRiskClassifier**: Risk level classification (Low/Medium/High)

## Requirements
- Python 3.8+
- CUDA-capable GPU (optional, will use CPU otherwise)
