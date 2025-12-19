# CrowdMind AI

## Project Overview
Real-time crowd analysis and risk assessment system with modern FastAPI backend and React frontend.

## Quick Start

### Option 1: Start Everything (Recommended)
Double-click **`start-all.bat`** to launch both backend and frontend servers automatically.

### Option 2: Manual Start

**Backend:**
```bash
cd backend
venv\Scripts\activate
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install  # First time only
npm start
```

## Project Structure

```
CrowdMind AI/
├── backend/          # FastAPI backend with AI models
├── frontend/         # React frontend with modern UI
├── start-all.bat     # Launch both servers
└── README.md         # This file
```

## Features

- 🎥 Real-time video processing and crowd analysis
- 🔥 Density heatmap visualization
- 📊 Live statistics dashboard
- ⚠️ Stampede risk assessment (Low/Medium/High)
- 🚀 GPU acceleration support
- 🎨 Modern dark theme with glassmorphism effects
- 📱 Responsive design

## Documentation

- [Backend README](backend/README.md) - API documentation and setup
- [Frontend README](frontend/README.md) - React app details
- [Walkthrough](https://github.com/...) - Full project details

## Technology Stack

**Backend:** FastAPI, PyTorch, OpenCV, Python 3.8+  
**Frontend:** React 18, WebSocket, Canvas API  
**AI Models:** CSRNet (density), CrowdRiskClassifier (risk)

## Requirements

- Python 3.8+
- Node.js 14+
- 8GB RAM minimum
- NVIDIA GPU (optional, recommended for performance)

## URLs After Starting

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## License

Proprietary - CrowdMind AI Project
