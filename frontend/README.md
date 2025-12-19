# CrowdMind AI - Frontend

## Overview
Modern React-based frontend for real-time crowd analysis dashboard with WebSocket streaming.

## Features
- Real-time video feed display
- Density heatmap visualization
- Live statistics dashboard
- WebSocket integration for streaming data
- Drag-and-drop video upload
- Sample video selection
- Modern dark theme with glassmorphism
- Responsive design

## Setup

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Environment Variables (Optional)
Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=localhost:8000
```

### 3. Start Development Server
```bash
npm start
```

The app will open at `http://localhost:3000`

### 4. Build for Production
```bash
npm run build
```

## Project Structure
```
frontend/
├── public/
│   └── index.html
├── src/
│   ├── components/
│   │   ├── Header.js          # Header with branding and clock
│   │   ├── VideoUpload.js     # Upload and sample selection
│   │   ├── LiveFeed.js        # Video feed and heatmap display
│   │   ├── StatsPanel.js      # Statistics dashboard
│   │   └── AlertBanner.js     # Risk alerts
│   ├── hooks/
│   │   └── useWebSocket.js    # WebSocket hook
│   ├── utils/
│   │   └── api.js             # API utilities
│   ├── App.js                 # Main component
│   ├── App.css                # Styles
│   ├── index.js               # Entry point
│   └── index.css              # Global styles
└── package.json
```

## API Integration
The frontend connects to the backend API at `http://localhost:8000` by default.

- Video uploads: `POST /api/videos/upload`
- WebSocket stream: `ws://localhost:8000/api/analysis/ws/{video_id}`
- Sample videos: `GET /api/analysis/sample-videos`

## Design Features
- Dark theme with gradient accents
- Glassmorphism effects
- Smooth animations and transitions
- Responsive grid layout
- Color-coded risk levels (green/orange/red)
- Real-time clock display
- Connection status indicators

## Technologies
- React 18
- Axios for HTTP requests
- WebSocket for real-time updates
- Canvas API for video rendering
- CSS3 with custom properties
