import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Video API
export const uploadVideo = async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/videos/upload', formData, {
        headers: {
            'Content-Type': 'multipart/form-data',
        },
    });

    return response.data;
};

export const getVideos = async () => {
    const response = await api.get('/api/videos/');
    return response.data;
};

export const deleteVideo = async (videoId) => {
    const response = await api.delete(`/api/videos/${videoId}`);
    return response.data;
};

export const getSampleVideos = async () => {
    const response = await api.get('/api/analysis/sample-videos');
    return response.data;
};

// WebSocket URL
export const getWebSocketUrl = (videoId) => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsHost = process.env.REACT_APP_WS_URL || 'localhost:8000';
    return `${wsProtocol}//${wsHost}/api/analysis/ws/${videoId}`;
};
