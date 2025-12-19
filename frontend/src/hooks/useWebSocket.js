import { useState, useEffect, useRef } from 'react';
import { getWebSocketUrl } from '../utils/api';

export const useWebSocket = (videoId, onMessage) => {
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState(null);
    const wsRef = useRef(null);
    const reconnectTimeoutRef = useRef(null);

    useEffect(() => {
        if (!videoId) return;

        const connect = () => {
            try {
                const wsUrl = getWebSocketUrl(videoId);
                wsRef.current = new WebSocket(wsUrl);

                wsRef.current.onopen = () => {
                    console.log('WebSocket connected');
                    setIsConnected(true);
                    setError(null);
                };

                wsRef.current.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.error) {
                            setError(data.error);
                        } else {
                            onMessage(data);
                        }
                    } catch (err) {
                        console.error('Failed to parse WebSocket message:', err);
                    }
                };

                wsRef.current.onerror = (event) => {
                    console.error('WebSocket error:', event);
                    setError('Connection error');
                };

                wsRef.current.onclose = () => {
                    console.log('WebSocket disconnected');
                    setIsConnected(false);

                    // Attempt to reconnect after 3 seconds
                    reconnectTimeoutRef.current = setTimeout(() => {
                        console.log('Attempting to reconnect...');
                        connect();
                    }, 3000);
                };
            } catch (err) {
                console.error('Failed to create WebSocket:', err);
                setError('Failed to connect');
            }
        };

        connect();

        return () => {
            if (reconnectTimeoutRef.current) {
                clearTimeout(reconnectTimeoutRef.current);
            }
            if (wsRef.current) {
                wsRef.current.close();
            }
        };
    }, [videoId, onMessage]);

    return { isConnected, error };
};
