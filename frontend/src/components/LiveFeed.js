import React, { useEffect, useRef } from 'react';

const LiveFeed = ({ frameData, heatmapData }) => {
    const canvasRef = useRef(null);
    const heatmapCanvasRef = useRef(null);

    useEffect(() => {
        if (frameData && canvasRef.current) {
            const canvas = canvasRef.current;
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };

            img.src = `data:image/jpeg;base64,${frameData}`;
        }
    }, [frameData]);

    useEffect(() => {
        if (heatmapData && heatmapCanvasRef.current) {
            const canvas = heatmapCanvasRef.current;
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);
            };

            img.src = `data:image/jpeg;base64,${heatmapData}`;
        }
    }, [heatmapData]);

    return (
        <div className="live-feed-container">
            <div className="feed-section">
                <h3 className="feed-title">Live Feed</h3>
                <div className="canvas-wrapper">
                    {frameData ? (
                        <canvas ref={canvasRef} className="video-canvas"></canvas>
                    ) : (
                        <div className="feed-placeholder">
                            <svg className="placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                            </svg>
                            <p>No video feed</p>
                        </div>
                    )}
                </div>
            </div>

            <div className="feed-section">
                <h3 className="feed-title">Heatmap & Crowd Flow</h3>
                <div className="canvas-wrapper">
                    {heatmapData ? (
                        <canvas ref={heatmapCanvasRef} className="video-canvas"></canvas>
                    ) : (
                        <div className="feed-placeholder">
                            <svg className="placeholder-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                            </svg>
                            <p>No heatmap data</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LiveFeed;
