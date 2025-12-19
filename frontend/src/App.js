import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import VideoUpload from './components/VideoUpload';
import LiveFeed from './components/LiveFeed';
import StatsPanel from './components/StatsPanel';
import AlertBanner from './components/AlertBanner';
import { useWebSocket } from './hooks/useWebSocket';
import './App.css';

function App() {
    const [videoId, setVideoId] = useState(null);
    const [analysisData, setAnalysisData] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);

    const handleWebSocketMessage = useCallback((data) => {
        setAnalysisData(data);
        setIsProcessing(true);
    }, []);

    const { isConnected, error } = useWebSocket(videoId, handleWebSocketMessage);

    const handleVideoSelect = (selectedVideoId) => {
        setVideoId(selectedVideoId);
        setIsProcessing(false);
        setAnalysisData(null);
    };

    return (
        <div className="App">
            <Header />

            <main className="main-content">
                <AlertBanner
                    riskClass={analysisData?.risk_class}
                    show={isProcessing}
                />

                <div className="content-grid">
                    <div className="left-section">
                        {!videoId ? (
                            <div className="upload-container">
                                <VideoUpload onVideoSelect={handleVideoSelect} />
                            </div>
                        ) : (
                            <>
                                <div className="connection-status">
                                    {isConnected ? (
                                        <span className="status-connected">
                                            <span className="status-dot"></span>
                                            Connected - Processing
                                        </span>
                                    ) : (
                                        <span className="status-disconnected">
                                            <span className="status-dot"></span>
                                            {error || 'Connecting...'}
                                        </span>
                                    )}
                                </div>
                                <LiveFeed
                                    frameData={analysisData?.frame}
                                    heatmapData={analysisData?.heatmap}
                                />
                                <button
                                    className="btn-change-video"
                                    onClick={() => setVideoId(null)}
                                >
                                    Change Video
                                </button>
                            </>
                        )}
                    </div>

                    <div className="right-section">
                        <StatsPanel stats={analysisData} />
                    </div>
                </div>
            </main>
        </div>
    );
}

export default App;
