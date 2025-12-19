import React, { useEffect, useState } from 'react';

const Header = () => {
    const [currentTime, setCurrentTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => {
            setCurrentTime(new Date());
        }, 1000);

        return () => clearInterval(timer);
    }, []);

    const formatTime = (date) => {
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false
        });
    };

    return (
        <header className="app-header">
            <div className="header-content">
                <div className="header-left">
                    <h1 className="app-title">CrowdMind AI</h1>
                    <p className="app-subtitle">Real-time Crowd Analysis Dashboard</p>
                </div>
                <div className="header-right">
                    <div className="live-indicator">
                        <span className="live-dot"></span>
                        <span className="live-text">LIVE</span>
                    </div>
                    <div className="clock">
                        {formatTime(currentTime)}
                    </div>
                </div>
            </div>
        </header>
    );
};

export default Header;
