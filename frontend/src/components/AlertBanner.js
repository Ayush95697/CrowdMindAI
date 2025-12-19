import React from 'react';

const AlertBanner = ({ riskClass, show }) => {
    if (!show || riskClass !== 2) return null;

    return (
        <div className="alert-banner alert-danger">
            <div className="alert-content">
                <svg className="alert-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2L1 21h22L12 2zm0 3.5L19.5 19h-15L12 5.5zM11 10v4h2v-4h-2zm0 5v2h2v-2h-2z" />
                </svg>
                <span className="alert-text">⚠️ ALERT: High crowd density detected in Zone A</span>
            </div>
        </div>
    );
};

export default AlertBanner;
