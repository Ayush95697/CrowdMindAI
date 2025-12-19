import React from 'react';

const StatsPanel = ({ stats }) => {
    const {
        people_count = 0,
        zone_a = 0,
        zone_b = 0,
        risk_class = 0,
        risk_label = 'Low',
        redistribution = null
    } = stats || {};

    const getRiskColor = (riskClass) => {
        switch (riskClass) {
            case 2: return '#ef4444'; // red
            case 1: return '#f59e0b'; // orange
            default: return '#10b981'; // green
        }
    };

    return (
        <div className="stats-panel">
            <h3 className="panel-title">Live Statistics</h3>

            <div className="stat-card total-count">
                <div className="stat-label">Total People Count</div>
                <div className="stat-value">{people_count}</div>
            </div>

            <div className="zone-stats">
                <h4 className="section-title">Zone-wise Density</h4>
                <div className="stat-card">
                    <div className="stat-label">Zone A</div>
                    <div className="stat-value zone-value">{zone_a}</div>
                </div>
                <div className="stat-card">
                    <div className="stat-label">Zone B</div>
                    <div className="stat-value zone-value">{zone_b}</div>
                </div>
            </div>

            <div className="risk-section">
                <h4 className="section-title">Stampede Risk Level</h4>
                <div className="risk-card" style={{ borderColor: getRiskColor(risk_class) }}>
                    <div
                        className="risk-label"
                        style={{ color: getRiskColor(risk_class) }}
                    >
                        {risk_label}
                    </div>
                    <div className="risk-indicator">
                        <div className="risk-bar">
                            <div
                                className="risk-bar-fill"
                                style={{
                                    width: `${(risk_class / 2) * 100}%`,
                                    backgroundColor: getRiskColor(risk_class)
                                }}
                            ></div>
                        </div>
                    </div>
                </div>
            </div>

            <div className="redistribution-section">
                <h4 className="section-title">Redistribution Suggestion</h4>
                <div className="redistribution-card">
                    {redistribution ? (
                        <p className="redistribution-text">{redistribution}</p>
                    ) : (
                        <p className="redistribution-text no-action">No redistribution needed</p>
                    )}
                </div>
            </div>
        </div>
    );
};

export default StatsPanel;
