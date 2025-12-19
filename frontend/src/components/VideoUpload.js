import React, { useState } from 'react';
import { uploadVideo, getSampleVideos } from '../utils/api';

const VideoUpload = ({ onVideoSelect }) => {
    const [uploading, setUploading] = useState(false);
    const [dragActive, setDragActive] = useState(false);
    const [samples, setSamples] = useState([]);
    const [showSamples, setShowSamples] = useState(false);

    React.useEffect(() => {
        loadSamples();
    }, []);

    const loadSamples = async () => {
        try {
            const data = await getSampleVideos();
            setSamples(data.samples || []);
        } catch (error) {
            console.error('Failed to load sample videos:', error);
        }
    };

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = async (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            await handleFile(e.dataTransfer.files[0]);
        }
    };

    const handleChange = async (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            await handleFile(e.target.files[0]);
        }
    };

    const handleFile = async (file) => {
        setUploading(true);
        try {
            const result = await uploadVideo(file);
            onVideoSelect(result.video_id);
        } catch (error) {
            console.error('Upload failed:', error);
            alert('Failed to upload video. Please try again.');
        } finally {
            setUploading(false);
        }
    };

    const handleSampleSelect = (sampleId) => {
        onVideoSelect(sampleId);
        setShowSamples(false);
    };

    return (
        <div className="video-upload-section">
            <div
                className={`upload-area ${dragActive ? 'drag-active' : ''} ${uploading ? 'uploading' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
            >
                <input
                    type="file"
                    id="file-upload"
                    accept=".mp4,.mov,.avi"
                    onChange={handleChange}
                    disabled={uploading}
                    style={{ display: 'none' }}
                />
                <label htmlFor="file-upload" className="upload-label">
                    {uploading ? (
                        <>
                            <div className="upload-spinner"></div>
                            <p>Uploading...</p>
                        </>
                    ) : (
                        <>
                            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                            <p className="upload-text">Drop video here or click to upload</p>
                            <p className="upload-hint">Supports MP4, MOV, AVI (max 500MB)</p>
                        </>
                    )}
                </label>
            </div>

            <div className="sample-videos-section">
                <button
                    className="btn-samples"
                    onClick={() => setShowSamples(!showSamples)}
                >
                    {showSamples ? 'Hide' : 'Show'} Sample Videos
                </button>

                {showSamples && samples.length > 0 && (
                    <div className="samples-list">
                        {samples.map((sample) => (
                            <button
                                key={sample.id}
                                className="sample-item"
                                onClick={() => handleSampleSelect(sample.id)}
                            >
                                <svg className="sample-icon" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M8 5v14l11-7z" />
                                </svg>
                                <span>{sample.name}</span>
                            </button>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
};

export default VideoUpload;
