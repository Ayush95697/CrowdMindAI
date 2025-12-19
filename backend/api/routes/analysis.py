from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from core.processor import CrowdMindProcessor
from core.config import settings
import cv2
import numpy as np
import base64
import json
import asyncio
from pathlib import Path

router = APIRouter(prefix="/analysis", tags=["analysis"])

# Global processor instance
processor = None

def get_processor():
    global processor
    if processor is None:
        processor = CrowdMindProcessor()
    return processor

@router.websocket("/ws/{video_id}")
async def websocket_analysis(websocket: WebSocket, video_id: str):
    """
    WebSocket endpoint for real-time video analysis
    Streams processed frames and statistics to the client
    """
    await websocket.accept()
    
    try:
        # Get processor
        proc = get_processor()
        
        # Determine video path - check if it's a sample or uploaded video
        sample_path = settings.BASE_DIR / "Samples" / f"{video_id}.mp4"
        upload_path = settings.UPLOADS_DIR / f"{video_id}.mp4"
        
        video_path = None
        if sample_path.exists():
            video_path = sample_path
        elif upload_path.exists():
            video_path = upload_path
        else:
            # Try other extensions
            for ext in settings.SUPPORTED_VIDEO_FORMATS:
                test_path = settings.UPLOADS_DIR / f"{video_id}.{ext}"
                if test_path.exists():
                    video_path = test_path
                    break
        
        if video_path is None:
            await websocket.send_json({
                "error": "Video not found"
            })
            await websocket.close()
            return
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            await websocket.send_json({
                "error": "Failed to open video"
            })
            await websocket.close()
            return
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            
            # Loop video if it ends
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process frame
            result_frame, people_count, risk_class, risk_percent, density_map = proc.process_frame(frame)
            
            # Calculate zone counts (simplified - 52% in Zone A)
            zone_a_count = int(people_count * 0.52)
            zone_b_count = people_count - zone_a_count
            
            # Generate heatmap
            heatmap_img = None
            if density_map is not None:
                density_map_np = density_map.squeeze().cpu().numpy()
                h, w = frame.shape[:2]
                
                density_map_resized = cv2.resize(density_map_np, (w, h), interpolation=cv2.INTER_LINEAR)
                density_map_blurred = cv2.GaussianBlur(density_map_resized, (21, 21), 0)
                
                normed_map = cv2.normalize(density_map_blurred, None, 0, 255, cv2.NORM_MINMAX)
                heatmap = cv2.applyColorMap(np.uint8(normed_map), cv2.COLORMAP_JET)
                
                overlay_heatmap = cv2.addWeighted(result_frame, 0.5, heatmap, 0.5, 0)
                
                # Encode heatmap as base64
                _, heatmap_buffer = cv2.imencode('.jpg', overlay_heatmap)
                heatmap_img = base64.b64encode(heatmap_buffer).decode('utf-8')
            
            # Encode result frame as base64
            _, buffer = cv2.imencode('.jpg', result_frame)
            frame_img = base64.b64encode(buffer).decode('utf-8')
            
            # Calculate redistribution if high risk
            redistribution = None
            if risk_class == 2:
                redistribution_percent = int((zone_a_count - zone_b_count) / 2 / people_count * 100) if people_count > 0 else 0
                redistribution = f"Move {redistribution_percent}% from Zone A to Zone B"
            
            # Send data to client
            await websocket.send_json({
                "frame": frame_img,
                "heatmap": heatmap_img,
                "people_count": people_count,
                "zone_a": zone_a_count,
                "zone_b": zone_b_count,
                "risk_class": risk_class,
                "risk_label": proc.risk_labels.get(risk_class, "Unknown"),
                "risk_percent": risk_percent,
                "redistribution": redistribution,
                "frame_number": frame_count
            })
            
            frame_count += 1
            
            # Small delay to control frame rate
            await asyncio.sleep(0.1)
            
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for video {video_id}")
    except Exception as e:
        print(f"Error in WebSocket: {e}")
        await websocket.send_json({"error": str(e)})
    finally:
        if 'cap' in locals():
            cap.release()

@router.get("/sample-videos")
async def get_sample_videos():
    """
    Get list of available sample videos
    """
    samples_dir = settings.BASE_DIR / "Samples"
    if not samples_dir.exists():
        return {"samples": []}
    
    samples = []
    for video_file in samples_dir.glob("*.mp4"):
        samples.append({
            "id": video_file.stem,
            "name": video_file.name,
            "path": str(video_file)
        })
    
    return {"samples": samples}
