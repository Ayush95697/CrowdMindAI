from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from core.config import settings
from pathlib import Path
import shutil
import uuid

router = APIRouter(prefix="/videos", tags=["videos"])

# Store uploaded videos metadata
videos_db = {}

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing
    """
    # Validate file extension
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {settings.SUPPORTED_VIDEO_FORMATS}"
        )
    
    # Generate unique video ID
    video_id = str(uuid.uuid4())
    video_filename = f"{video_id}.{file_ext}"
    video_path = settings.UPLOADS_DIR / video_filename
    
    # Save file
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")
    
    # Store metadata
    videos_db[video_id] = {
        "id": video_id,
        "filename": file.filename,
        "path": str(video_path),
        "status": "uploaded"
    }
    
    return JSONResponse(content={
        "video_id": video_id,
        "filename": file.filename,
        "message": "Video uploaded successfully"
    })

@router.get("/")
async def list_videos():
    """
    Get list of all uploaded videos
    """
    return {"videos": list(videos_db.values())}

@router.get("/{video_id}")
async def get_video(video_id: str):
    """
    Get metadata for a specific video
    """
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return videos_db[video_id]

@router.delete("/{video_id}")
async def delete_video(video_id: str):
    """
    Delete a video
    """
    if video_id not in videos_db:
        raise HTTPException(status_code=404, detail="Video not found")
    
    # Delete file
    video_path = Path(videos_db[video_id]["path"])
    if video_path.exists():
        video_path.unlink()
    
    # Remove from database
    del videos_db[video_id]
    
    return {"message": "Video deleted successfully"}
