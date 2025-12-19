from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API Settings
    API_TITLE: str = "CrowdMind AI API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: list = ["http://localhost:3000", "http://127.0.0.1:3000"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    ASSETS_DIR: Path = BASE_DIR / "assets"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # Model Paths
    CSRNET_MODEL_PATH: Path = ASSETS_DIR / "partB_model_best.pth.tar"
    RISK_MODEL_PATH: Path = ASSETS_DIR / "crowd_risk_dl.pth"
    
    # Processing Settings
    MAX_VIDEO_SIZE_MB: int = 500
    SUPPORTED_VIDEO_FORMATS: list = ["mp4", "avi", "mov"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()

# Create required directories
settings.UPLOADS_DIR.mkdir(exist_ok=True)
