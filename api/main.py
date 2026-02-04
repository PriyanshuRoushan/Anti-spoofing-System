import os
import base64
import uuid
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Header, Depends, Request
from configs.exceptions import APIException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from spoof_detection import load_model, predict
from dotenv import load_dotenv
from contextlib import asynccontextmanager


load_dotenv()   # Reads .env into os.environ


# ===========================
# CONFIG
# ===========================

API_KEY = None # Move to ENV in production
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ===========================
# LOGGING CONFIG
# ===========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ===========================
# FASTAPI APP & APP STARTUP/SHUTDOWN
# ===========================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    require_env("API_TOKEN")

    print("Loading model...")
    model = load_model()
    logger.info("Model loaded successfully")

    yield  # App runs here

    logger.info("Shutting down...")



app = FastAPI(
    title="Voice Detection API",
    version="1.0.0",
    docs_url="/docs", 
    lifespan=lifespan
)

# =====================================================
# GLOBAL MODEL (LOADED ONCE PER WORKER)
# =====================================================

model = None



# ===========================
# REQUEST MODEL
# ===========================

class VoiceDetectionRequest(BaseModel):
    language: str = Field(
        ...,
        json_schema_extra={"example": "Tamil"}
    )
    audioFormat: str = Field(
        ...,
        json_schema_extra={"example": "mp3"}
    )
    audioBase64: str = Field(..., min_length=100)



# ===========================
# RESPONSE MODEL
# ===========================

class VoiceDetectionResponse(BaseModel):
    request_id: str
    status: str = "success"
    language: str
    classification: str
    confidenceScore: float
    explanation: str
    processing_time_ms: int


# ===========================
# AUTH DEPENDENCY
# ===========================


def require_env(name: str) -> str:

    value = os.getenv(name)

    if not value:
        raise RuntimeError(f"Missing env variable: {name}")

    return value

def verify_api_key(x_api_key: str = Header(...)):
    API_KEY = require_env("API_TOKEN")
    if x_api_key != API_KEY:
        logger.warning("Unauthorized access attempt")
        raise APIException(
            status_code=401,
            detail="Invalid API Key",
            exceptionType="UnauthorizedAccess"
        )



# @app.on_event("startup")
# async def load_model_onstartup():
#     global model
    
#     API_KEY = require_env("API_TOKEN")
#     # print(API_KEY)
#     print("Loading model...")
#     model = load_model()
#     logger.info("Model loaded successfully")


# ===========================
# UTILITIES
# ===========================

def save_base64_audio(
    base64_str: str,
    extension: str
) -> str:
    """
    Decode base64 and save audio file
    """
    try:
        audio_bytes = base64.b64decode(base64_str)
    except Exception:
        raise APIException(
            status_code=400,
            detail="Invalid base64 encoding",
            exceptionType="InvalidBase64 Audio String"
        )

    file_id = uuid.uuid4().hex
    file_path = os.path.join(
        UPLOAD_DIR,
        f"{file_id}.{extension}"
    )

    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    return file_path


def run_voice_detection_model(file_path: str) -> tuple[str, float]:
    """
    Placeholder for ML inference
    Replace with real model
    """
    genuine_confidence, spoof_confidence = predict(model, file_path)

    verdict = "HUMAN" if genuine_confidence > spoof_confidence else "AI_GENERATED"
    confidence = max(genuine_confidence, spoof_confidence)

    return verdict, confidence

# ===========================
# MIDDLEWARE: REQUEST LOGGING
# ===========================

@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = uuid.uuid4().hex

    start_time = datetime.utcnow()

    logger.info(
        f"REQUEST {request_id} | "
        f"{request.method} {request.url.path}"
    )

    response = await call_next(request)

    process_time = (
        datetime.utcnow() - start_time
    ).total_seconds() * 1000

    logger.info(
        f"RESPONSE {request_id} | "
        f"Status {response.status_code} | "
        f"{process_time:.2f} ms"
    )

    response.headers["X-Request-ID"] = request_id

    return response


# ===========================
# MAIN ROUTE
# ===========================

@app.post(
    "/api/voice-detection",
    response_model=VoiceDetectionResponse,
    dependencies=[Depends(verify_api_key)]
)
async def detect_voice(payload: VoiceDetectionRequest):

    # print(payload)
    start = datetime.utcnow()

    request_id = uuid.uuid4().hex

    logger.info(
        f"Processing request {request_id} "
        f"Language={payload.language}"
    )

    # Validate format
    if payload.audioFormat.lower() not in ["mp3", "wav", "ogg"]:
        raise APIException(
            status_code=400,
            detail="Unsupported audio format"
        )

    # Save audio
    file_path = save_base64_audio(
        payload.audioBase64,
        payload.audioFormat.lower()
    )

    logger.info(f"Saved file {file_path}")

    try:
        # Run ML model
        verdict, confidence = run_voice_detection_model(file_path)

    except Exception as e:
        logger.exception("Model inference failed")

        raise APIException(
            status_code=500,
            detail="Voice detection failed",
            exceptionType="VoiceDetectionFailed or ModelInferenceError"
        )

    finally:
        print("Cleanup placeholder")
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)

    elapsed = (
        datetime.utcnow() - start
    ).total_seconds() * 1000

    logger.info(
        f"Completed {request_id} "
        f"Verdict={verdict} "
        f"Confidence={confidence}"
    )

    return VoiceDetectionResponse(
        request_id=request_id,
        language=payload.language,
        classification=verdict,
        confidenceScore=float(confidence).__round__(2),
        explanation=f"Detected as {verdict}",
        processing_time_ms=int(elapsed)
    )


# ===========================
# HEALTH CHECK
# ===========================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"status": "running"}
