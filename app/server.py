from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classifier import train_face_classifier
from .config import Settings, load_dotenv_if_available
from .database import get_record, list_records, verify_offline_record
from .service import PipelineDossier, run_pipeline_service

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path("app/static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="FaceWebChain API",
    description="Privacy-preserving face discovery, social verification, and blockchain integrity proof.",
    version="2.0.0",
)

# Enable CORS for frontend integration (Vite, React, Vue, Vanilla HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount artifacts and static folders
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_DIR)), name="artifacts")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ScanJsonRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image or data URL")
    threshold: float = Field(0.5, description="Face comparison distance threshold")
    detector_model: str = Field("hog", description="Face detector: hog or cnn")
    upsample_times: int = Field(0, description="Upsample count for small faces")
    skip_blockchain: bool = Field(False, description="Bypass Sepolia on-chain registration")
    require_liveness: bool = Field(False, description="Enforce strict anti-spoofing gating")


class TrainRequest(BaseModel):
    dataset_dir: str = Field("data/authorized_faces", description="Directory of labeled identity images")
    model_output: str = Field("models/face_model.pkl", description="Path to save trained weights")
    model: str = Field("hog", description="Detection model")
    threshold: float = Field(0.5, description="Match threshold")


@app.get("/")
def root() -> Any:
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(str(index_path))
    return {
        "system": "FaceWebChain Core API",
        "status": "ONLINE",
        "docs": "/docs",
    }


@app.get("/api/sample")
def get_sample_image() -> dict[str, Any]:
    """Provide bundled reference image as base64 for instant demo."""
    input_dir = Path("input")
    supported = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in supported]
    if not images:
        raise HTTPException(status_code=404, detail="No sample image in input/ directory")
    sample_path = images[0]
    raw_b64 = base64.b64encode(sample_path.read_bytes()).decode("utf-8")
    mime = "image/jpeg" if sample_path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    return {
        "filename": sample_path.name,
        "image": f"data:{mime};base64,{raw_b64}",
        "size_bytes": sample_path.stat().st_size,
    }


@app.get("/api/health")
def health() -> dict[str, Any]:
    load_dotenv_if_available()
    settings = Settings.from_environment()
    has_google = bool(settings.google_credentials)
    has_sepolia = bool(settings.sepolia_rpc_url and settings.private_key and settings.contract_address)
    records = list_records(limit=1)

    return {
        "status": "healthy",
        "google_vision_configured": has_google,
        "sepolia_configured": has_sepolia,
        "total_records": len(list_records(limit=1000)),
        "explorer_base_url": settings.explorer_base_url,
    }


@app.post("/api/scan")
def scan_face(payload: ScanJsonRequest) -> dict[str, Any]:
    """Process a face scan via Base64 payload, discover social matches, and verify on-chain."""
    try:
        dossier: PipelineDossier = run_pipeline_service(
            image_input=payload.image,
            output_dir=OUTPUT_DIR,
            threshold=payload.threshold,
            detector_model=payload.detector_model,
            upsample_times=payload.upsample_times,
            skip_blockchain=payload.skip_blockchain,
            require_liveness=payload.require_liveness,
        )
        return dossier.to_dict()
    except (FileNotFoundError, LookupError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/scan/upload")
async def scan_face_upload(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
    detector_model: str = Form("hog"),
    upsample_times: int = Form(0),
    skip_blockchain: bool = Form(False),
    require_liveness: bool = Form(False),
) -> dict[str, Any]:
    """Process a face scan via direct multipart file upload."""
    content = await file.read()
    try:
        dossier: PipelineDossier = run_pipeline_service(
            image_input=content,
            output_dir=OUTPUT_DIR,
            threshold=threshold,
            detector_model=detector_model,
            upsample_times=upsample_times,
            skip_blockchain=skip_blockchain,
            require_liveness=require_liveness,
        )
        return dossier.to_dict()
    except (FileNotFoundError, LookupError, ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/records")
def get_records(limit: int = 50) -> list[dict[str, Any]]:
    """Fetch historical evidence records from the offline database."""
    return list_records(limit=limit)


@app.get("/api/records/{record_id}")
def get_single_record(record_id: int) -> dict[str, Any]:
    """Retrieve details for a specific record."""
    record = get_record(record_id)
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    return record


@app.get("/api/verify/{record_id}")
def verify_record(record_id: int) -> dict[str, Any]:
    """Re-verify an evidence record against on-disk hashes."""
    try:
        return verify_offline_record(record_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/train")
def train_model(payload: TrainRequest) -> dict[str, Any]:
    """Train a custom face recognition model from a directory."""
    try:
        classifier = train_face_classifier(
            dataset_dir=Path(payload.dataset_dir),
            output_model_path=Path(payload.model_output),
            model=payload.model,
            threshold=payload.threshold,
        )
        return {
            "status": "TRAINED",
            "model_path": payload.model_output,
            "classes": classifier.classes,
            "sample_count": len(classifier.labels),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
