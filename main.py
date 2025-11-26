from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import tempfile
import uuid

from analysis import analyze_video

app = FastAPI(title="CourtIQ – CV Backend v0")

# Allow your frontend (Base44, local dev, etc.) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # in production, restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health():
    return {
        "status": "ok",
        "message": "CourtIQ CV backend v0 running.",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...),
    jersey_color: str = Form("royal-blue"),
    game_title: str = Form("Untitled Game"),
    opponent: str = Form("Unknown Opponent"),
):
    """
    Entry point for your frontend:
    - Saves the uploaded video temporarily
    - Runs analyze_video(video_path, jersey_color)
    - Returns a JSON scouting report
    """

    # Save uploaded file to a temporary location
    suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "mp4")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video_path = tmp.name
        content = await file.read()
        tmp.write(content)

    game_id = str(uuid.uuid4())
    report = analyze_video(video_path, jersey_color)

    # Attach metadata from the form
    report["game_title"] = game_title
    report["opponent"] = opponent

    return JSONResponse({
        "game_id": game_id,
        "report": report,
    })
