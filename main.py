import os
import shutil
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from communication_grade import WeightedCommunicationScorer
from content_analyzer import ContentAnalyzer
from ollama_grader import OllamaGrader
from scoring_engine import VideoScorer
from speaker_tracker import compare_script, progress_for, record_progress
from presenter_rubric import PresenterRubric
from feedback_store import save_feedback
from feature_extractor import VideoFeatureExtractor
from speech_extractor import SpeakingSkillsAnalyzer
from speech_scorer import SpeakingSkillsScorer
from gaze_detector import GazeDetector
from ml_scoring_engine import MLScoringEngine
from database import Database
from assessment_router import AssessmentRouter


class FeedbackRequest(BaseModel):
    rating: int = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=1000)
    speaker: str = Field(default="Speaker", max_length=120)
    filename: str = Field(default="", max_length=255)
    score: int | None = Field(default=None, ge=0, le=100)

app = FastAPI(
    title="Offline Video Grading System",
    description="Professional video assessment powered by local computer vision, Whisper, and optional local Ollama coaching without external cloud APIs.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

video_scorer = VideoScorer()
ollama = OllamaGrader()
gaze_detector = GazeDetector()
video_features = VideoFeatureExtractor(gaze_detector=gaze_detector)
speech_analyzer = SpeakingSkillsAnalyzer(whisper_size="tiny")
ml_scorer = MLScoringEngine()
database = Database()
assessment_router = AssessmentRouter()
job_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("GRADING_WORKERS", "1"))))
job_lock = threading.Lock()
jobs = {}
UPLOAD_DIR = "temp_speech_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
FRONTEND_DIR = Path(__file__).parent / "frontend"
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(FRONTEND_DIR / "index.html")


def _save_upload(upload):
    suffix = os.path.splitext(upload.filename or "video.mp4")[1]
    handle, path = tempfile.mkstemp(suffix=suffix, dir=UPLOAD_DIR)
    os.close(handle)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return path


def _grade_video(video_path, filename, speaker_name="Default speaker"):
    try:
        visual_metrics = video_features.extract_features(video_path)
        visual_evaluation = video_scorer.evaluate(visual_metrics)
    except Exception as error:
        visual_metrics = {
            "duration_sec": 0,
            "resolution": "0x0",
            "avg_sharpness": 0,
            "avg_brightness": 0,
            "avg_motion": 0,
            "reading_behavior": {
                "status": "Insufficient camera evidence",
                "likelihood": 0,
                "confidence": 0,
                "face_visibility": 0,
                "eye_line_shift": 0,
                "downward_gaze_ratio": 0,
                "note": "This is a visual proxy, not proof of script use or presenter intent.",
            },
        }
        visual_evaluation = {"overall_score": 0, "overall_status": "Video analysis unavailable", "classifications": {}, "improvements_needed": str(error)}
    try:
        speech_metrics = speech_analyzer.extract_audio_features(video_path)
        speech_evaluation = SpeakingSkillsScorer.evaluate(speech_metrics)
        presentation = {
            "metrics": speech_metrics,
            "score": speech_evaluation["speaking_score"],
            "status": speech_evaluation["status"],
            "feedback": speech_evaluation["feedback"],
        }
    except Exception as error:
        speech_metrics = {
            "transcript": "",
            "duration_sec": 1,
            "word_count": 0,
            "wpm": 0,
            "filler_count": 0,
            "pitch_variation": 0,
            "pause_seconds": 0,
            "speech_ratio": 0,
            "energy_variation": 0,
            "avg_word_prob": 0,
        }
        speech_evaluation = {}
        presentation = {
            "metrics": speech_metrics,
            "score": 0,
            "status": "Audio unavailable",
            "feedback": f"Presentation analysis was skipped: {error}",
        }

    content_analysis = ContentAnalyzer.analyze(speech_metrics.get("transcript", ""))
    module_selection = assessment_router.select(
        visual_metrics,
        speech_metrics,
        content_analysis,
        visual_metrics.get("reading_behavior", {}),
        ml_scorer.available and ml_scorer.rank_eligible,
        ml_scorer.rank_eligible,
    )
    ai_review = ollama.review(filename, visual_metrics, visual_evaluation, speech_metrics, speech_evaluation)
    ml_score = ml_scorer.predict_score(speech_metrics, visual_metrics.get("reading_behavior", {})) if module_selection["learned_available"] else None
    rank_ml_score = ml_score if module_selection["learned_rank_eligible"] else None
    communication_assessment = WeightedCommunicationScorer.evaluate(
        visual_metrics,
        speech_metrics,
        content_analysis=content_analysis,
        gaze_metrics=visual_metrics.get("reading_behavior", {}),
        ml_score=rank_ml_score,
        active_modules=module_selection["active"],
    )
    rubric = PresenterRubric.build(communication_assessment["breakdown"])
    overall_score = communication_assessment["overall_score"]
    overall_status = (
        "Pass / Good" if overall_score >= 80 else "Improve" if overall_score >= 50 else "Need to Improve"
    )

    return {
        "filename": filename,
        "overview": {
            "filename": filename,
            "overall_score": overall_score,
            "grade": communication_assessment["grade"],
            "status": overall_status,
            "grading_mode": "Weighted communication scoring from video + audio features",
            "module_selection": module_selection,
            "weights": communication_assessment["weights"],
            "rubric": rubric,
        },
        "visual_quality": {
            "metrics": visual_metrics,
            "classifications": visual_evaluation["classifications"],
            "feedback": visual_evaluation["improvements_needed"],
            "score": visual_evaluation["overall_score"],
        },
        "reading_assessment": visual_metrics.get("reading_behavior", {}),
        "presentation": presentation,
        "ml_score": ml_score,
        "ml_rank_eligible": module_selection["learned_rank_eligible"],
        "content_analysis": content_analysis,
        "communication_assessment": communication_assessment,
        "ai_coaching": ai_review,
    }


@app.get("/health")
def health_check():
    return {
        "status": "ready",
        "headings": ["Overview", "Content Analysis", "Presentation", "Visual Quality", "AI Coaching"],
        "ollama": {"available": ollama.is_available(), "model": ollama.model},
    }


def _grade_uploaded_files(files, script, speaker):
    reports = []
    for temp_path, filename in files:
        try:
            report = _grade_video(temp_path, filename, speaker)
            script_adherence = compare_script(report["presentation"]["metrics"].get("transcript", ""), script) if script.strip() else {"provided": False, "score": None, "status": "No script provided", "matched_words": 0, "script_words": 0, "missing_keywords": []}
            report["script_adherence"] = script_adherence
            report["reading_assessment"]["reference_script_provided"] = bool(script.strip())
            report["reading_assessment"]["script_alignment"] = script_adherence
            report["content_analysis"] = ContentAnalyzer.analyze(report["presentation"]["metrics"].get("transcript", ""), script_adherence.get("score"))
            module_selection = assessment_router.select(
                report["visual_quality"]["metrics"],
                report["presentation"]["metrics"],
                report["content_analysis"],
                report.get("reading_assessment", {}),
                ml_scorer.available and ml_scorer.rank_eligible,
                ml_scorer.rank_eligible,
            )

            communication_assessment = WeightedCommunicationScorer.evaluate(
                report["visual_quality"]["metrics"],
                report["presentation"]["metrics"],
                script_adherence.get("score"),
                content_analysis=report["content_analysis"],
                gaze_metrics=report.get("reading_assessment", {}),
                ml_score=report.get("ml_score") if module_selection["learned_rank_eligible"] else None,
                active_modules=module_selection["active"],
            )
            report["communication_assessment"] = communication_assessment
            report["overview"]["rubric"] = PresenterRubric.build(communication_assessment["breakdown"])
            report["overview"]["overall_score"] = communication_assessment["overall_score"]
            report["overview"]["grade"] = communication_assessment["grade"]
            report["overview"]["status"] = "Pass / Good" if communication_assessment["overall_score"] >= 80 else "Improve" if communication_assessment["overall_score"] >= 50 else "Need to Improve"
            report["overview"]["module_selection"] = module_selection
            report["overall_score"] = communication_assessment["overall_score"]
            report["speaking_score"] = report["presentation"].get("score")
            report["script_adherence_score"] = script_adherence.get("score")
            report["progress_entry"] = record_progress(speaker, report)
            report["database_entry"] = database.save_attempt(speaker, filename, report)
            reports.append(report)
        except Exception as error:
            reports.append({"filename": filename, "error": str(error)})
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return {"status": "success", "speaker": speaker, "results": reports}


def _run_grading_job(job_id, files, script, speaker):
    with job_lock:
        jobs[job_id]["status"] = "running"
    try:
        result = _grade_uploaded_files(files, script, speaker)
        with job_lock:
            jobs[job_id].update(status="completed", result=result)
            if len(jobs) > 100:
                oldest = next(iter(jobs))
                if oldest != job_id:
                    jobs.pop(oldest, None)
    except Exception as error:
        with job_lock:
            jobs[job_id].update(status="failed", error=str(error))


def _enqueue_grading(files, script, speaker):
    job_id = uuid.uuid4().hex
    with job_lock:
        jobs[job_id] = {"job_id": job_id, "status": "queued"}
    job_executor.submit(_run_grading_job, job_id, files, script, speaker)
    return {"job_id": job_id, "status": "queued", "status_url": f"/jobs/{job_id}"}


@app.post("/grade-videos")
def grade_videos(
    files: List[UploadFile] = File(...),
    script: str = Form(default=""),
    speaker: str = Form(default="Default speaker"),
):
    saved_files = [(_save_upload(file), file.filename or "video") for file in files]
    return _enqueue_grading(saved_files, script, speaker)


@app.post("/grade-video")
def grade_video(
    file: UploadFile = File(...),
    script: str = Form(default=""),
    speaker: str = Form(default="Default speaker"),
):
    saved_file = (_save_upload(file), file.filename or "video")
    return _enqueue_grading([saved_file], script, speaker)


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    with job_lock:
        job = jobs.get(job_id)
        if job is None:
            return {"job_id": job_id, "status": "not_found"}
        return dict(job)


@app.get("/progress")
def get_progress(speaker: str = ""):
    return {"speaker": speaker or "All speakers", "results": progress_for(speaker)}


@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    entry = save_feedback(feedback.model_dump())
    return {"status": "saved", "message": "Thank you. Your feedback helps improve the assessment experience.", "feedback": entry}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)