import os, shutil, tempfile, threading, uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from communication_grade import WeightedCommunicationScorer
from content_analyzer import ContentAnalyzer
from feature_extractor import VideoFeatureExtractor
from speech_extractor import SpeakingSkillsAnalyzer
from speech_scorer import SpeakingSkillsScorer
from presenter_rubric import PresenterRubric
from feedback_store import save_feedback
from database import Database
from assessment_router import AssessmentRouter
from speaker_tracker import compare_script, progress_for, record_progress
from gaze_detector import GazeDetector
from ml_scoring_engine import MLScoringEngine

app=FastAPI(title="Presenter Performance Studio",version="2.0.0",description="Local-first open-source video presentation assessment.")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"])
ROOT=Path(__file__).parent
FRONTEND=ROOT/"frontend"; UPLOAD_DIR=ROOT/"temp_speech_files"; UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/assets",StaticFiles(directory=FRONTEND/"assets"),name="assets")
executor=ThreadPoolExecutor(max_workers=max(1,int(os.getenv("GRADING_WORKERS","1")))); jobs={}; lock=threading.Lock()
video_features=VideoFeatureExtractor(gaze_detector=GazeDetector()); speech=SpeakingSkillsAnalyzer(); database=Database(); router=AssessmentRouter(); ml=MLScoringEngine()

class FeedbackRequest(BaseModel):
    rating:int=Field(ge=1,le=5); comment:str=Field(default="",max_length=1000); speaker:str="Speaker"; filename:str=""; score:int|None=Field(default=None,ge=0,le=100)

@app.get("/",include_in_schema=False)
def home(): return FileResponse(FRONTEND/"index.html")

def save_upload(upload):
    suffix=Path(upload.filename or "video.mp4").suffix or ".mp4"; fd,path=tempfile.mkstemp(suffix=suffix,dir=UPLOAD_DIR); os.close(fd)
    with open(path,"wb") as out: shutil.copyfileobj(upload.file,out)
    return path

def grade(path,filename,speaker_name,script):
    visual=video_features.extract_features(path)
    visual_eval={"overall_score":0,"classifications":{},"improvements_needed":""}
    try:
        sharp=visual["avg_sharpness"]; bright=visual["avg_brightness"]; motion=visual["avg_motion"]
        visual_score=min(100,40+min(sharp/3,40)+min(max(bright,0)/2,50)-min(max(motion-8,0)*2,30))
        visual_eval={"overall_score":round(max(0,visual_score)),"classifications":{"sharpness":round(sharp,1),"brightness":round(bright,1),"motion":round(motion,2)},"improvements_needed":"Improve lighting, framing or camera stability where needed."}
    except Exception: pass
    speech_metrics=speech.extract_audio_features(path)
    speech_eval=SpeakingSkillsScorer.evaluate(speech_metrics)
    transcript=speech_metrics.get("transcript","")
    script_adherence=compare_script(transcript,script) if script.strip() else {"provided":False,"score":None,"status":"No script provided","matched_words":0,"script_words":0,"missing_keywords":[]}
    content=ContentAnalyzer.analyze(transcript,script_adherence.get("score"))
    module=router.select(visual,speech_metrics,content,visual.get("reading_behavior",{}),False,False)
    comm=WeightedCommunicationScorer.evaluate(visual,speech_metrics,script_adherence.get("score"),content_analysis=content,gaze_metrics=visual.get("reading_behavior",{}),active_modules=module["active"])
    report={"filename":filename,"overview":{"filename":filename,"overall_score":comm["overall_score"],"grade":comm["grade"],"status":"Pass / Good" if comm["overall_score"]>=80 else "Improve" if comm["overall_score"]>=50 else "Need to Improve","grading_mode":"Explainable local scoring","module_selection":module,"weights":comm["weights"],"rubric":PresenterRubric.build(comm["breakdown"])},
    report[0]["visual_quality"]={"metrics":visual,"classifications":visual_eval["classifications"],"feedback":visual_eval["improvements_needed"],"score":visual_eval["overall_score"]}
    report[0]["reading_assessment"]=visual.get("reading_behavior",{})
    report[0]["presentation"]={"metrics":speech_metrics,"score":speech_eval["speaking_score"],"status":speech_eval["status"],"feedback":speech_eval["feedback"]}
    report[0]["content_analysis"]=content; report[0]["script_adherence"]=script_adherence; report[0]["communication_assessment"]=comm
    report[0]["ml_score"]=None; report[0]["ml_rank_eligible"]=False; report[0]["ai_coaching"]={"available":False,"mode":"Rule-based local coaching","message":"Coaching is generated from measurable local signals. No cloud AI is used."}
    report[0]["overall_score"]=comm["overall_score"]; report[0]["progress_entry"]=record_progress(speaker_name,report[0]); report[0]["database_entry"]=database.save_attempt(speaker_name,filename,report[0])
    return report[0]

def worker(job_id,files,script,speaker):
    with lock: jobs[job_id]["status"]="running"
    results=[]
    try:
        for path,name in files:
            try: results.append(grade(path,name,speaker,script))
            except Exception as e: results.append({"filename":name,"error":str(e)})
            finally:
                if os.path.exists(path): os.remove(path)
        with lock: jobs[job_id].update(status="completed",result={"status":"success","speaker":speaker,"results":results})
    except Exception as e:
        with lock: jobs[job_id].update(status="failed",error=str(e))

def enqueue(files,script,speaker):
    job_id=uuid.uuid4().hex
    with lock: jobs[job_id]={"job_id":job_id,"status":"queued"}
    executor.submit(worker,job_id,files,script,speaker)
    return {"job_id":job_id,"status":"queued","status_url":"/jobs/"+job_id}

@app.post("/grade-video")
def grade_video(file:UploadFile=File(...),script:str=Form(""),speaker:str=Form("Presenter")):
    return enqueue([(save_upload(file),file.filename or "video")],script,speaker)

@app.post("/grade-videos")
def grade_videos(files:List[UploadFile]=File(...),script:str=Form(""),speaker:str=Form("Presenter")):
    return enqueue([(save_upload(f),f.filename or "video") for f in files],script,speaker)

@app.get("/jobs/{job_id}")
def get_job(job_id:str):
    with lock:return dict(jobs.get(job_id,{"job_id":job_id,"status":"not_found"}))

@app.get("/progress")
def get_progress(speaker:str=""): return {"speaker":speaker or "All speakers","results":progress_for(speaker)}

@app.post("/feedback")
def feedback(req:FeedbackRequest): return {"status":"saved","feedback":save_feedback(req.model_dump())}

@app.get("/health")
def health(): return {"status":"ready","local_only":True,"transcription_engine":"faster-whisper" if speech._get_model() is not None else "unavailable until installed/model cached","version":"2.0.0"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=int(os.getenv("PORT","8000")))
