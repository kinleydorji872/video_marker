import json
from datetime import datetime,timezone
from pathlib import Path
PATH=Path("speaker_progress.json")
def record_progress(speaker,report):
    data=[]
    if PATH.exists():
        try:data=json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:pass
    e={"speaker":speaker,"filename":report.get("filename",""),"score":report.get("overall_score",report.get("overview",{}).get("overall_score",0)),"created_at":datetime.now(timezone.utc).isoformat(timespec="seconds")};data.append(e);PATH.write_text(json.dumps(data[-1000:],indent=2),encoding="utf-8");return e
def progress_for(speaker=""):
    if not PATH.exists():return []
    try:data=json.loads(PATH.read_text(encoding="utf-8"))
    except Exception:return []
    return [x for x in data if not speaker or x.get("speaker")==speaker]
