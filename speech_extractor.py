import os,re
import numpy as np
import librosa
from audio_processor import AudioProcessor
try:
    from faster_whisper import WhisperModel
except ImportError: WhisperModel=None

class SpeakingSkillsAnalyzer:
    def __init__(self,whisper_size="tiny"): self.model_name=whisper_size or os.getenv("WHISPER_MODEL","tiny");self._model=None
    def _get_model(self):
        if WhisperModel is None:return None
        if self._model is None:
            self._model=WhisperModel(os.getenv("WHISPER_MODEL_PATH") or self.model_name,device=os.getenv("WHISPER_DEVICE","cpu"),compute_type=os.getenv("WHISPER_COMPUTE_TYPE","int8"))
        return self._model
    @staticmethod
    def _pitch(y,sr):
        if len(y)<sr:return 0.0
        try:
            f0,_,_=librosa.pyin(y,fmin=70,fmax=400,sr=sr);v=f0[np.isfinite(f0)]
            return float(np.std(v)) if len(v) else 0.0
        except Exception:return 0.0
    def extract_audio_features(self,path):
        p=AudioProcessor(16000);y=p.extract(path);sr=p.target_sample_rate;duration=max(len(y)/sr,.01);text="";prob=0
        model=self._get_model()
        if model is not None:
            try:
                segs,_=model.transcribe(str(path),beam_size=3,vad_filter=True);parts=[];scores=[]
                for s in segs:
                    if s.text.strip():parts.append(s.text.strip())
                    scores.append(float(getattr(s,"avg_logprob",-1)))
                text=" ".join(parts)
                if scores:prob=float(np.clip(np.mean(scores)+1,0,1))
            except Exception:pass
        words=re.findall(r"\b[\w']+\b",text);intervals=librosa.effects.split(y,top_db=30);speech=sum(e-s for s,e in intervals)/sr if len(intervals) else 0
        rms=librosa.feature.rms(y=y)[0];energy=float(np.std(rms)/(np.mean(rms)+1e-6)) if len(rms) else 0
        fillers=len(re.findall(r"\b(um+|uh+|erm+|like|you know|basically)\b",text.lower()))
        return {"transcript":text,"duration_sec":round(duration,2),"word_count":len(words),"wpm":round(len(words)/(duration/60),1) if words else 0,"filler_count":fillers,"pitch_variation":round(self._pitch(y,sr),2),"pause_seconds":round(max(0,duration-speech),2),"speech_ratio":round(speech/duration,3),"energy_variation":round(energy,3),"avg_word_prob":round(prob,3),"transcription_engine":"faster-whisper (local)" if model is not None else "Not installed"}
