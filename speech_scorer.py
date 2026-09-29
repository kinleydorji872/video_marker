class SpeakingSkillsScorer:
    @staticmethod
    def evaluate(m):
        w=float(m.get("wpm",0) or 0);d=max(float(m.get("duration_sec",1) or 1),1);fr=float(m.get("filler_count",0) or 0)/(d/60);pace=max(0,100-min(abs(w-140),80)*1.1) if w else 50;filler=max(0,100-fr*10);presence=min(100,float(m.get("speech_ratio",0))*100);score=round(pace*.35+filler*.25+presence*.25+min(100,float(m.get("energy_variation",0))*120)*.15)
        return {"speaking_score":score,"status":"Strong" if score>=80 else "Developing" if score>=60 else "Needs practice","feedback":"Aim for a steadier pace and fewer filler words." if score<70 else "Speech timing and vocal activity are consistent."}
