"""Select grading modules from evidence actually extracted from one upload."""


class AssessmentRouter:
    @staticmethod
    def select(visual_metrics, speech_metrics, content_analysis, gaze_metrics, ml_available, ml_rank_eligible=False):
        visual_available = (
            float(visual_metrics.get("duration_sec", 0) or 0) > 0
            and visual_metrics.get("resolution", "0x0") != "0x0"
        )
        audio_available = (
            float(speech_metrics.get("duration_sec", 0) or 0) > 0
            and (
                float(speech_metrics.get("speech_ratio", 0) or 0) > 0
                or int(speech_metrics.get("word_count", 0) or 0) > 0
                or float(speech_metrics.get("pitch_variation", 0) or 0) > 0
            )
        )
        content_available = bool(
            content_analysis
            and content_analysis.get("available")
            and content_analysis.get("score") is not None
        )
        gaze_available = bool(
            gaze_metrics
            and gaze_metrics.get("available")
            and int(gaze_metrics.get("samples", 0) or 0) >= 3
        )
        ml_input_available = (
            audio_available
            and ml_available
            and (
                int(speech_metrics.get("word_count", 0) or 0) > 0
                or float(speech_metrics.get("avg_word_prob", 0) or 0) > 0
            )
        )
        modules = {
            "visual": visual_available,
            "audio": audio_available,
            "content": content_available,
            "gaze": gaze_available,
            "learned": ml_input_available and ml_rank_eligible,
        }
        reasons = {
            "visual": "Video image metrics available" if visual_available else "Video could not provide usable image metrics",
            "audio": "Audio activity detected" if audio_available else "No usable speech or audio evidence",
            "content": "Transcript available" if content_available else "No reliable transcript was produced",
            "gaze": "Face landmarks available" if gaze_available else "Not enough face landmarks were detected",
            "learned": "Production-calibrated ML model contributes to rank" if ml_input_available and ml_rank_eligible else "ML prediction is advisory only; real labeled calibration is required for ranking",
        }
        return {
            "modules": modules,
            "active": [name for name, enabled in modules.items() if enabled],
            "excluded": [name for name, enabled in modules.items() if not enabled],
            "reasons": reasons,
            "learned_available": ml_input_available,
            "learned_rank_eligible": bool(ml_input_available and ml_rank_eligible),
            "note": "Only modules with usable evidence contribute to the overall score.",
        }
