class WeightedCommunicationScorer:
    """Transparent weighted speaker communication scoring model.

    This project does not use pretrained model weights. Instead it converts the
    extracted video/audio features into explainable weighted scores for visual
    quality, speaking flow, pronunciation proxy, and presentation clarity.
    Each assessment is independent and uses only the supplied presentation features.
    """

    @staticmethod
    def _clamp(value, minimum=0, maximum=100):
        return max(minimum, min(maximum, value))

    @staticmethod
    def _sharpness_score(sharpness):
        if sharpness <= 0:
            return 20
        if sharpness < 100:
            return 30 + (sharpness / 100.0) * 30
        if sharpness < 300:
            return 60 + ((sharpness - 100) / 200.0) * 35
        return 100

    @staticmethod
    def _brightness_score(brightness):
        if brightness < 30:
            return max(0, 100 - (30 - brightness) * 2.2)
        if brightness <= 200:
            return 100
        return max(0, 100 - (brightness - 200) * 0.7)

    @staticmethod
    def _stability_score(motion):
        if motion <= 8:
            return 100
        if motion <= 20:
            return max(0, 100 - (motion - 8) * 4)
        return max(0, 52 - (motion - 20) * 2.5)

    @staticmethod
    def _pace_score(wpm):
        if 120 <= wpm <= 160:
            return 100
        if wpm <= 0:
            return 55
        diff = min(abs(wpm - 140), 80)
        return max(0, 100 - diff * 1.1)

    @staticmethod
    def _filler_score(fillers_per_minute):
        return max(0, 100 - fillers_per_minute * 10)

    @staticmethod
    def _pitch_score(pitch_variation):
        if pitch_variation <= 0:
            return 20
        return WeightedCommunicationScorer._clamp((pitch_variation / 45.0) * 100)

    @staticmethod
    def _pause_score(pause_seconds, duration_sec):
        if duration_sec <= 0:
            return 50
        pause_ratio = pause_seconds / duration_sec
        return max(0, 100 - pause_ratio * 200)

    @staticmethod
    def _speech_presence_score(speech_metrics):
        ratio = float(speech_metrics.get("speech_ratio", 0) or 0)
        energy_variation = float(speech_metrics.get("energy_variation", 0) or 0)
        activity_score = ratio * 100
        energy_score = min(100, energy_variation * 55)
        return WeightedCommunicationScorer._clamp(activity_score * 0.75 + energy_score * 0.25)

    @staticmethod
    def _pronunciation_proxy(speech_metrics):
        wpm = float(speech_metrics.get("wpm", 0) or 0)
        fillers = int(speech_metrics.get("filler_count", 0) or 0)
        pauses = float(speech_metrics.get("pause_seconds", 0) or 0)
        duration = float(speech_metrics.get("duration_sec", 1) or 1)
        pitch = float(speech_metrics.get("pitch_variation", 0) or 0)

        duration_minutes = max(duration / 60.0, 1 / 60.0)
        filler_penalty = (fillers / duration_minutes) * 7
        pause_penalty = (pauses / duration_minutes) * (50 / 60.0)
        pace_penalty = max(0, abs(wpm - 140) * 0.35)
        tone_bonus = min(15, pitch * 0.2)
        score = 100 - filler_penalty - pause_penalty - pace_penalty + tone_bonus
        return WeightedCommunicationScorer._clamp(score)

    @staticmethod
    def _grade_from_score(score):
        if score >= 90:
            return "Excellent"
        if score >= 80:
            return "Strong"
        if score >= 70:
            return "Good"
        if score >= 60:
            return "Fair"
        return "Need to Improve"

    @classmethod
    def evaluate(
        cls,
        visual_metrics,
        speech_metrics,
        script_score=None,
        historical_scores=None,
        content_analysis=None,
        gaze_metrics=None,
        ml_score=None,
        active_modules=None,
    ):
        visual_quality = (
            cls._sharpness_score(float(visual_metrics.get("avg_sharpness", 0))) * 0.40
            + cls._brightness_score(float(visual_metrics.get("avg_brightness", 0))) * 0.30
            + cls._stability_score(float(visual_metrics.get("avg_motion", 0))) * 0.30
        )

        duration_minutes = max(float(speech_metrics.get("duration_sec", 1) or 1) / 60.0, 1 / 60.0)
        fillers_per_minute = float(speech_metrics.get("filler_count", 0) or 0) / duration_minutes
        pauses_per_minute = float(speech_metrics.get("pause_seconds", 0) or 0) / duration_minutes
        speech_presence_score = cls._speech_presence_score(speech_metrics)
        fluency_score = (
            cls._pace_score(float(speech_metrics.get("wpm", 0))) * 0.25
            + cls._filler_score(fillers_per_minute) * 0.20
            + cls._pause_score(
                pauses_per_minute,
                60,
            ) * 0.20
            + cls._pitch_score(float(speech_metrics.get("pitch_variation", 0))) * 0.10
            + speech_presence_score * 0.25
        )

        pronunciation_score = cls._pronunciation_proxy(speech_metrics)
        vocal_expression_score = cls._pitch_score(float(speech_metrics.get("pitch_variation", 0)))
        confidence_score = cls._clamp(
            cls._pace_score(float(speech_metrics.get("wpm", 0))) * 0.25
            + cls._filler_score(fillers_per_minute) * 0.20
            + cls._pause_score(pauses_per_minute, 60) * 0.15
            + vocal_expression_score * 0.15
            + speech_presence_score * 0.25
        )

        content_component = 45.0
        content_available = False
        if content_analysis and content_analysis.get("score") is not None:
            content_component = float(content_analysis["score"])
            content_available = bool(content_analysis.get("available", True))
        elif script_score is not None:
            content_component = float(script_score)
            content_available = True

        content_metrics = (content_analysis or {}).get("metrics", {})
        message_structure = float(content_metrics.get("structure_score", content_component) or content_component)
        message_clarity = float(content_metrics.get("clarity_score", content_component) or content_component)
        evidence_score = float(content_metrics.get("evidence_score", content_component) or content_component)
        audience_engagement = cls._clamp(
            float(content_metrics.get("engagement_score", content_component) or content_component) * 0.65
            + vocal_expression_score * 0.35
        )

        active = set(
            ("visual", "audio", "content", "gaze", "learned")
            if active_modules is None
            else active_modules
        )
        overall_metrics = {}
        overall_weights = {}

        def add_metric(name, score, weight):
            overall_metrics[name] = cls._clamp(score)
            overall_weights[name] = weight

        if "visual" in active:
            add_metric("visual_quality", visual_quality, 0.10)
        if "content" in active:
            add_metric("message_structure", message_structure, 0.10)
            add_metric("message_clarity", message_clarity, 0.10)
            add_metric("evidence", evidence_score, 0.08)
            add_metric("audience_engagement", audience_engagement, 0.10)
        if "audio" in active:
            add_metric("fluency", fluency_score, 0.16)
            add_metric("pronunciation", pronunciation_score, 0.12)
            add_metric("confidence", confidence_score, 0.16)
            add_metric("vocal_expression", vocal_expression_score, 0.09)
            if "speech_ratio" in speech_metrics or "energy_variation" in speech_metrics:
                add_metric("speech_presence", speech_presence_score, 0.08)
        if script_score is not None and "content" in active:
            add_metric("script_adherence", script_score, 0.05)
        if ml_score is not None and "learned" in active:
            overall_metrics["learned_score"] = cls._clamp(float(ml_score))
            overall_weights["learned_score"] = 0.05
        if gaze_metrics and gaze_metrics.get("available") and "gaze" in active:
            overall_metrics["gaze_quality"] = cls._clamp(100 - float(gaze_metrics.get("reading_confidence_score", 0) or 0))
            overall_weights["gaze_quality"] = 0.05
        weight_total = sum(overall_weights.values())
        overall_score = round(
            sum(overall_metrics[name] * weight for name, weight in overall_weights.items()) / weight_total
        ) if weight_total else 0
        raw_score = overall_score

        recommendations = []
        if "visual" in active and visual_quality < 70:
            recommendations.append("Improve lighting and camera stability for a more professional presentation; this has limited impact on the communication grade.")
        if "audio" in active and fluency_score < 70:
            recommendations.append("Reduce filler words and long pauses to make the speech smoother and more persuasive.")
        if "audio" in active and pronunciation_score < 70:
            recommendations.append("Work on clearer articulation and more consistent pacing to strengthen pronunciation and speech clarity.")
        if "audio" in active and vocal_expression_score < 70:
            recommendations.append("Add more vocal emphasis and natural pitch variation to sound more engaging and confident.")
        if "content" in active and message_structure < 70:
            recommendations.append("Use a clear opening, ordered points, supporting evidence, and a concise conclusion.")
        if "content" in active and message_clarity < 70:
            recommendations.append("Make sentences shorter and state one main idea at a time.")
        if "content" in active and evidence_score < 70:
            recommendations.append("Support important claims with a concrete example, result, data point, or personal experience.")
        if "content" in active and audience_engagement < 70:
            recommendations.append("Create more audience connection with a question, direct benefit, or clear call to action.")
        if "audio" in active and confidence_score < 70:
            recommendations.append("Rehearse the opening and key transitions to reduce hesitation and sound more assured.")
        if "audio" in active and speech_presence_score < 60:
            recommendations.append("The audio signal contains limited consistent speech; check the microphone and make sure the speaker is audible.")
        if "content" in active and script_score is not None and script_score < 70:
            recommendations.append("Keep key ideas more direct and structured so the message stays aligned with the intended content.")
        if "content" not in active:
            recommendations.append("Content scoring was skipped because no reliable transcript evidence was available.")

        if not recommendations:
            recommendations.append("Excellent communication profile. Keep the current style and continue presenting with consistency.")

        return {
            "overall_score": overall_score,
            "raw_score": raw_score,
            "grade": cls._grade_from_score(overall_score),
            "overall_metrics": {
                "scores": {name: round(value, 2) for name, value in overall_metrics.items()},
                "weights": overall_weights,
                "weight_total": round(weight_total, 2),
                "active_modules": sorted(active),
            },
            "weights": overall_weights,
            "breakdown": {
                "visual_quality": round(cls._clamp(visual_quality), 2),
                "message_structure": round(cls._clamp(message_structure), 2),
                "message_clarity": round(cls._clamp(message_clarity), 2),
                "evidence": round(cls._clamp(evidence_score), 2),
                "fluency": round(cls._clamp(fluency_score), 2),
                "pronunciation": round(cls._clamp(pronunciation_score), 2),
                "audience_engagement": round(cls._clamp(audience_engagement), 2),
                "confidence": round(cls._clamp(confidence_score), 2),
                "speech_presence": round(cls._clamp(speech_presence_score), 2),
                "vocal_expression": round(cls._clamp(vocal_expression_score), 2),
                "content_quality": round(cls._clamp(content_component), 2),
                "script_adherence": round(cls._clamp(float(script_score)), 2) if script_score is not None else None,
            },
            "recommendations": recommendations,
            "historical_calibration": [],
            "rates_per_minute": {
                "fillers": round(fillers_per_minute, 2),
                "pauses": round(pauses_per_minute, 2),
            },
        }
