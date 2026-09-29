"""Explainable content and message-quality analysis for spoken videos."""

import re
from collections import Counter


class ContentAnalyzer:
    STRUCTURE_MARKERS = {
        "opening": ("today", "introduction", "first", "begin", "start", "topic"),
        "sequence": ("first", "second", "then", "next", "finally", "because"),
        "evidence": ("example", "evidence", "data", "result", "research", "experience"),
        "closing": ("finally", "in conclusion", "to summarize", "overall", "thank you"),
    }
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
        "i", "in", "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "we",
        "were", "with", "you",
    }

    @classmethod
    def analyze(cls, transcript, script_score=None):
        text = " ".join((transcript or "").split())
        words = re.findall(r"[A-Za-z0-9']+", text.lower())
        sentences = [part.strip() for part in re.split(r"[.!?]+", text) if part.strip()]
        content_words = [word for word in words if word not in cls.STOP_WORDS]
        unique_words = set(content_words)
        word_count = len(words)
        sentence_count = len(sentences)
        lexical_diversity = (len(unique_words) / len(content_words)) * 100 if content_words else 0
        average_sentence_length = word_count / sentence_count if sentence_count else 0
        question_count = len(re.findall(r"\?", text))
        action_count = len(re.findall(r"\b(should|need to|must|will|can|recommend)\b", text.lower()))
        markers = {
            name: any(marker in text.lower() for marker in values)
            for name, values in cls.STRUCTURE_MARKERS.items()
        }
        repeated_terms = [
            {"term": term, "count": count}
            for term, count in Counter(content_words).most_common(8)
            if count >= 3
        ]

        if not words:
            return {
                "available": False,
                "status": "Transcript unavailable",
                "score": None,
                "metrics": {
                    "word_count": 0,
                    "sentence_count": 0,
                    "lexical_diversity": 0,
                    "average_sentence_length": 0,
                    "structure_score": 0,
                    "clarity_score": 0,
                    "vocabulary_score": 0,
                    "evidence_score": 0,
                    "engagement_score": 0,
                    "question_count": 0,
                    "action_language_count": 0,
                    "structure_markers": markers,
                    "repeated_terms": [],
                },
                "strengths": [],
                "improvements": ["Enable or download the local speech model to assess message content and pronunciation."],
            }

        structure_score = sum(markers.values()) / len(markers) * 100
        clarity_score = 100 if 8 <= average_sentence_length <= 28 else max(35, 100 - abs(18 - average_sentence_length) * 3)
        vocabulary_score = min(100, lexical_diversity * 1.35)
        evidence_score = 100 if markers["evidence"] else min(70, 35 + action_count * 8)
        engagement_score = min(100, 35 + question_count * 20 + action_count * 8 + (20 if markers["opening"] else 0) + (15 if markers["closing"] else 0))
        repetition_penalty = min(25, sum(item["count"] - 2 for item in repeated_terms) * 2)
        content_score = round(max(0, min(100, structure_score * 0.25 + clarity_score * 0.25 + vocabulary_score * 0.20 + evidence_score * 0.15 + engagement_score * 0.15 - repetition_penalty)))
        if script_score is not None:
            content_score = round(content_score * 0.75 + float(script_score) * 0.25)

        strengths = []
        improvements = []
        if structure_score >= 60:
            strengths.append("The message shows recognizable organization and progression.")
        if vocabulary_score >= 60:
            strengths.append("The speaker uses a reasonably varied vocabulary.")
        if clarity_score < 70:
            improvements.append("Use shorter, clearer sentences and pause between major ideas.")
        if structure_score < 60:
            improvements.append("Add a clear opening, ordered points, supporting example, and concise conclusion.")
        if repetition_penalty:
            improvements.append("Reduce repeated terms and replace them with precise wording.")
        if script_score is not None and script_score < 70:
            improvements.append("Keep the main message closer to the intended topic and key points.")
        if not strengths:
            strengths.append("The transcript provides a basis for targeted content feedback.")
        if not improvements:
            improvements.append("Content structure and clarity are consistent; continue strengthening evidence and examples.")

        return {
            "available": True,
            "status": "Content analyzed",
            "score": content_score,
            "metrics": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "lexical_diversity": round(lexical_diversity, 1),
                "average_sentence_length": round(average_sentence_length, 1),
                "structure_score": round(structure_score, 1),
                "clarity_score": round(clarity_score, 1),
                "vocabulary_score": round(vocabulary_score, 1),
                "evidence_score": round(evidence_score, 1),
                "engagement_score": round(engagement_score, 1),
                "question_count": question_count,
                "action_language_count": action_count,
                "structure_markers": markers,
                "repeated_terms": repeated_terms,
            },
            "strengths": strengths,
            "improvements": improvements,
        }
