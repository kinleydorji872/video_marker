# Presenter Performance Studio

## Product approach

The system follows a presenter-assessment rubric inspired by established presentation coaching dimensions:

- Message structure: opening, progression, and conclusion.
- Message clarity: sentence length and understandable ideas.
- Evidence and examples: data, research, results, recommendations, and experience markers.
- Delivery and fluency: pace, filler control, pause rhythm, and continuity.
- Articulation: explainable pronunciation proxy from speech timing and clarity signals.
- Audience engagement: questions, relevance language, and calls to action.
- Presenter confidence: pace control, filler control, pause control, and vocal steadiness.
- Technical presentation: lighting, stability, sharpness, and framing. This is intentionally only 5% of the grade.

The numeric grade is deterministic and explainable. Ollama is optional and only adds coaching language.

## Layout

```text
backend/
  app.py                 Production FastAPI entry point
frontend/
  index.html             Presentation layer markup
  assets/styles.css      Visual system and responsive layout
  assets/app.js          Upload workflow and report rendering
main.py                  Backward-compatible application entry point
content_analyzer.py      Message and transcript analysis
presenter_rubric.py      Presenter dimension definitions
communication_grade.py   Weighted scoring engine
feature_extractor.py     Low-weight technical video signals
speech_extractor.py      Audio, timing, pitch, and transcription signals
speaker_tracker.py       Local progress history
gaze_detector.py       Local MediaPipe 3D iris and downward-gaze features
ml_scoring_engine.py   Local RandomForest presentation score with fallback
train_model.py         Offline CSV model training utility
database.py            Thread-safe SQLite attempt and coaching persistence

The learned scorer is optional at runtime. Place a trained model at
`models/presentation_scorer.joblib`; otherwise the deterministic local fallback
is used. MediaPipe requires a local `face_landmarker.task` asset configured with
`MEDIAPIPE_FACE_MODEL` or placed at `models/face_landmarker.task`.
```

## Run

From the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8001
```

The legacy command `python main.py` remains supported. Open `http://127.0.0.1:8001` for the frontend.
