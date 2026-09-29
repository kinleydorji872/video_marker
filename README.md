# Presenter Performance Studio

A local-first, open-source video presentation assessment system.

## Features
- Responsive browser upload dashboard
- Local video analysis with OpenCV
- Local audio analysis with librosa
- Optional local speech transcription with faster-whisper
- Explainable content and presentation scoring
- Optional local gaze analysis with MediaPipe assets
- SQLite history and presenter progress
- No OpenAI, Gemini, Azure, AWS or paid grading API

## Run
Install Python 3.10+, FFmpeg, then:

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open http://127.0.0.1:8000.

For fully offline transcription, keep a faster-whisper model locally and set `WHISPER_MODEL_PATH` to that directory. Without a local model, video/audio analysis still works and transcript-based content analysis is marked unavailable.

## Architecture
Browser -> FastAPI -> local OpenCV/librosa/Whisper features -> deterministic scoring -> SQLite.
