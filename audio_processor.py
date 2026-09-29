"""Offline audio extraction, normalization, and lightweight VAD."""

import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

try:
    import av
except ImportError:
    av = None

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None


class AudioProcessor:
    def __init__(self, target_sample_rate=16000, top_db=30):
        self.target_sample_rate = int(target_sample_rate)
        self.top_db = int(top_db)

    def extract(self, media_path):
        waveform = None
        if AudioSegment is not None:
            try:
                segment = AudioSegment.from_file(str(media_path)).set_channels(1).set_frame_rate(self.target_sample_rate)
                if segment.dBFS != float("-inf"):
                    segment = segment.apply_gain(-segment.dBFS)
                samples = np.asarray(segment.get_array_of_samples(), dtype=np.float32)
                scale = float(1 << (8 * segment.sample_width - 1))
                waveform = samples / max(scale, 1.0)
            except (OSError, RuntimeError, ValueError):
                waveform = None
        if waveform is None:
            try:
                waveform, _ = librosa.load(media_path, sr=self.target_sample_rate, mono=True)
            except Exception:
                if av is None:
                    raise
                with av.open(str(media_path)) as container:
                    stream = next((item for item in container.streams if item.type == "audio"), None)
                    if stream is None:
                        raise ValueError("The media file contains no audio stream")
                    chunks = []
                    source_rate = stream.rate or self.target_sample_rate
                    for frame in container.decode(stream):
                        values = frame.to_ndarray()
                        if values.ndim == 2:
                            values = values.mean(axis=0)
                        chunks.append(values.astype(np.float32) / 32768.0)
                    waveform = librosa.resample(np.concatenate(chunks), orig_sr=source_rate, target_sr=self.target_sample_rate)
        sample_rate = self.target_sample_rate
        if waveform.size == 0:
            raise ValueError("The media file contains no decodable audio")
        intervals = librosa.effects.split(waveform, top_db=self.top_db)
        if len(intervals):
            start = int(intervals[0][0])
            end = int(intervals[-1][1])
            trimmed = waveform[start:end]
        else:
            trimmed = waveform
        peak = float(np.max(np.abs(trimmed))) if trimmed.size else 0.0
        if peak > 0:
            trimmed = trimmed / peak * 0.95
        duration = len(trimmed) / self.target_sample_rate
        return trimmed.astype(np.float32), self.target_sample_rate, duration

    def prepare(self, media_path):
        waveform, sample_rate, duration = self.extract(media_path)
        handle, output_path = tempfile.mkstemp(prefix="presentation_audio_", suffix=".wav")
        os.close(handle)
        sf.write(output_path, waveform, sample_rate, subtype="PCM_16")
        return Path(output_path), duration

    @staticmethod
    def cleanup(audio_path):
        try:
            Path(audio_path).unlink(missing_ok=True)
        except OSError:
            pass
