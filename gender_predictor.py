import librosa
import numpy as np
import logging

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioAnalyzerConfig:
    def __init__(self):
        self.pitch_fmin_hz = librosa.note_to_hz('C3')  # 130.81 Hz
        self.pitch_fmax_hz = librosa.note_to_hz('C6')  # 1046.50 Hz
        self.gender_female_pitch_threshold = 165       # Threshold pitch to classify female

def extract_avg_pitch(file_path: str, config: AudioAnalyzerConfig) -> float:
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            return 0.0
    except:
        return 0.0

    try:
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.pitch_fmin_hz,
            fmax=config.pitch_fmax_hz,
            sr=sr
        )
        valid_pitch = pitch[~np.isnan(pitch) & voiced_flag]
        if len(valid_pitch) == 0:
            return 0.0
        return np.mean(valid_pitch)
    except:
        return 0.0

def predict_gender(file_path: str) -> str:
    config = AudioAnalyzerConfig()
    avg_pitch = extract_avg_pitch(file_path, config)

    if avg_pitch == 0.0:
        return "Unable to determine: No pitch detected"
    elif avg_pitch > config.gender_female_pitch_threshold:
        return "Female"
    else:
        return "Male"
