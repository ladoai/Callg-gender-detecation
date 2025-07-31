import librosa
import numpy as np
import soundfile as sf # soundfile का उपयोग बेहतर I/O के लिए
import logging

# लॉगिंग कॉन्फ़िगरेशन
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AudioAnalyzerConfig:
    """ऑडियो विश्लेषण के लिए कॉन्फ़िगरेशन पैरामीटर्स."""
    def __init__(self):
        self.pitch_fmin_hz = librosa.note_to_hz('C3')  # 130.81 Hz
        self.pitch_fmax_hz = librosa.note_to_hz('C6')  # 1046.50 Hz
        self.frame_length_ms = 20  # SNR गणना के लिए फ्रेम की लंबाई (मिलीसेकंड)
        self.hop_length_ms = 10    # SNR गणना के लिए हॉप की लंबाई (मिलीसेकंड)
        self.noise_estimation_duration_s = 0.5 # शोर अनुमान के लिए प्रारंभिक ऑडियो अवधि (सेकंड)
        self.signal_energy_percentile = 90     # सिग्नल ऊर्जा के लिए RMS ऊर्जा का परसेंटाइल
        self.min_snr_threshold_db = 7.0        # न्यूनतम SNR (dB में) स्पष्टता के लिए
        self.min_total_energy_threshold = 0.0005 # न्यूनतम औसत ऊर्जा स्पष्टता के लिए
        self.gender_female_pitch_threshold = 165 # महिला पिच के लिए थ्रेशोल्ड (Hz)

def extract_features_with_clarity(file_path: str, config: AudioAnalyzerConfig) -> tuple:
    """
    ऑडियो फ़ाइल से पिच, स्पष्टता संदेश, SNR और कुल ऊर्जा निकालता है।

    Args:
        file_path (str): ऑडियो फ़ाइल का पथ।
        config (AudioAnalyzerConfig): विश्लेषण के लिए कॉन्फ़िगरेशन ऑब्जेक्ट।

    Returns:
        tuple: (avg_pitch, clarity_message, snr_db, total_energy)
               यदि कोई त्रुटि होती है, तो (None, त्रुटि संदेश, None, None)
    """
    try:
        y, sr = librosa.load(file_path, sr=None) # sr=None original sample rate रखता है
        if y.size == 0:
            logging.warning(f"Audio file '{file_path}' is empty or too short.")
            return None, "Error: Audio file is empty or too short.", None, None
    except FileNotFoundError:
        logging.error(f"Error: Audio file not found at '{file_path}'")
        return None, "Error: Audio file not found.", None, None
    except Exception as e:
        logging.error(f"Error loading audio file '{file_path}': {e}")
        return None, f"Error loading audio file: {e}", None, None

    avg_pitch: float = 0.0
    # 1. पिच निकालना
    try:
        # pyin अधिक मजबूत है और voiced/unvoiced जानकारी भी देता है
        pitch, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=config.pitch_fmin_hz,
            fmax=config.pitch_fmax_hz,
            sr=sr,
            frame_length=2048, # default value, good for voice
            win_length=2048    # default value
        )

        # अमान्य पिच मानों को हटा दें (nan) और केवल वॉयस्ड सेगमेंट से पिच लें
        valid_pitch = pitch[~np.isnan(pitch) & voiced_flag]

        if len(valid_pitch) == 0:
            logging.info(f"No valid pitch detected for '{file_path}'.")
            avg_pitch = 0.0
        else:
            avg_pitch = np.mean(valid_pitch)

    except Exception as e:
        logging.error(f"Error extracting pitch for '{file_path}': {e}")
        avg_pitch = 0.0 # यदि पिच निकालने में त्रुटि होती है

    # 2. सिग्नल-टू-नॉइज़ रेश्यो (SNR) का अनुमान लगाना
    # यह एक सरल SNR अनुमान है. अधिक सटीक तरीकों के लिए उन्नत DSP या ML की आवश्यकता होती है।
    
    frame_length = int(config.frame_length_ms / 1000 * sr)
    hop_length = int(config.hop_length_ms / 1000 * sr)
    
    if frame_length == 0 or hop_length == 0:
        logging.warning("Frame or hop length calculated as zero, defaulting to fixed values.")
        frame_length = 2048
        hop_length = 512

    try:
        # librosa.util.frame की बजाय librosa.feature.rms का उपयोग करें जो अधिक सीधे है
        rms_energy = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

        if rms_energy.size == 0:
            logging.warning(f"RMS energy could not be calculated for '{file_path}'.")
            snr_db = -float('inf') # संकेत देता है कि सिग्नल बहुत कमजोर है
            total_energy = 0.0
        else:
            # नॉइज़ का अनुमान (शुरुआती X सेकंड से)
            noise_frames_count = int(config.noise_estimation_duration_s * sr / hop_length)
            noise_rms = rms_energy[:noise_frames_count]
            
            if len(noise_rms) > 0:
                noise_energy = np.mean(noise_rms) # RMS को ऊर्जा के रूप में उपयोग करें
            else:
                noise_energy = np.min(rms_energy[rms_energy > 0]) if np.any(rms_energy > 0) else 1e-6 # न्यूनतम गैर-शून्य RMS या बहुत छोटा मान

            # सिग्नल एनर्जी (आवाज वाले हिस्सों से) - एक सरल अनुमान
            signal_energy = np.percentile(rms_energy, config.signal_energy_percentile)
            
            # SNR की गणना (dB में)
            # 0 से विभाजन से बचने के लिए बहुत छोटे मान जोड़ें
            snr_db = 10 * np.log10((signal_energy + 1e-9) / (noise_energy + 1e-9))

            # 3. कुल ऑडियो ऊर्जा (RMS ऊर्जा का औसत)
            total_energy = np.mean(rms_energy)
            
    except Exception as e:
        logging.error(f"Error calculating SNR or energy for '{file_path}': {e}")
        snr_db = -float('inf') # त्रुटि होने पर बहुत कम SNR
        total_energy = 0.0

    # 4. स्पष्टता की जाँच (थ्रेशोल्ड आधारित)
    clarity_message: str = "voice clear"
    reasons: list = []

    if snr_db < config.min_snr_threshold_db:
        reasons.append(f"High background noise (SNR: {snr_db:.2f} dB)")
    
    if total_energy < config.min_total_energy_threshold:
        reasons.append(f"Too low volume or silence (Avg RMS: {total_energy:.4f})")
    
    # यदि पिच लगभग 0 है और कोई अन्य स्पष्टता समस्या नहीं है, तो यह भी अस्पष्टता का संकेत दे सकता है
    # यह तब होता है जब ऑडियो में कोई स्पष्ट स्वर नहीं होता (जैसे केवल शोर या संगीत)
    if avg_pitch == 0.0 and not reasons:
        reasons.append("No clear speech pitch detected")

    if reasons:
        clarity_message = "Voice not clear: " + ", ".join(reasons)
    
    return avg_pitch, clarity_message, snr_db, total_energy

def predict_gender_with_clarity_check(file_path: str, config: AudioAnalyzerConfig) -> tuple:
    """
    ऑडियो फ़ाइल के लिए लिंग का अनुमान लगाता है और स्पष्टता की जाँच करता है।

    Args:
        file_path (str): ऑडियो फ़ाइल का पथ।
        config (AudioAnalyzerConfig): विश्लेषण के लिए कॉन्फ़िगरेशन ऑब्जेक्ट।

    Returns:
        tuple: (gender_prediction: str, avg_pitch: float | None)
               gender_prediction स्पष्टता संदेश या अनुमानित लिंग होगा।
               avg_pitch केवल तभी होगा जब लिंग का सफलतापूर्वक अनुमान लगाया गया हो।
    """
    avg_pitch, clarity_message, snr_db, total_energy = extract_features_with_clarity(file_path, config)

    # यदि extract_features_with_clarity से कोई त्रुटि या स्पष्टता समस्या है
    if avg_pitch is None: # यदि फाइल लोड करने में त्रुटि हुई
        return clarity_message, None

    if clarity_message != "voice clear":
        logging.info(f"Clarity check failed for '{file_path}': {clarity_message}")
        return clarity_message, None
    
    # लिंग का अनुमान
    # यदि कोई वैध पिच नहीं मिली, तो लिंग निर्धारित नहीं किया जा सकता है
    if avg_pitch == 0.0:
        logging.info(f"Cannot determine gender for '{file_path}': No clear pitch detected after clarity check.")
        return "Cannot determine gender: No clear pitch detected", None
    elif avg_pitch > config.gender_female_pitch_threshold:
        gender = "Female"
    else:
        gender = "Male"
        
    logging.info(f"Gender predicted for '{file_path}': {gender} (Pitch: {avg_pitch:.2f} Hz)")
    return gender, avg_pitch

# --- उपयोग का उदाहरण ---
if __name__ == "__main__":
    # कॉन्फ़िगरेशन ऑब्जेक्ट बनाएं
    app_config = AudioAnalyzerConfig()

    # यहां अपने वास्तविक ऑडियो फ़ाइलों के पथ डालें
    # सुनिश्चित करें कि ये फाइलें मौजूद हैं और पठनीय हैं।
    clear_male_audio_path = "path/to/your/clear_male_voice.wav"
    clear_female_audio_path = "path/to/your/clear_female_voice.wav"
    noisy_audio_path = "path/to/your/noisy_voice.wav"
    low_volume_audio_path = "path/to/your/low_volume_voice.wav"
    empty_or_bad_audio_path = "path/to/your/non_existent_or_corrupt_file.wav"

    # यदि आपके पास परीक्षण के लिए वास्तविक फ़ाइलें नहीं हैं, तो कुछ डमी फ़ाइलें बनाएं
    # ध्यान दें: ये डमी फाइलें वास्तविक दुनिया के ऑडियो की जटिलता को सटीक रूप से प्रस्तुत नहीं करेंगी।
    # वास्तविक परीक्षण के लिए, वास्तविक रिकॉर्डिंग का उपयोग करें।
    from scipy.io.wavfile import write
    sample_rate = 22050
    duration = 3 # seconds
    t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False) # Fix: endpoint=False to avoid off-by-one sample
    
    # Create dummy files for demonstration
    try:
        # Clear Male Voice
        frequency_male = 120 # Hz
        data_male = 0.5 * np.sin(2. * np.pi * frequency_male * t)
        sf.write("dummy_clear_male.wav", data_male.astype(np.float32), sample_rate)
        clear_male_audio_path = "dummy_clear_male.wav"

        # Clear Female Voice
        frequency_female = 220 # Hz
        data_female = 0.5 * np.sin(2. * np.pi * frequency_female * t)
        sf.write("dummy_clear_female.wav", data_female.astype(np.float32), sample_rate)
        clear_female_audio_path = "dummy_clear_female.wav"

        # Noisy Voice
        noise = np.random.normal(0, 0.2, len(t))
        data_noisy = 0.3 * np.sin(2. * np.pi * 150 * t) + noise
        sf.write("dummy_noisy_voice.wav", data_noisy.astype(np.float32), sample_rate)
        noisy_audio_path = "dummy_noisy_voice.wav"

        # Low Volume Voice
        data_low_volume = 0.01 * np.sin(2. * np.pi * 180 * t)
        sf.write("dummy_low_volume_voice.wav", data_low_volume.astype(np.float32), sample_rate)
        low_volume_audio_path = "dummy_low_volume_voice.wav"

    except Exception as e:
        print(f"Could not create dummy audio files: {e}. Please ensure 'scipy' is installed and soundfile works.")
        print("You will need to manually provide paths to existing audio files.")


    test_files = {
        "Clear Male Voice": clear_male_audio_path,
        "Clear Female Voice": clear_female_audio_path,
        "Noisy Voice": noisy_audio_path,
        "Low Volume Voice": low_volume_audio_path,
        "Non-existent/Corrupt File": empty_or_bad_audio_path
    }

    for name, path in test_files.items():
        print(f"\n--- {name} Example ({path}) ---")
        gender_result, pitch_val = predict_gender_with_clarity_check(path, app_config)
        if pitch_val is not None:
            print(f"Predicted Gender: {gender_result}, Average Pitch: {pitch_val:.2f} Hz")
        else:
            print(f"Message: {gender_result}")

    # डमी फाइलों को हटा दें
    import os
    dummy_files = [
        "dummy_clear_male.wav",
        "dummy_clear_female.wav",
        "dummy_noisy_voice.wav",
        "dummy_low_volume_voice.wav"
    ]
    for df in dummy_files:
        if os.path.exists(df):
            os.remove(df)
            logging.info(f"Removed dummy file: {df}")
