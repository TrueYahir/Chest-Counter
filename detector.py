import pyaudiowpatch as pyaudio
import librosa
import numpy as np
import joblib

CHUNK_DURATION = 0.5
PROBABILITY_THRESHOLD = 0.80

def extract_features_from_array(audio_data, sr):
    if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
        audio_data = np.mean(audio_data, axis=1)
        
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

def main():
    print("Loading model...")
    try:
        model = joblib.load("chest_model.pkl")
    except FileNotFoundError:
        print("Error: 'chest_model.pkl' not found.")
        return

    audio = pyaudio.PyAudio()
    
    print("\nAvailable loopback devices:")
    devices = list(audio.get_loopback_device_info_generator())
    
    if not devices:
        print("No loopback devices found.")
        return
        
    for index, device in enumerate(devices):
        print(f"{index + 1}. {device['name']}")

    try:
        selection = int(input("\nSelect your device: ")) - 1
        device = devices[selection]
    except (ValueError, IndexError):
        print("Invalid selection.")
        audio.terminate()
        return

    channels = int(device["maxInputChannels"])
    sample_rate = int(device["defaultSampleRate"])
    CHUNK = 1024

    print(f"\nListening on: {device['name']}")
    print("Waiting for chest sounds... (Press Ctrl+C to stop)")

    stream = None
    try:
        stream = audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK
        )

        chest_count = 0
        chunks_per_second = int(sample_rate / CHUNK * CHUNK_DURATION)
        
        is_chest_active = False
        background_consecutive_frames = 0

        while True:
            frames = []
            for _ in range(chunks_per_second):
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            
            audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
            audio_data = audio_data.reshape(-1, channels).astype(np.float32)
            
            audio_data = audio_data / 32768.0

            if np.max(np.abs(audio_data)) < 0.005:
                background_consecutive_frames += 1
                if background_consecutive_frames >= 3:
                    is_chest_active = False
                continue
                
            features = extract_features_from_array(audio_data, sample_rate)
            features = features.reshape(1, -1)
            
            probabilities = model.predict_proba(features)[0]
            chest_probability = probabilities[1]
            
            if chest_probability >= PROBABILITY_THRESHOLD:
                background_consecutive_frames = 0
                if not is_chest_active:
                    chest_count += 1
                    is_chest_active = True
                    print(f"Chest detected! (Confidence: {chest_probability:.2f}) Total opened: {chest_count}", flush=True)
            else:
                background_consecutive_frames += 1
                if background_consecutive_frames >= 3:
                    is_chest_active = False

    except KeyboardInterrupt:
        print(f"\nFinished. Total chests opened: {chest_count}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        audio.terminate()

if __name__ == "__main__":
    main()