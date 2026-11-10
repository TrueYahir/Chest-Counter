import sklearn
import customtkinter as ctk
import tkinter as tk
import pyaudiowpatch as pyaudio
import librosa
import numpy as np
import joblib
import threading
import json
import os
import webbrowser
from PIL import Image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

CONFIG_FILE = "config.json"

class ChestCounterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Fortnite Chest Counter")
        self.geometry("500x700")
        self.iconbitmap("icon.ico")
        
        self.is_running = False
        self.detector_thread = None
        self.audio_interface = pyaudio.PyAudio()
        self.devices_info = self.get_audio_devices()
        
        self.config = self.load_config()

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self.chest_label = ctk.CTkLabel(self, text="CHEST OPENED:", font=("Arial", 16))
        self.chest_label.grid(row=0, column=0, padx=20, pady=(30, 10), sticky="w")
        
        self.chest_value = ctk.CTkLabel(self, text=str(self.config["chests"]), font=("Arial", 24, "bold"))
        self.chest_value.grid(row=0, column=1, padx=20, pady=(30, 10), sticky="e")

        self.save_data_var = ctk.BooleanVar(value=self.config["save_data"])
        self.save_checkbox = ctk.CTkCheckBox(self, text="Save data", variable=self.save_data_var)
        self.save_checkbox.grid(row=1, column=0, padx=20, pady=10, sticky="w")

        ctk.CTkLabel(self, text="Model").grid(row=2, column=0, padx=20, pady=(15, 0), sticky="w")
        self.model_entry = ctk.CTkEntry(self, width=460)
        self.model_entry.grid(row=3, column=0, columnspan=2, padx=20, pady=5)
        self.model_entry.insert(0, self.config["model"])

        ctk.CTkLabel(self, text="Chunk duration").grid(row=4, column=0, padx=20, pady=(15, 0), sticky="w")
        self.chunk_entry = ctk.CTkEntry(self, width=460)
        self.chunk_entry.grid(row=5, column=0, columnspan=2, padx=20, pady=5)
        self.chunk_entry.insert(0, str(self.config["chunk_duration"]))

        ctk.CTkLabel(self, text="Probability threshold").grid(row=6, column=0, padx=20, pady=(15, 0), sticky="w")
        self.prob_entry = ctk.CTkEntry(self, width=460)
        self.prob_entry.grid(row=7, column=0, columnspan=2, padx=20, pady=5)
        self.prob_entry.insert(0, str(self.config["probability_threshold"]))

        ctk.CTkLabel(self, text="Audio device").grid(row=8, column=0, padx=20, pady=(15, 0), sticky="w")
        device_names = [d["name"] for d in self.devices_info]
        self.device_combo = ctk.CTkOptionMenu(self, values=device_names, width=460)
        self.device_combo.grid(row=9, column=0, columnspan=2, padx=20, pady=5)
        if self.config["device"] in device_names:
            self.device_combo.set(self.config["device"])
        elif device_names:
            self.device_combo.set(device_names[0])

        try:
            kofi_img_data = Image.open("kofi.png")
            self.kofi_image = ctk.CTkImage(light_image=kofi_img_data, dark_image=kofi_img_data, size=(20, 20))
        except Exception:
            self.kofi_image = None
            
        try:
            github_img_data = Image.open("github.png")
            self.github_image = ctk.CTkImage(light_image=github_img_data, dark_image=github_img_data, size=(20, 20))
        except Exception:
            self.github_image = None

        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.grid(row=10, column=0, columnspan=2, padx=20, pady=(30, 10), sticky="e")

        self.save_btn = ctk.CTkButton(self.action_frame, text="save", width=120, command=self.save_config)
        self.save_btn.pack(side="left", padx=(0, 10))

        self.run_btn = ctk.CTkButton(self.action_frame, text="run", width=120, command=self.toggle_detector)
        self.run_btn.pack(side="left")

        self.grid_rowconfigure(11, weight=1)

        self.social_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.social_frame.grid(row=12, column=0, columnspan=2, padx=20, pady=20, sticky="sw")

        self.kofi_btn = ctk.CTkButton(self.social_frame, text="ko-fi", image=self.kofi_image, width=120, fg_color="transparent", border_width=1, command=self.open_kofi)
        self.kofi_btn.pack(side="left", padx=(0, 10))

        self.github_btn = ctk.CTkButton(self.social_frame, text="github", image=self.github_image, width=120, fg_color="transparent", border_width=1, command=self.open_github)
        self.github_btn.pack(side="left")

        self.overlay = None
        self.overlay_label = None

    def load_config(self):
        default_config = {
            "chests": 0,
            "save_data": True,
            "model": "chest_model.pkl",
            "chunk_duration": 0.5,
            "probability_threshold": 0.8,
            "device": ""
        }
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    data = json.load(f)
                    default_config.update(data)
            except Exception:
                pass
        return default_config

    def save_config(self):
        self.config["save_data"] = self.save_data_var.get()
        self.config["model"] = self.model_entry.get()
        self.config["chunk_duration"] = float(self.chunk_entry.get())
        self.config["probability_threshold"] = float(self.prob_entry.get())
        self.config["device"] = self.device_combo.get()
        
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=4)

    def save_chests_count(self):
        if self.save_data_var.get():
            self.config["chests"] = int(self.chest_value.cget("text"))
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)

    def get_audio_devices(self):
        devices = list(self.audio_interface.get_loopback_device_info_generator())
        if not devices:
            for i in range(self.audio_interface.get_device_count()):
                dev = self.audio_interface.get_device_info_by_index(i)
                if dev.get("maxInputChannels") > 0:
                    devices.append(dev)
        return devices

    def open_kofi(self):
        webbrowser.open("https://ko-fi.com/trueyahir")

    def open_github(self):
        webbrowser.open("https://github.com/TrueYahir")

    def show_overlay(self):
        self.withdraw()
        
        self.overlay = ctk.CTkToplevel(self)
        self.overlay.geometry("100x100+50+50")
        self.overlay.overrideredirect(True)
        self.overlay.attributes("-topmost", True)
        self.overlay.attributes("-alpha", 0.7)
        self.overlay.configure(fg_color="#111111")

        self.overlay_label = ctk.CTkLabel(
            self.overlay, 
            text=self.chest_value.cget("text"), 
            font=("Arial", 40, "bold"), 
            text_color="white"
        )
        self.overlay_label.pack(expand=True, fill="both")

        self.overlay.bind("<ButtonPress-1>", self.start_move)
        self.overlay.bind("<B1-Motion>", self.do_move)

        self.context_menu = tk.Menu(self.overlay, tearoff=0, bg="#2b2b2b", fg="white")
        self.context_menu.add_command(label="Close HUD", command=self.close_overlay)

        self.overlay.bind("<Button-3>", self.show_context_menu)

    def start_move(self, event):
        self.overlay.x = event.x
        self.overlay.y = event.y

    def do_move(self, event):
        x = self.overlay.winfo_pointerx() - self.overlay.x
        y = self.overlay.winfo_pointery() - self.overlay.y
        self.overlay.geometry(f"+{x}+{y}")

    def show_context_menu(self, event):
        self.context_menu.tk_popup(event.x_root, event.y_root)

    def close_overlay(self):
        if self.overlay:
            self.overlay.destroy()
            self.overlay = None
            self.overlay_label = None
        
        self.is_running = False
        self.run_btn.configure(text="run", fg_color=["#3a7ebf", "#1f538d"])
        self.save_chests_count()
        self.deiconify()

    def toggle_detector(self):
        if not self.is_running:
            self.is_running = True
            self.run_btn.configure(text="stop", fg_color="red")
            self.save_config()
            self.show_overlay()
            self.detector_thread = threading.Thread(target=self.run_detection_loop, daemon=True)
            self.detector_thread.start()

    def extract_features(self, audio_data, sr):
        if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean

    def update_ui_counter(self, count):
        self.chest_value.configure(text=str(count))
        if self.overlay and self.overlay_label:
            self.overlay_label.configure(text=str(count))
        self.save_chests_count()

    def run_detection_loop(self):
        model_path = self.model_entry.get()
        chunk_dur = float(self.chunk_entry.get())
        prob_thresh = float(self.prob_entry.get())
        
        if prob_thresh > 1.0:
            prob_thresh = prob_thresh / 100.0

        selected_device_name = self.device_combo.get()
        
        try:
            model = joblib.load(model_path)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.after(0, self.close_overlay)
            return

        device = next((d for d in self.devices_info if d["name"] == selected_device_name), None)
        if not device:
            print("Audio device not found.")
            self.after(0, self.close_overlay)
            return

        print(f"Listening on device: {device['name']}")

        channels = int(device["maxInputChannels"])
        if channels == 0: channels = 2 
        sample_rate = int(device["defaultSampleRate"])
        CHUNK = 1024

        stream = None
        try:
            stream = self.audio_interface.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device["index"],
                frames_per_buffer=CHUNK
            )

            chunks_per_second = int(sample_rate / CHUNK * chunk_dur)
            is_chest_active = False
            background_consecutive_frames = 0
            current_chests = int(self.chest_value.cget("text"))

            print("--- STARTING AUDIO DETECTION ---")

            while self.is_running:
                frames = []
                for _ in range(chunks_per_second):
                    if not self.is_running:
                        break
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                
                if not self.is_running:
                    break

                audio_data = np.frombuffer(b"".join(frames), dtype=np.int16)
                audio_data = audio_data.reshape(-1, channels).astype(np.float32)
                audio_data = audio_data / 32768.0

                max_vol = np.max(np.abs(audio_data))
                
                if max_vol < 0.0005: 
                    print(f"Silence... (Volume: {max_vol:.5f})")
                    background_consecutive_frames += 1
                    if background_consecutive_frames >= 3:
                        is_chest_active = False
                    continue
                    
                features = self.extract_features(audio_data, sample_rate)
                features = features.reshape(1, -1)
                
                probabilities = model.predict_proba(features)[0]
                chest_probability = probabilities[1]
                
                print(f"Audio detected -> Volume: {max_vol:.4f} | Chest Probability: {chest_probability:.4f} | Threshold req: {prob_thresh}")
                
                if chest_probability >= prob_thresh:
                    background_consecutive_frames = 0
                    if not is_chest_active:
                        print("CHEST DETECTED AND ADDED!")
                        current_chests += 1
                        self.after(0, self.update_ui_counter, current_chests)
                        is_chest_active = True
                else:
                    background_consecutive_frames += 1
                    if background_consecutive_frames >= 3:
                        is_chest_active = False

        except Exception as e:
            print(f"Error in detection stream: {e}")
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            print("--- DETECTION STOPPED ---")

if __name__ == "__main__":
    app = ChestCounterApp()
    app.mainloop()