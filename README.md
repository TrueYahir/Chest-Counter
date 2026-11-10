# Chest Counter

A Python-based desktop application that listens to your system's audio in real-time and automatically counts how many chests you open in Fortnite using a trained Machine Learning model.

## Features
* **Real-Time Audio Detection:** Uses `pyaudiowpatch` and `librosa` to capture desktop audio and extract MFCC features on the fly.
* **AI-Powered:** Utilizes a pre-trained scikit-learn Random Forest model (`joblib`) to identify the exact acoustic signature of a Fortnite chest opening.
* **In-Game Overlay (HUD):** When active, the main window hides and displays a minimalistic, transparent, and draggable counter overlay.
* **Modern UI:** Built with `customtkinter` for a sleek, dark-mode graphical interface.
* **Persistent Data:** Automatically saves your chest count, selected audio device, and threshold configurations.

## Release / Executable Version
If you downloaded the compiled `.exe` version, you do not need to install Python. Just make sure the following files are kept together in the exact same folder:
* `gui_app.exe` (or whatever you named the executable)
* `chest_model.pkl`
* `icon.ico`
* `kofi.png`
* `github.png`
* `config.json` (This will be auto-generated if missing)

Run the `.exe` and you are good to go!

## Running from Source
If you want to run or modify the Python scripts directly, follow these steps:

### Prerequisites
* Python 3.8 or higher.
* C++ Build Tools (sometimes required by `librosa` or `soundfile` on Windows).

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/TrueYahir/Fortnite-Chest-Counter.git](https://github.com/TrueYahir/Fortnite-Chest-Counter.git)
   cd Fortnite-Chest-Counter

2. Create a Virtual environmet:
    ```bash
    python -m venv .venv
    .venv\Scripts\activate

3. Install dependencies:
    ```bash
    pip install customtkinter pyaudiowpatch librosa numpy joblib scikit-learn pillow

### How to use
1. Audio Device: In the dropdown menu, select your system's output loopback device (usually labeled with [Loopback] or Stereo Mix).

2. Model: Ensure chest_model.pkl is typed in the Model field.

3. Threshold & Chunk: You can tweak the chunk duration (default 0.5) and probability threshold (default 0.8). Lowering the threshold makes it more sensitive but might cause false positives.

4. Run: Click the run button. The main menu will disappear, and a small counter HUD will appear.

5. HUD Controls:

    - Left-Click & Drag: Move the counter anywhere on your screen.

    - Right-Click: Open the context menu to close the HUD and return to the main settings.

## Support
If you like this project or it helped you out, consider supporting the development!
[![Ko-fi](kofi.png)](https://ko-fi.com/trueyahir)
[![GitHub](github.png)](https://github.com/TrueYahir)