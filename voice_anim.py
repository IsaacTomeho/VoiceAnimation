import sys
import numpy as np
import pyaudio
from vispy import app, scene
from vispy.scene.visuals import Line, Text
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, lfilter
import speech_recognition as sr
import threading
import logging
import wave
from pydub import AudioSegment
import io
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)

# Audio configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Speech recognition configuration
BUFFER_SECONDS = 5  # Buffer audio data for 3 seconds
RECOGNIZE_INTERVAL = 5  # Recognize speech every 2 seconds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence level for recognized text

# Filter settings
filter_type = 'low'
cutoff_freq = 5000

def apply_filter(data, filter_type, cutoff_freq):
    nyq = 0.5 * RATE
    if filter_type == 'low':
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(5, normal_cutoff, btype='low')
    elif filter_type == 'high':
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(5, normal_cutoff, btype='high')
    elif filter_type == 'band':
        low_cutoff = max(cutoff_freq - 1000, 100)
        high_cutoff = min(cutoff_freq + 1000, 10000)
        low_normal_cutoff = low_cutoff / nyq
        high_normal_cutoff = high_cutoff / nyq
        b, a = butter(5, [low_normal_cutoff, high_normal_cutoff], btype='band')
    else:
        return data
    return lfilter(b, a, data)

# Prepare the vispy canvas
canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
view = canvas.central_widget.add_view()
view.camera = scene.cameras.TurntableCamera(up='z', fov=60)
line = Line(color='red', width=2, parent=view.scene)
points = np.zeros((CHUNK, 3))

# Subtitle text
subtitle_text = Text(font_size=14, color='white', anchor_x='center', anchor_y='bottom', parent=view.scene)
subtitle_text.pos = (0, -0.5, 0)

# Initialize PyAudio and Speech Recognition
p = pyaudio.PyAudio()
recognizer = sr.Recognizer()
lock = threading.Lock()
audio_buffer = deque(maxlen=int(RATE / CHUNK * BUFFER_SECONDS))
recognizing = threading.Event()

def getAudio(audio_segment):
    try:
        # Export to WAV for recognition
        with io.BytesIO() as wav_buffer:
            audio_segment.export(wav_buffer, format='wav')
            wav_buffer.seek(0)
            with sr.AudioFile(wav_buffer) as source:
                audio = recognizer.record(source)
                result = recognizer.recognize_google(audio, language='en-US', show_all=True)
                if 'alternative' in result:
                    for alt in result['alternative']:
                        if 'confidence' in alt and alt['confidence'] >= CONFIDENCE_THRESHOLD:
                            return alt['transcript']
                return "... ... ..."
    except sr.UnknownValueError:
        return "Speech not recognized"
    except sr.RequestError as e:
        logging.error(f"Could not request results; {e}")
        return "Error: Speech recognition service down"

def recognize_speech():
    global audio_buffer

    while True:
        recognizing.wait()  # Wait for the next recognition signal
        with lock:
            if not audio_buffer:
                recognizing.clear()
                continue
            audio_data = b''.join(audio_buffer)
            audio_buffer.clear()

        try:
            # Convert audio data to WAV format using pydub
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=RATE,
                channels=1
            )

            # Apply noise reduction
            audio_segment = audio_segment.remove_dc_offset().normalize()

            # Recognize speech
            text = getAudio(audio_segment)

            with lock:
                subtitle_text.text = text
        except Exception as e:
            logging.error("Speech recognition exception: %s", str(e))
            with lock:
                subtitle_text.text = "Error: " + str(e)

        recognizing.clear()

# Start speech recognition thread
recognition_thread = threading.Thread(target=recognize_speech, daemon=True)
recognition_thread.start()

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    filtered_data = apply_filter(audio_data, filter_type, cutoff_freq)
    normalized_data = filtered_data / 32768.0
    points[:, 0] = np.linspace(-1, 1, num=CHUNK)
    points[:, 1] = normalized_data
    line.set_data(points)

    # Accumulate audio data for speech recognition
    with lock:
        audio_buffer.append(in_data)

    return (in_data, pyaudio.paContinue)

def start_recognition_timer():
    if not recognizing.is_set():
        recognizing.set()

    QtCore.QTimer.singleShot(RECOGNIZE_INTERVAL * 1000, start_recognition_timer)

stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                frames_per_buffer=CHUNK, stream_callback=audio_callback)

# GUI Controls using PyQt5
class ControlWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.layout = QtWidgets.QVBoxLayout()
        self.filter_type_combo = QtWidgets.QComboBox()
        self.filter_type_combo.addItems(['low', 'high', 'band'])
        self.filter_type_combo.currentTextChanged.connect(self.update_filter_type)
        self.cutoff_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.cutoff_slider.setRange(100, 10000)
        self.cutoff_slider.setValue(5000)
        self.cutoff_slider.valueChanged.connect(self.update_cutoff_frequency)
        self.layout.addWidget(self.filter_type_combo)
        self.layout.addWidget(self.cutoff_slider)
        self.setLayout(self.layout)

    def update_filter_type(self, value):
        global filter_type
        filter_type = value

    def update_cutoff_frequency(self, value):
        global cutoff_freq
        cutoff_freq = value

if __name__ == '__main__':
    appQt = QtWidgets.QApplication(sys.argv)
    control_window = ControlWindow()
    control_window.show()
    stream.start_stream()

    # Start the recognition timer to trigger speech recognition periodically
    start_recognition_timer()

    try:
        sys.exit(appQt.exec_())
    finally:
        stream.stop_stream()
        stream.close()
        recognition_thread.join()
        p.terminate()