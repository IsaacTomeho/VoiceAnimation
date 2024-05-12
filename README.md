# Voice-Driven Animation System

## Overview
The Voice-Driven Animation System is a Python application that uses real-time audio processing and speech recognition to create dynamic visualizations of spoken words. The system captures audio, processes and recognizes speech, and then visually animates the words on a screen.

## Features
- **Real-Time Audio Processing**: Uses PyAudio to capture live audio data.
- **Speech Recognition**: Incorporates Google's speech recognition to convert spoken words into text.
- **Dynamic Visualization**: Utilizes Vispy for real-time visualization of the audio signals and text.
- **Interactive GUI**: Features a PyQt5-based GUI for real-time control over audio processing settings.

## Technologies Used
- **Python**: Primary programming language.
- **PyAudio**: For capturing audio data.
- **Vispy**: For rendering visual animations.
- **SciPy**: Provides signal processing capabilities.
- **speech_recognition**: Handles the conversion of speech to text.
- **PyQt5**: For the graphical user interface.
- **PyDub**: Used for manipulating audio data.
- **Threading**: Supports concurrent execution for seamless audio processing and UI interaction.

## Instal the required packages
pip install numpy pyaudio vispy PyQt5 scipy SpeechRecognition pydub

