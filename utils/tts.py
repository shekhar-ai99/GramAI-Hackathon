# utils/tts.py
from gtts import gTTS
import os

def speak_odia(text: str) -> str:
    path = "result_odia.mp3"
    try:
        gTTS(text, lang='or').save(path)
    except:
        gTTS(text, lang='hi').save(path)
    return path