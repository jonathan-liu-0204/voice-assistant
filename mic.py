import speech_recognition as sr
import io
import torch
from pydub import AudioSegment
import numpy as np
import whisper

import asyncio
import time

def record_audio():
    #load the speech recognizer and set the initial energy threshold and pause threshold
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.pause_threshold = 0.8
    r.dynamic_energy_threshold = False

    with sr.Microphone(sample_rate=16000) as source:
        print(" \n\n\n\n\n\n\n\n\n\n\n ============================== \n\nSay something!")
        i = 0
        #get and save audio to wav file
        audio = r.listen(source)
        torch_audio = torch.from_numpy(np.frombuffer(audio.get_raw_data(), np.int16).flatten().astype(np.float32) / 32768.0)
        audio_data = torch_audio

        print("\nMessage sent to Whisper. Transcribing...")

        return audio_data

def transcribe_forever(audio):

    start_time = time.time()

    audio_data = audio
    audio_model = whisper.load_model("base")
    result = audio_model.transcribe(audio_data,language='zh') #Chinese model

    end_time = time.time()

    whisper_time = end_time - start_time

    #result = audio_model.transcribe(audio_data)


    predicted_text = result["text"]
    return predicted_text, whisper_time

    #result_queue.put_nowait(result) #Complete result