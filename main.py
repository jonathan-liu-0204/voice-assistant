import io
import mic
import pygame
import openai

from gtts import gTTS
from secret_key import openai_key

openai.api_key = openai_key

def chat(question, personality):
    messages = [
        {"role": "system", "content": personality},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )

    return response.choices[0].message['content']

while True:
    audio = mic.record_audio()
    text = mic.transcribe_forever(audio)
    print("You said:" + text)
    final = chat(text, "請你用繁體中文回答我這個問題")
    print(final)

    tts = gTTS(final, lang="zh-tw")

    # Save the TTS audio as an in-memory file
    audio_data = io.BytesIO()
    tts.write_to_fp(audio_data)
    audio_data.seek(0)  # Move to the beginning of the in-memory audio data


    pygame.mixer.init()
    pygame.mixer.music.load(audio_data)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.delay(100)
