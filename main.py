import io
import mic
import time
import pygame
import openai

from gtts import gTTS
from secret_key import openai_key

start_time = time.time()

history = []

openai.api_key = openai_key

def chat(question):
    # messages = [
    #     {"role": "system", "content": personality},
    #     {"role": "user", "content": question}
    # ]

    # Add the message into the history
    history.append({"role": 'user', "content": question})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=history,
        max_tokens=1000,
        # stream=True
    )

    history.append({
        "role": response.choices[0].message['role'],
        "content": response.choices[0].message['content']
    })

    return response.choices[0].message['content']

def main():
    while True:
        audio = mic.record_audio()
        text, whisper_time = mic.transcribe_forever(audio)
        print("\nWhisper's Transcribe Result:" + text)
        print("Whisper Time Taken: {:.2f} seconds".format(whisper_time))

        print("\nMessage Sent to ChatGPT. Generating Response...")
        
        start_time = time.time()
        responses = chat(text)
        end_time = time.time()

        print("\nChatGPT Response:\n\n",responses)
        print("\nChatGPT Time Taken: {:.2f} seconds".format(end_time - start_time))

        print("\nSpeaking...")

        # for response in responses:
        #     generated_text = response['choices'][0]['delta']["content"]
        #     decoded_text = generated_text
        #     print(decoded_text, end="")
        
        # print("\n\n\n\n")

        tts = gTTS(responses, lang="en-us")

        # Save the TTS audio as an in-memory file
        audio_data = io.BytesIO()
        tts.write_to_fp(audio_data)
        audio_data.seek(0)  # Move to the beginning of the in-memory audio data


        pygame.mixer.init()
        pygame.mixer.music.load(audio_data)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.delay(100)

        

if __name__ == "__main__":
    main()