import requests
import pyttsx3
import time

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Speed of speech
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech failed:", e)

def correct_grammar_with_deepseek(text):
    prompt = (
        f"Correct the grammar of this sentence and complete it: \"{text}\".\n"
        "Only reply with the corrected sentence. No other explanation, character, key, word, etc. even if you feel you should type something out, don't do it. I just want the corrected sentence, that is it. I don't want you to say, the corrected sentence is blah blah."
    )

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "deepseek-llm",
                "prompt": prompt,
                "stream": False
            }
        )

        if response.status_code == 200:
            corrected = response.json()["response"].strip().strip('"\n ')
            print("\n📝 Corrected Sentence:")
            print(corrected)
            speak_text(corrected)
            return corrected
        else:
            print("Error from model:", response.status_code)
            return None
    except Exception as e:
        print("Connection failed:", e)
        return None

while True:
    user_input = input("\n✍️ Enter a sentence to correct (or type 'q' to quit):\n> ")
    if user_input.lower() == 'q':
        break
    correct_grammar_with_deepseek(user_input)
