import requests
import pyttsx3
import time

def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)      # Speed of speech
        engine.setProperty('volume', 1.0)    # Volume (0.0 to 1.0)

        # Uncomment this to try a female voice if available
        # voices = engine.getProperty('voices')
        # for voice in voices:
        #     if "female" in voice.name.lower():
        #         engine.setProperty('voice', voice.id)
        #         break

        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Speech failed:", e)

def correct_grammar_with_deepseek(text):
    prompt = (
        f"Correct the grammar of this sentence and complete it: \"{text}\".\n"
        "Only reply with the corrected sentence. No other explanation, character, key, word, etc. even if you feel you should type something out, dont do it. I just want the corrected sentence, that is it. I dont want you to say, the corrected sentence is blah blah"
    )

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
        print("Original:", text)
        print("Corrected:", corrected)

        time.sleep(0.5)  # Small delay before speaking
        speak_text(corrected)

        return corrected
    else:
        print("Error:", response.status_code)
        return None
    
    
correct_grammar_with_deepseek("At school, a boy hit my face, then I cried")
