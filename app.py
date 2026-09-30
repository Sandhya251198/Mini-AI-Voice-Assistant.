from transformers import pipeline
from faster_whisper import WhisperModel

from datetime import datetime
import webbrowser
import threading
import queue

import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3
import tkinter as tk


# =========================================================
# 1. LOAD AI MODELS
# =========================================================

print("Loading AI model...")

assistant = pipeline(
    "text-generation",
    model="HuggingFaceTB/smolLM2-360M-Instruct"
)

print("Loading speech recognition model...")

model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Models loaded.")


# =========================================================
# 2. CONVERSATION MEMORY
# =========================================================

conversation_history = []


# =========================================================
# 3. TEXT TO SPEECH
# =========================================================

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print("TTS error:",e)

# =========================================================
# 4. GET CURRENT TIME
# =========================================================

def get_time():

    return datetime.now().strftime(
        "%I:%M %p"
    )


# =========================================================
# 5. CALCULATOR
# =========================================================

def calculate(expression):

    allowed = "0123456789+-*/(). "

    if not all(
        character in allowed
        for character in expression
    ):
        return "I can only calculate basic arithmetic."

    try:

        return str(eval(expression))

    except Exception:

        return "I could not calculate that."


# =========================================================
# 6. OPEN WEBSITE
# =========================================================

def open_website(website):

    try:

        webbrowser.open(website)

        return "Opening YouTube."

    except Exception:

        return "I could not open the website."


# =========================================================
# 7. SPEECH TO TEXT
# =========================================================

def listen():

    print("\nSpeak now...")

    samplerate = 16000

    block_duration = 0.1

    block_size = int(
        samplerate * block_duration
    )

    silence_threshold = 0.01

    silence_duration = 1.2

    audio_data = []

    silence_time = 0

    speech_started = False

    try:

        with sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="float32",
            blocksize=block_size
        ) as stream:

            while True:

                audio, overflowed = stream.read(
                    block_size
                )

                audio_data.append(
                    audio.copy()
                )

                volume = np.sqrt(
                    np.mean(audio ** 2)
                )

                print(
                    f"\rVolume: {volume:.4f}",
                    end=""
                )

                # User is speaking
                if volume > silence_threshold:

                    speech_started = True

                    silence_time = 0

                # User stopped speaking
                elif speech_started:

                    silence_time += block_duration

                    if silence_time >= silence_duration:

                        print(
                            "\nStopped listening."
                        )

                        break

        if not audio_data:

            return ""

        audio = np.concatenate(
            audio_data
        )

        sf.write(
            "test.wav",
            audio,
            samplerate
        )

        # Whisper
        segments, info = model.transcribe(
            "test.wav",
            language="en",
            vad_filter=True
        )

        text = ""

        for segment in segments:

            text += segment.text + " "

        return text.strip()

    except Exception as e:

        print(
            "\nSpeech recognition error:",
            e
        )

        return ""


# =========================================================
# 8. TOOL ROUTER
# =========================================================

def ask_llm(question):

    question = question.lower().strip()

    # -------------------------
    # TIME
    # -------------------------

    if any(
        word in question
        for word in [
            "time",
            "current time",
            "what time"
        ]
    ):

        return "get_time"


    # -------------------------
    # CALCULATOR
    # -------------------------

    elif (
        any(
            word in question
            for word in [
                "calculate",
                "solve",
                "what is"
            ]
        )
        and any(
            character.isdigit()
            for character in question
        )
    ):

        return "calculate"


    # -------------------------
    # YOUTUBE
    # -------------------------

    elif (
        "youtube" in question
        and any(
            word in question
            for word in [
                "open",
                "launch",
                "go to"
            ]
        )
    ):

        return "open_website"


    # -------------------------
    # NO TOOL
    # -------------------------

    else:

        return "none"


# =========================================================
# 9. GET TOOL ARGUMENT
# =========================================================

def get_argument(question, tool):

    question = question.lower().strip()


    # Calculator argument
    if tool == "calculate":

        expression = question

        for phrase in [
            "please calculate",
            "calculate",
            "what is",
            "solve"
        ]:

            expression = expression.replace(
                phrase,
                ""
            )

        # Convert common spoken operators
        expression = expression.replace(
            "multiplied by",
            "*"
        )

        expression = expression.replace(
            "times",
            "*"
        )

        expression = expression.replace(
            "plus",
            "+"
        )

        expression = expression.replace(
            "minus",
            "-"
        )

        expression = expression.replace(
            "divided by",
            "/"
        )
        numbers ={"zero":"0",
                  "one":"1",
                  "two":"2",
                  "three":"3",
                  "four":"4",
                  "five":"5",
                  "six":"6",
                  "seven":"7",
                  "eight":"8",
                  "nine":"9",
                  "ten":"10"}
        for word, number in numbers.items():
            expression = expression.replace(word,number)
        return expression.strip()


    # YouTube argument
    elif tool == "open_website":

        if "youtube" in question:

            return "https://www.youtube.com"


    # Time doesn't need an argument
    elif tool == "get_time":

        return None


    return None


# =========================================================
# 10. RUN TOOL
# =========================================================

def run_tool(tool, argument):

    if tool == "get_time":

        current_time = get_time()

        return (
            "The current time is "
            + current_time
        )


    elif tool == "calculate":

        result = calculate(
            argument
        )

        return (
            "The answer is "
            + result
        )


    elif tool == "open_website":

        return open_website(
            argument
        )


    return "I could not perform that action."


# =========================================================
# 11. AI / LLM
# =========================================================

def ask_ai(question):

    try:

        conversation_history.append(
            "User: " + question
        )

        # Keep recent conversation
        history = "\n".join(
            conversation_history[-10:]
        )

        prompt = f"""
You are a voice assistant.

Answer the user's question directly and naturally.

Rules:
- Do not say "Let me explain".
- Do not say "Let me break this down".
- Do not say "Sure".
- Do not repeat the user's question.
- Do not give unnecessary background information.
- Keep the answer concise.
- Do not make up information.
- Do not show your reasoning.

Conversation:
{history}

Assistant:
"""

        result = assistant(
            prompt,
            max_new_tokens=80,
            return_full_text=False
        )

        answer = result[0][
            "generated_text"
        ].strip()

        conversation_history.append(
            "Assistant: " + answer
        )

        return answer

    except Exception as e:

        print(
            "AI error:",
            e
        )

        return (
            "Sorry, I could not "
            "generate an answer."
        )


# =========================================================
# 12. GUI STATUS
# =========================================================

def update_status(message):

    root.after(
        0,
        lambda: status_label.config(
            text=message
        )
    )


# =========================================================
# 13. SHOW ANSWER IN GUI
# =========================================================

def show_answer(answer):

    conversation.insert(
        tk.END,
        "AI: " + answer + "\n\n",
        "ai"
    )

    conversation.see(
        tk.END
    )

    update_status(
        "🔊 Speaking..."
    )


# =========================================================
# 14. FINISH RESPONSE
# =========================================================

def finish_response():

    speak_button.config(
        state="normal",
        text="🎤 SPEAK"
    )

    update_status(
        "✅ Ready. Press SPEAK to talk again."
    )


# =========================================================
# 15. GENERATE ANSWER
# =========================================================

def generate_answer(question):

    # Decide whether a tool is needed
    tool = ask_llm(question)


    if tool != "none":

        argument = get_argument(
            question,
            tool
        )

        answer = run_tool(
            tool,
            argument
        )


    else:

        # General question → LLM
        answer = ask_ai(
            question
        )


    # Update GUI safely
    root.after(
        0,
        lambda: show_answer(answer)
    )


    # Speak answer
    speak(answer)


    # Enable button after speaking
    root.after(
        0,
        finish_response
    )
 
# =========================================================
# 16. PROCESS QUESTION
# =========================================================

def process_question(question):

    if not question:

        update_status(
            "❌ I didn't hear anything."
        )

        speak(
            "I didn't hear anything."
        )

        finish_response()

        return


    # Show user's question
    conversation.insert(
        tk.END,
        "You: " + question + "\n",
        "user"
    )

    conversation.see(
        tk.END
    )


    update_status(
        "🧠 Thinking..."
    )


    # AI processing happens in background
    threading.Thread(
        target=generate_answer,
        args=(question,),
        daemon=True
    ).start()


# =========================================================
# 17. VOICE PROCESSING
# =========================================================

def process_voice():

    try:

        question = listen()

        # Send result to Tkinter thread
        root.after(
            0,
            lambda: process_question(
                question
            )
        )

    except Exception as e:

        print(
            "Voice processing error:",
            e
        )

        root.after(
            0,
            finish_response
        )


# =========================================================
# 18. START LISTENING
# =========================================================

def start_listening():

    speak_button.config(
        state="disabled",
        text="🎤 Listening..."
    )

    update_status(
        "🎤 Speak now..."
    )


    threading.Thread(
        target=process_voice,
        daemon=True
    ).start()


# =========================================================
# 19. TKINTER GUI
# =========================================================

root = tk.Tk()

root.title(
    "Mini AI Voice Assistant"
)

root.geometry(
    "700x600"
)


# Title
title_label = tk.Label(
    root,
    text="🤖 Mini AI Voice Assistant",
    font=("Arial", 20, "bold")
)

title_label.pack(
    pady=15
)


# Status
status_label = tk.Label(
    root,
    text="✅ Ready. Press SPEAK to talk.",
    font=("Arial", 11)
)

status_label.pack(
    pady=5
)


# Conversation box
conversation = tk.Text(
    root,
    height=22,
    width=75,
    font=("Arial", 11),
    wrap=tk.WORD
)

conversation.pack(
    padx=20,
    pady=15
)


# Text styles
conversation.tag_config(
    "user",
    font=("Arial", 11, "bold")
)

conversation.tag_config(
    "ai",
    font=("Arial", 11)
)


# Speak button
speak_button = tk.Button(
    root,
    text="🎤 SPEAK",
    font=("Arial", 14, "bold"),
    command=start_listening,
    padx=30,
    pady=10
)

speak_button.pack(
    pady=10
)


# =========================================================
# 20. START GUI
# =========================================================

root.mainloop()