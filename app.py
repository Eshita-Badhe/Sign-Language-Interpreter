import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import threading
import pyttsx3
import mediapipe as mp
import joblib
import numpy as np
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv() 

# -----Load trained mode -----
model = joblib.load('sign_model.pkl')  


# ----- Text-to-Speech -----
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()

class GestureDetector:
    def __init__(self, buffer_size=15):
        self.buffer_size = buffer_size
        self.buffer = []
        self.last_confirmed = None

    def confirm(self, prediction):
        self.buffer.append(prediction)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)

        if len(set(self.buffer)) == 1:
            if self.buffer[0] != self.last_confirmed:
                self.last_confirmed = self.buffer[0]

        return self.last_confirmed or "Detecting..."

gesture_detector = GestureDetector(buffer_size=15)

# ----- Gesture Recognition Function -----
def recognize_gesture_from_frame(frame, hands, mpDraw):
    h, w, _ = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            # Extract normalized landmarks (x, y, z)
            landmark_points = []
            for lm in handLms.landmark:
                landmark_points.extend([lm.x, lm.y, lm.z])

            # Predict gesture using trained model
            raw_prediction = model.predict([landmark_points])[0]

            # Use gesture detector to confirm after stability
            confirmed_prediction = gesture_detector.confirm(raw_prediction)

            return confirmed_prediction, frame

    return "No Hand Detected", frame

# Initialize OpenAI with your API key
genai.configure(api_key=os.environ["GENAI_API_KEY"])
gemini = genai.GenerativeModel('gemini-2.0-flash')

def polish_text_with_ai(raw_text):
    prompt = '''Convert the following sequence of sign language letters into a grammatically correct and meaningful English sentence.
- 'SPACE' means a space between words.
- Ignore noise words like 'DEL' or treat them as random errors — do not include them in the final output.
- Some words may have spelling mistakes — correct them using only the given letters.
- You may remove unwanted or extra letters to form proper words.
- Do not add any letters not present in the input.
- Output only the corrected sentence. Do not explain your reasoning.
Examples:
Input: K I T K SPACE I J SPACE X F L Y I A N D G
Output: Kite is flying!

Input: D K I T K
Output: The kite

Input: K I L E
Output: Kite

Input: A SPACE D E L H O N E SPACE I S SPACE R I M G I N G
Output: A phone is ringing.

{raw_text}'''

    response = gemini.generate_content(prompt)
    polished_sentence = response.text.strip()
    return polished_sentence


# ----- Webcam Mode -----
def start_webcam():
    def webcam_thread():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam")
            return

        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=2)
        mpDraw = mp.solutions.drawing_utils

        last_prediction = None
        accumulated_text = ""  # To store all gestures continuously

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gesture, processed_frame = recognize_gesture_from_frame(frame, hands, mpDraw)

            # Add gesture only if it’s new and valid
            if gesture != last_prediction and gesture not in ["Unknown", "No Hand Detected"]:
                accumulated_text += gesture + " "
                last_prediction = gesture

            # Update displayed text with all gestures so far
            result_text.set(f"It says - {accumulated_text.strip()}")
            cv2.putText(processed_frame, gesture, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("WebCam Gesture Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Speak out all gestures together at the end
        if accumulated_text:
            # Get the polished sentence from AI
            polished_sentence = polish_text_with_ai(accumulated_text.strip())

            # Update Tkinter window with polished sentence
            result_text.set(f"It says: {polished_sentence}")

            # Optionally, speak the polished sentence
            speak(polished_sentence)
        else:
            result_text.set("No gestures detected")

    threading.Thread(target=webcam_thread).start()


# ----- Upload Mode -----
def upload_file():
    file_path = filedialog.askopenfilename(filetypes=[("Media", "*.png;*.jpg;*.jpeg;*.mp4")])
    if not file_path:
        return

    ext = file_path.split('.')[-1].lower()
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1)
    mpDraw = mp.solutions.drawing_utils

    last_prediction = None

    if ext in ["png", "jpg", "jpeg"]:
        img = cv2.imread(file_path)
        gesture, processed_frame = recognize_gesture_from_frame(img, hands, mpDraw)

        cv2.putText(processed_frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image Prediction", processed_frame)

        # Polishing single gesture using Gemini
        if gesture not in ["Unknown", "No Hand Detected"]:
            polished_sentence = polish_text_with_ai(gesture)
            result_text.set(f"AI says: {polished_sentence}")
            speak(polished_sentence)
        else:
            result_text.set(f"It says - {gesture}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif ext == "mp4":
        cap = cv2.VideoCapture(file_path)
        accumulated_text = ""  # To accumulate gestures

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gesture, processed_frame = recognize_gesture_from_frame(frame, hands, mpDraw)

            if gesture != last_prediction and gesture not in ["Unknown", "No Hand Detected"]:
                accumulated_text += gesture + " "
                last_prediction = gesture

            result_text.set(f"It says - {gesture}")
            cv2.putText(processed_frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Prediction", processed_frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        if accumulated_text.strip():
            polished_sentence = polish_text_with_ai(accumulated_text.strip())
            result_text.set(f"AI says: {polished_sentence}")
            speak(polished_sentence)
        else:
            result_text.set("No valid gestures detected")

    else:
        messagebox.showerror("Unsupported", "Only .jpg, .png, .jpeg, and .mp4 are supported.")


# ----- GUI Setup -----
root = tk.Tk()
root.title("Sign Language To Text & Speech")
root.geometry("600x400")
root.configure(bg="#f0f4f7")

# ----- Style -----
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)

# ----- Heading -----
tk.Label(root, text="Sign Language To Text & Speech",
         font=("Helvetica", 20, "bold"), bg="#f0f4f7", fg="#333333").pack(pady=20)

# ----- Buttons -----
button_frame = tk.Frame(root, bg="#f0f4f7")
button_frame.pack(pady=10)

upload_btn = ttk.Button(button_frame, text="Upload Image or Video", command=upload_file)
upload_btn.grid(row=0, column=0, padx=20)

webcam_btn = ttk.Button(button_frame, text="Record using WebCam", command=start_webcam)
webcam_btn.grid(row=0, column=1, padx=20)

# ----- Result Display -----
tk.Label(root, text="It says -", font=("Helvetica", 14, "bold"), bg="#f0f4f7", fg="#555555").pack(pady=(30, 5))

result_text = tk.StringVar()
result_text.set("Waiting for sign input...")
result_display = tk.Label(root, textvariable=result_text,
                          font=("Helvetica", 32, "bold"), fg="#222222",
                          bg="#e6f0ff", wraplength=500, justify="center",
                          relief="ridge", bd=5, padx=10, pady=10)
result_display.pack(pady=20)

# ----- Mainloop -----
root.mainloop()
