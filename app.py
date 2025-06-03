import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import threading
import pyttsx3
import mediapipe as mp
import joblib
import numpy as np

# -----Load trained mode -----
model = joblib.load('sign_model.pkl')  


# ----- Text-to-Speech -----
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()

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
            prediction = model.predict([landmark_points])[0]

            return prediction, frame

    return "No Hand Detected", frame

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
            speak(accumulated_text.strip())

        result_text.set("Webcam stopped")

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
        speak(gesture)
        result_text.set(f"It says - {gesture}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif ext == "mp4":
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gesture, processed_frame = recognize_gesture_from_frame(frame, hands, mpDraw)

            if gesture != last_prediction and gesture not in ["Unknown", "No Hand Detected"]:
                speak(gesture)
                last_prediction = gesture

            result_text.set(f"It says - {gesture}")
            cv2.putText(processed_frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Prediction", processed_frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
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
