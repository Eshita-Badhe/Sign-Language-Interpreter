import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import threading
import numpy as np
import joblib
import pyttsx3

# # ----- Load KNN Model -----
model = joblib.load("knn_sign_model.joblib")

# ----- Predict Function -----
def predict_sign(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flatten = img.flatten().reshape(1, -1)
    prediction = model.predict(img_flatten)
    return prediction[0]

# ----- Upload Handler -----
def upload_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:
        try:
            prediction = predict_sign(file_path)
            result_text.set(f"It says - {prediction}")
            print(f"Predicted: {prediction}")
            speak(prediction) 
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image.\n{e}")

# --- Web Cam Handler ---
def start_webcam():
    def webcam_thread():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam")
            return

        # Set webcam properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        last_prediction = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame for prediction (resize, grayscale, flatten)
            img = cv2.resize(frame, (64, 64))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_flatten = gray.flatten().reshape(1, -1)

            try:
                prediction = model.predict(img_flatten)[0]
            except Exception as e:
                prediction = "Error"

            # Only speak if prediction changes
            if prediction != last_prediction and prediction != "Error":
                speak(prediction)
                last_prediction = prediction

            # Show prediction text on the frame
            cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("WebCam - Press 'q' to Quit", frame)

            # Update the GUI label safely using `result_text.set()`
            result_text.set(f"It says - {prediction}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        result_text.set("Webcam stopped")

    threading.Thread(target=webcam_thread).start()

# --- Text-to-Speech Function ---
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)  # speech speed
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()

# ----- GUI Setup -----
root = tk.Tk()
root.title("Sign Language To Text & Speech")
root.geometry("600x400")
root.configure(bg="#f0f4f7")

# ----- Style -----
style = ttk.Style()
style.configure("TButton",
                font=("Helvetica", 12),
                padding=10,
                background="#007acc",
                foreground="black")

# ----- Heading -----
title_label = tk.Label(root, text="Sign Language To Text & Speech", 
                      font=("Helvetica", 20, "bold"), 
                      bg="#f0f4f7", fg="#333333")
title_label.pack(pady=20)

# ----- Buttons -----
button_frame = tk.Frame(root, bg="#f0f4f7")
button_frame.pack(pady=10)

upload_btn = ttk.Button(button_frame, text="Upload Image or Video", command=upload_file)
upload_btn.grid(row=0, column=0, padx=20)

webcam_btn = ttk.Button(button_frame, text="Record using WebCam", command=start_webcam)
webcam_btn.grid(row=0, column=1, padx=20)

# ----- Result Display Box -----
result_label = tk.Label(root, text="It says -", font=("Helvetica", 14, "bold"), bg="#f0f4f7", fg="#555555")
result_label.pack(pady=(30, 5))

result_text = tk.StringVar()
result_text.set("Waiting for sign input...")
result_display = tk.Label(root, textvariable=result_text, 
                          font=("Helvetica", 32, "bold"),
                          fg="#222222", bg="#e6f0ff",
                          wraplength=500, justify="center",
                          relief="ridge", bd=5, padx=10, pady=10)
result_display.pack(pady=20)

# ----- Mainloop -----
root.mainloop()

