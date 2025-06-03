import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import cv2
import threading
import pyttsx3
import mediapipe as mp

# ----- Text-to-Speech -----
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 0.9)
    engine.say(text)
    engine.runAndWait()

# ----- Gesture Mapping -----
def get_gesture(fingers):
    if fingers == [0,0,0,0,0]:
        return "Fist"
    elif fingers == [1,1,1,1,1]:
        return "Open Hand"
    elif fingers == [0,1,1,0,0]:
        return "Peace"
    elif fingers == [1,1,0,0,1]:
        return "Rock"
    elif fingers == [1,0,0,0,0]:
        return "Thumbs up"
    elif fingers == [1,0,1,1,1]:
        return "Nice"
    elif fingers == [0,1,0,0,0]:
        return "Index Finger"
    elif fingers == [1,0,0,0,1]:
        return "Call Me"
    elif fingers == [1,1,0,0,0]:
        return "L - Sign"
    elif fingers == [0,1,1,1,1]:
        return "4 Fingers Up"
    else:
        return "Unknown"

# ----- Gesture Recognition Function -----
def recognize_gesture_from_frame(frame, hands, tipIds, mpDraw):
    h, w, _ = frame.shape
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

    fingers = []
    if len(lmList) != 0:
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return get_gesture(fingers), frame
    return "No Hand Detected", frame

# ----- Webcam Mode -----
def start_webcam():
    def webcam_thread():
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot access webcam")
            return

        mpHands = mp.solutions.hands
        hands = mpHands.Hands(max_num_hands=1)
        mpDraw = mp.solutions.drawing_utils
        tipIds = [4, 8, 12, 16, 20]

        last_prediction = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            gesture, processed_frame = recognize_gesture_from_frame(frame, hands, tipIds, mpDraw)

            if gesture != last_prediction and gesture not in ["Unknown", "No Hand Detected"]:
                speak(gesture)
                last_prediction = gesture

            result_text.set(f"It says - {gesture}")
            cv2.putText(processed_frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("WebCam Gesture Recognition", processed_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
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
    tipIds = [4, 8, 12, 16, 20]

    last_prediction = None

    if ext in ["png", "jpg", "jpeg"]:
        img = cv2.imread(file_path)
        gesture, output_img = recognize_gesture_from_frame(img, hands, tipIds, mpDraw)
        cv2.putText(output_img, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Image Prediction", output_img)
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

            gesture, output_img = recognize_gesture_from_frame(frame, hands, tipIds, mpDraw)
            if gesture != last_prediction and gesture not in ["Unknown", "No Hand Detected"]:
                speak(gesture)
                last_prediction = gesture

            result_text.set(f"It says - {gesture}")
            cv2.putText(output_img, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video Prediction", output_img)

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
