# Sign Language Interpreter Project

## Overview

This project translates sign language gestures into text using computer vision and machine learning. It uses **MediaPipe** for hand detection, a **K-Nearest Neighbors (KNN)** classifier trained on hand gesture images for letter recognition, and **Gemini** to convert detected letters into meaningful sentences. The output is shown in a **Tkinter GUI** and read aloud with text-to-speech.

Supports two input modes:  
- Webcam live feed  
- File upload (image/video)

---

## Technologies and Libraries

| Technology          | Purpose                         |
|---------------------|--------------------------------|
| Python              | Programming language            |
| OpenCV              | Image and video processing      |
| MediaPipe           | Hand detection                  |
| scikit-learn (KNN)  | Gesture classification          |
| Tkinter             | GUI                            |
| Gemini API          | Sentence structuring            |
| pyttsx3             | Text-to-speech output          |

---

## Dataset Structure

Organize dataset folders by class label (A-Z), each containing corresponding hand images resized to 64×64 pixels.
Taken from Kaggle


---

## Workflow

1. User selects input mode: webcam or upload.
2. Frames captured from input source.
3. MediaPipe detects hands and draws landmarks.
4. Frame is resized and preprocessed for KNN.
5. KNN predicts letter from frame.
6. Predictions buffered over 15 frames for stability.
7. Predicted set of letters sent to Gemini for sentence formation.
8. Result displayed on Tkinter GUI and read aloud.

---

## Function Descriptions

### `def recognize_gesture_from_frame(frame, hands, mpDraw):`

- Detects hand landmarks on `frame`.
- Draws landmarks.
- Preprocesses frame (resize, normalize, flatten).
- Uses KNN to predict the gesture.
- Returns predicted letter and annotated frame.

### `def polish_text_with_ai(raw_text):`

- Sends raw letter sequence to Gemini API.
- Returns natural language sentence.

### `speak(sentence)`

- Uses text-to-speech to vocalize the sentence.

### `def start_webcam():`

- Captures live video frames.
- Runs prediction pipeline.
- Updates GUI continuously.

### `upload_file()`

- Allows file selection for image/video.
- Processes frames as in webcam mode.
- Displays final structured sentence.

---
# Future Improvements
- Replace KNN with CNN for higher accuracy.
- Use hand cropping to focus predictions.
- Expand vocabulary with numbers and special gestures.
- Enhance GUI with editing and correction options.
- Support multiple languages with Gemini.

# Summary
A practical sign language recognition system integrating MediaPipe hand tracking,
KNN classification, and natural language generation for accessible communication.



