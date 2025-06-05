# 🤟 Sign Language Interpreter  
Sign Language Interpreter is an AI-powered system that translates hand gestures into natural language text and speech. Using MediaPipe for hand detection, KNN for gesture classification, and Gemini AI for structuring sentences, it provides a user-friendly interface with Tkinter. The app supports webcam and file upload inputs, making sign language communication more accessible and interactive.

<br>

---

## Example Interface
<img src="https://github.com/Eshita-Badhe/Sign-Language-Interpreter/blob/main/images/GUI.png" alt="Sign Language Interpreter GUI" style="width:75%;height:auto;">

## Features
<ul>
    <li><strong>Real-time Gesture Detection:</strong> Uses MediaPipe to detect hand landmarks live from webcam or uploaded videos/images.</li>
    <li><strong>Accurate Gesture Classification:</strong> KNN model trained on hand gesture images to recognize alphabets (A-Z).</li>
    <li><strong>Prediction Buffering:</strong> Stabilizes predictions using a 15-frame buffer to avoid jittery results.</li>
    <li><strong>Natural Language Structuring:</strong> Sends recognized letters to Gemini AI to generate meaningful sentences.</li>
    <li><strong>GUI Display & Text-to-Speech:</strong> Shows recognized sentences on Tkinter window and reads them aloud.</li>
    <li><strong>Multiple Input Modes:</strong> Supports both webcam live feed and uploaded media files.</li>
</ul>

## Technologies Used
<ul>
    <li><strong>Frontend:</strong> Tkinter (Python GUI)</li>
    <li><strong>Backend:</strong> Python, OpenCV</li>
    <li><strong>Hand Detection:</strong> MediaPipe</li>
    <li><strong>Gesture Classification:</strong> scikit-learn (KNN)</li>
    <li><strong>Language Processing:</strong> Gemini AI</li>
    <li><strong>Text-to-Speech:</strong> pyttsx3</li>
</ul>

## Installation

### Prerequisites
<ul>
    <li>Python 3.8 to 3.10(any) installed</li>
    <li>Basic command line knowledge</li>
</ul>

### Steps
<ol>
    <li>Clone the repository:
        <pre><code>git clone https://github.com/Eshita-Badhe/Sign-Language-Interpreter.git</code></pre>
    </li>
    <li>Navigate to the project directory:
        <pre><code>cd Sign-Language-Interpreter</code></pre>
    </li>
    <li>Install required packages:
        <pre><code>pip install -r requirements.txt</code></pre>
    </li>
    <li>Run the application:
        <pre><code>python app.py</code></pre>
    </li>
</ol>

## Usage
<ol>
    <li>Launch the app.</li>
    <li>Select input mode: Webcam or Upload a media file.</li>
    <li>Start making hand gestures representing alphabets and click 'q' when completed.</li>
    <li>Wait as the app detects, predicts letters, buffers results, and forms sentences.</li>
    <li>View the recognized sentence on the GUI and listen to the audio output.</li>
</ol>

## Example 
<img src="https://github.com/Eshita-Badhe/Sign-Language-Interpreter/blob/main/images/example.png" alt="Sign Language Interpreter Example" style="width:80%;height:auto;">

## Contributing
<p>Contributions are welcome! Feel free to fork the repo, create branches for your features, and submit pull requests with clear explanations.</p>

## Future Improvements
<ul>
    <li>Replace KNN with CNN or other deep learning models for better accuracy.</li>
    <li>Implement hand cropping and preprocessing to improve gesture focus.</li>
    <li>Expand vocabulary to include numbers, special signs, and phrases.</li>
    <li>Add GUI features like sentence editing and correction.</li>
    <li>Enable multilingual support using Gemini AI capabilities.</li>
</ul>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
<p>For feedback or collaboration, reach out to:</p>
<ul>
    <li><strong>Name:</strong> Eshita Badhe</li>
    <li><strong>Email:</strong> sge.eshita31gb@gmail.com</li>
    <li><strong>GitHub:</strong> <a href="https://github.com/Eshita-Badhe">https://github.com/Eshita-Badhe</a></li>
</ul>
