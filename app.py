import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import speech_recognition as sr
from collections import defaultdict
import threading
import requests
from gtts import gTTS
import pygame
from tempfile import NamedTemporaryFile

app = Flask(__name__)
model = load_model('emotion_detection_model.h5')
emotion_labels = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Shared emotion list to capture emotions during audio recording
shared_emotions_list = []
capture_emotions = False

LLM_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.groq.com/openai/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "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"
LLM_CHAT_MODEL = os.getenv("LLM_CHAT_MODEL", "llama-3.1-8b-instant")
LLM_STREAMING = os.getenv("LLM_STREAMING", "no") != "no"

# Initialize pygame properly
pygame.init()

def chat(messages, handler=None):
    url = f"{LLM_API_BASE_URL}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    model = LLM_CHAT_MODEL
    max_tokens = 400
    temperature = 0
    stream = LLM_STREAMING and callable(handler)
    data = {"messages": messages, "model": model, "max_tokens": max_tokens, "temperature": temperature}
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    
    if not stream:
        data = response.json()
        choices = data["choices"]
        first = choices[0]
        message = first["message"]
        content = message["content"]
        answer = content.strip()
        if handler:
            handler(answer)
        return answer
    
    return None

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Silakan bicara...")
        recognizer.adjust_for_ambient_noise(source)
        
        global capture_emotions
        capture_emotions = True  # Start capturing emotions
        audio_thread = threading.Thread(target=capture_emotions_while_recording, args=(shared_emotions_list,))
        audio_thread.start()

        audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        capture_emotions = False  # Stop capturing emotions
        audio_thread.join()

        emotion_counts = defaultdict(int)
        for emotion in shared_emotions_list:
            emotion_counts[emotion] += 1

    try:
        text = recognizer.recognize_google(audio, language='id-ID')
        print(f"Recognized Speech: {text}")
        if emotion_counts:
            dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        else:
            dominant_emotion = None
        return text, emotion_counts, dominant_emotion
    except sr.UnknownValueError:
        print("Google Speech Recognition tidak mengerti audio")
        return "", {}, ""
    except sr.RequestError as e:
        print(f"Request kepada Google Speech Recognition gagal; {e}")
        return "", {}, ""

def detect_emotion(img):
    img = cv2.resize(img, (48, 48))
    img = img.reshape(1, 48, 48, 1) / 255.0
    prediction = model.predict(img)
    emotion = emotion_labels[np.argmax(prediction)]
    return emotion

def gen_frames(emotions_list=None):
    cap = cv2.VideoCapture(0)
    
    while capture_emotions:  # Capture emotions while audio is being recorded
        success, frame = cap.read()
        if not success:
            print("Failed to capture frame")
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                emotion = detect_emotion(face)
                
                if emotions_list is not None:
                    emotions_list.append(emotion)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def capture_emotions_while_recording(emotions_list):
    for _ in gen_frames(emotions_list=emotions_list):
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # Memulai pengenalan suara dan deteksi emosi
        text, emotion_counts, dominant_emotion = recognize_speech()

        if text:
            # Memproses teks yang dikenali dengan Groq API
            messages = [
                {"role": "system", "content": "You are a supportive and empathetic virtual psychologist. Your goal is to understand and help the user based on their emotional state but you just response in 1 paragraph and be more human."},
                {"role": "system", "content": f"The user seems to be feeling mostly {dominant_emotion}."},
                {"role": "user", "content": text}
            ]
            groq_response = chat(messages)

            # Mengonversi respons menjadi suara
            with NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
                temp_audio_name = temp_audio.name
                tts = gTTS(groq_response, lang='id')
                tts.save(temp_audio_name)

            # Memainkan audio respons
            pygame.mixer.music.load(temp_audio_name)
            pygame.mixer.music.play()

            # Event loop untuk menangani penghapusan file setelah selesai diputar
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Hapus file setelah selesai diputar
            try:
                os.remove(temp_audio_name)
            except Exception as e:
                print(f"Failed to delete temp audio file: {e}")

            # Reset capture_emotions dan shared_emotions_list
            global shared_emotions_list
            shared_emotions_list = []
            capture_emotions = False

            return jsonify({
                "request": text,
                "response": groq_response,
                "emotion_counts": dict(emotion_counts),
                "dominant_emotion": dominant_emotion
            }), 200
        else:
            return jsonify({"error": "Google Speech Recognition tidak dapat mengenali audio"}), 400
    except sr.UnknownValueError:
        print("Google Speech Recognition tidak mengerti audio")
        return jsonify({"error": "Google Speech Recognition tidak dapat mengenali audio"}), 400
    except sr.RequestError as e:
        print(f"Request kepada Google Speech Recognition gagal; {e}")
        return jsonify({"error": "Request kepada Google Speech Recognition gagal", "message": str(e)}), 500
    except Exception as e:
        print(f"Unhandled exception: {e}")
        return jsonify({"error": "Internal Server Error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
