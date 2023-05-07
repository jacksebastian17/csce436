from flask import Flask, render_template, Response
import cv2
import threading
import time
from gtts import gTTS
import os
import pygame
import tempfile

app = Flask(__name__)

def speak(text):
    tts = gTTS(text=text, lang="en")
    
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        temp_filename = f.name
    
    tts.save(temp_filename)

    pygame.mixer.init()
    pygame.mixer.music.load(temp_filename)
    # pygame.mixer.music.play()

    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    os.remove(temp_filename)

# Wrapper function for the speech synthesis thread
def speak_thread(text):
    threading.Thread(target=speak, args=(text,)).start()

net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/webcam')
def index():
    return render_template('index.html')

def gen_video_stream():
    last_spoken_time = time.time()
    detected_class_names = set()
    counter = 0

    while True:
        ret, frame = cap.read()

        (class_ids, scores, bboxes) = model.detect(frame, confThreshold=0.3, nmsThreshold=0.4)
        for class_id, score, bbox in zip(class_ids, scores, bboxes):
            (x, y, w, h) = bbox
            class_name = classes[class_id]

            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
            
            # Add detected class name to the set
            detected_class_names.add(class_name)

        # Speak the detected object names every 5 seconds
        if time.time() - last_spoken_time >= 5 and detected_class_names:
            for class_name in detected_class_names:
                speak_thread(class_name)
            last_spoken_time = time.time()
            detected_class_names.clear()

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_video_stream(),
                    content_type='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)