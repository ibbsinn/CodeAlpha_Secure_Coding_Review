from flask import Flask, render_template, Response
import cv2
import numpy as np
import os
import csv
import time
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    # Your code for face recognition and object detection here
    thres = 0.45  # Threshold to detect object
    classNames = []
    classFile = 'coco.names'
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip('\n').split('\n')

    configPath = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('Data/mymodel.xml')
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    name_list = ["", "Komail", "KomaiL", "komail", "Hassaan", "hassaan", "HaSsaan", "Ibrahim", "ibrahim", "IbrahIm"]
    confidence_threshold = 85
    COL_NAMES = ['NAME', 'TIME']

    start_time = time.time()
    attendance_interval = 5  # Take attendance every 5 seconds

    while True:
        success, frame = video.read()
        if not success:
            break

        classIds, confs, bbox = net.detect(frame, confThreshold=thres)
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(frame, box, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < confidence_threshold:
                name = name_list[serial]
                verification_result = "Verified"
            else:
                name = "Unknown"
                verification_result = "Not Verified"

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)

            cv2.putText(frame, f"{name} ({verification_result})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            current_time = time.time()
            if current_time - start_time >= attendance_interval:
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                attendance = [name, timestamp]
                file_path = f"Attendance/Attendance_{date}.csv"
                exists = os.path.isfile(file_path)
                with open(file_path, "a") as csvfile:
                    writer = csv.writer(csvfile)
                    if not exists:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)
                start_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
