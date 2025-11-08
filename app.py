import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils.emotion_labels import EMOTIONS
import os
import time

MODEL_PATH = os.path.join("models", "fer2013_mini_XCEPTION.110-0.65.hdf5")
CASCADE_PATH = os.path.join("haarcascades", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
emotion_model = load_model(MODEL_PATH, compile=False)

st.set_page_config(page_title="Live Emotion Detection", layout="wide")
st.title("🎭 Live Facial Emotion Detection By Aniket Chandra(2328225) and Sudhansu Mohapatra(2328051)")
st.write("Real-time emotion detection using your webcam. Press 'Stop' to exit.")


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

FRAME_WINDOW = st.image([])
stop = st.button("Stop")

frame_count = 0


while cap.isOpened():
    if stop:
        break

    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to grab frame from webcam.")
        break

    frame_count += 1
    if frame_count % 2 != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (64, 64))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)
        face = np.expand_dims(face, axis=-1)

        preds = emotion_model.predict(face, verbose=0)[0]
        emotion = EMOTIONS[np.argmax(preds)]
        confidence = np.max(preds)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        text = f"{emotion} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)


    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    time.sleep(0.03)

cap.release()
st.write("Webcam stopped.")
