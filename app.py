from ultralytics import YOLO
import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

model = YOLO("yolo11n.pt")

uploaded_file = st.file_uploader("Choose an image or video", 
                                 type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None and uploaded_file.type.startswith('image'):
    # Process image
    image = np.array(Image.open(uploaded_file))
    results = model(image)
    annotated_image = results[0].plot()
    st.image(annotated_image, caption="Processed Image", use_column_width=True)

elif uploaded_file is not None and uploaded_file.type.startswith('video'):
    # Process video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")
    cap.release()

elif uploaded_file is None:
    st.info("Gunakan Webcam untuk mendeteksi objek")
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame, channels="BGR")
    cap.release()