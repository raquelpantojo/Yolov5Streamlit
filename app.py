import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os

# Carregue o modelo YOLOv5 'finger.pt' localmente
model = torch.hub.load('ultralytics/yolov5', 'custom', path='finger.pt', force_reload=True)

st.title("Detecção da Ponta do Dedo em Vídeos")

# Função para realizar a detecção em um frame
def detect_finger(image):
    results = model(image)
    return results

# Upload de um vídeo
video_file = st.file_uploader("Carregar um vídeo", type=['mp4', 'mpeg', 'mov'])

if video_file is not None:
    # Abra o vídeo
    video_capture = cv2.VideoCapture(video_file)

    # Crie um temporizador para lidar com o limite de execução
    t = 0

    # Resto do código para processar o vídeo

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Realize a detecção no frame
        results = detect_finger(frame)
        detected_frame = results.render()[0]

        # Exiba o frame com a detecção
        st.image(detected_frame, caption="Resultado da Detecção", use_column_width=True)

        # Limite de execução: pare após algum tempo (por exemplo, 120 segundos)
        t += 1
        if t >= 120:  # 120 segundos
            break