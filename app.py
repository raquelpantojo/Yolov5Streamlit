import streamlit as st
import torch
import cv2
import numpy as np
import tempfile

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
    # Salve o vídeo em um arquivo temporário
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_filename = temp_file.name
        temp_file.write(video_file.read())

    # Abra o vídeo com o caminho do arquivo temporário
    video_capture = cv2.VideoCapture(temp_filename)

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

# Certifique-se de apagar o arquivo temporário após o uso
os.remove(temp_filename)
