import streamlit as st
import torch
import cv2
import numpy as np
import tempfile
import os

# Carregue o modelo YOLOv5 'finger.pt' localmente
model = torch.hub.load('ultralytics/yolov5', 'custom', path='amarelo.pt', force_reload=True)

st.title("Detecção da Ponta do Dedo em Vídeos")

# Função para realizar a detecção em um frame
def detect_finger(image):
    model.conf = 0.70  # Defina o threshold de confiança desejado
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

    # Inicialize variáveis
    detections_found = 0  # Quantas detecções encontradas
    target_detections = 3  # Quantidade de detecções desejadas

    # Abra o vídeo de saída para salvar as detecções
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width, frame_height))

    # Loop para processar cada frame do vídeo
    while detections_found < target_detections:
        ret, frame = video_capture.read()
        if not ret:
            break

        # Realize a detecção no frame
        results = detect_finger(frame)
        detected_frame = results.render()[0]

        # Se uma detecção foi encontrada, exiba o frame
        if len(results.xyxy[0]) > 0:
            detection = results.xyxy[0][0]  # Pegue a primeira detecção
            xmin, ymin, xmax, ymax = detection[0:4]  # Valores x, y, largura (w) e altura (h)
            
            x1, y1, x2, y2 = map(int, detection[0:4])  
            roi = frame[y1:y2, x1:x2]
            
            st.image(roi,channels ="BGR")
            st.image(detected_frame, caption=f"Detecção {detections_found + 1}", use_column_width=True,channels ="BGR")
            
            #st.write(f"x: {x}, y: {y}, largura (w): {w}, altura (h): {h}")
            
            # Converte para números inteiros
            #x1 = int(x - w / 2)
            #y1 = int(y - h / 2)
            #x2 = int(x + w / 2)
            #y2 = int(y + h / 2)
            
            st.write(f"YOLO xmin: {xmin}, ymin: {ymin}, xmax: {xmax}, ymax: {ymax}")
            st.write(f"OpenCV x: {x1}, y: {y1}, x2: {x2}, y2: {y2}")
            
    
            detections_found += 1

        # Escreva o frame no vídeo de saída
        out.write(detected_frame)

    # Fecha o vídeo de saída
    out.release()


    # Certifique-se de apagar o arquivo temporário após o uso
    os.remove(temp_filename)
