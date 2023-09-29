import streamlit as st
import torch
import cv2
import numpy as np

# Carregue o modelo YOLOv5 "finger.pt"
model_path="https://github.com/raquelpantojo/Yolov5Streamlit/tree/main/models/finger.pt"
model = torch.hub.load(model_path)

st.title("Detecção da Ponta do Dedo em Vídeos")

# Função para realizar a detecção em um frame
def detect_finger(image):
    results = model(image)
    return results

# Upload de um vídeo
video_file = st.file_uploader("Carregar um vídeo", type=['mp4', 'mpeg', 'mov'])

if video_file is not None:
    # Abre o vídeo
    video_capture = cv2.VideoCapture(video_file)
    
    # Configura a captura de vídeo
    frame_width = int(video_capture.get(3))
    frame_height = int(video_capture.get(4))
    out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    
    st.write("Vídeo de Entrada:")
    
    # Loop para processar cada frame do vídeo
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        # Realiza a detecção no frame
        results = detect_finger(frame)
        detected_frame = results.render()[0]
        
        # Exibe o frame com a detecção
        st.image(detected_frame, caption="Resultado da Detecção", use_column_width=True)
        
        # Escreve o frame no vídeo de saída
        out.write(detected_frame)
    
    # Fecha o vídeo de saída
    out.release()

    st.write("Vídeo de Saída com a Detecção:")
    st.video('output.avi')

    # Libera o vídeo
    video_capture.release()
    cv2.destroyAllWindows()
