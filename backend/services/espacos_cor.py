import cv2
from utils.validador_imagem import verificar_escala_cinza

def remover_canal_alpha(imagem):
    # Se a imagem tiver 4 canais (BGRA), convertemos para 3 canais (BGR)
    if len(imagem.shape) == 3 and imagem.shape[2] == 4:
        return cv2.cvtColor(imagem, cv2.COLOR_BGRA2BGR)
    return imagem

def decompor_rgb(imagem):
    if verificar_escala_cinza(imagem):
        raise ValueError("A imagem já está em escala de cinza, carregue uma imagem colorida")
    
    # Remove a transparência (se existir) antes de separar
    imagem_bgr = remover_canal_alpha(imagem)
    
    # Agora temos certeza que só existem 3 canais
    b, g, r = cv2.split(imagem_bgr)
    return r, g, b

def decompor_hsv(imagem):
    if verificar_escala_cinza(imagem):
        raise ValueError("A imagem já está em escala de cinza, carregue uma imagem colorida")
    
    # Remove a transparência para não atrapalhar a conversão do HSV
    imagem_bgr = remover_canal_alpha(imagem)
    
    # Converte de BGR para HSV
    hsv = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2HSV)
    
    h, s, v = cv2.split(hsv)
    return h, s, v