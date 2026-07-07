import cv2
from utils.validador_imagem import verificar_escala_cinza

def decompor_rgb(imagem):
    # decomposição só faz sentido se a imagem tiver cores
    if verificar_escala_cinza(imagem):
        raise ValueError("A imagem já está em escala de cinza, carregue uma imagem colorida")
    
    # o Opencv lê imagens coloridas no formato BGR (Azul, Verde, Vermelho) 
    # o cv2.split separa a imagem em 3 matrizes distintas (uma para cada canal)
    b, g, r = cv2.split(imagem)
    
    # Retornamos na ordem tradicional r g b
    return r, g, b

def decompor_hsv(imagem):
    if verificar_escala_cinza(imagem):
        raise ValueError("A imagem já está em escala de cinza, carregue uma imagem colorida")
    
    # converte o espaço de cor BGR nativo do OpenCV para HSV (Matiz, Saturação, Valor)
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    
    # separa os 3 canais HSV em 3 imagens distintas em tons de cinza
    h, s, v = cv2.split(hsv)
    return h, s, v