import cv2
import numpy as np
import base64

def carregar_imagem_bytes(conteudo_bytes: bytes):
    """
    Recebe os bytes puros do arquivo de imagem enviado pelo front-end
    e os converte para uma matriz legível pelo opencv
    """
    # transforma os bytes puros em um array unidimensional do numpy do tipo inteiro de 8 bits
    nparr = np.frombuffer(conteudo_bytes, np.uint8)
    
    # decodifica esse array em uma imagem opencv
    # cv2.IMREAD_UNCHANGED garante que a imagem seja carregada 
    # exatamente como ela é (mantendo 1 canal se for cinza, ou 3 se for colorida)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    return img

def verificar_escala_cinza(imagem):
    """
    Verifica matematicamente se a imagem possui apenas 1 canal 
    (8 bits em escala de cinza) para habilitar/desabilitar processos
    """
    # .shape retorna (altura, largura) para imagens em tons de cinza ou (altura, largura, canais) para imagens coloridas
    # Se tiver só 2 dimensões ou se a 3ª dimensão for 1, é escala de cinza
    return len(imagem.shape) == 2 or (len(imagem.shape) == 3 and imagem.shape[2] == 1)

def converter_para_base64(imagem):
    """
    Converte a matriz do OpenCV de volta para o formato de arquivo .png e, 
    em seguida, codifica em uma string Base64 para ser enviada via json
    """
    # é exigido especificamente o formato .png 
    # imencode converte a matriz na memória para o formato de arquivo especificado
    sucesso, buffer = cv2.imencode('.png', imagem) 
    
    if not sucesso:
        raise ValueError("Falha ao codificar a imagem para .png")
    
    # codifica os bytes do arquivo PNG gerado para uma string texto Base64
    imagem_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # adiciona o prefixo Data URI para que o front-end consiga colocar direto no atributo 'src' de uma tag <img> no HTML
    return f"data:image/png;base64,{imagem_base64}"