import cv2
import numpy as np
from utils.validador_imagem import verificar_escala_cinza

def limiarizacao(imagem, k: int):
    """Aplica uma limiarização binária baseada no valor k"""
    # garante que a imagem está em escala de cinza antes de tranformar em binário
    if not verificar_escala_cinza(imagem):
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    
    # cv2.threshold compara cada pixel com k
    # Se o pixel for maior que k, ele vira 255. Se for menor, vira 0
    _, img_limiarizada = cv2.threshold(imagem, k, 255, cv2.THRESH_BINARY)
    return img_limiarizada

def transformacao_logaritmica(imagem, c: float):
    """Aplica a fórmula s = c * log(1 + r), expande tons escuros e comprime tons claros"""
    # precisamos converter a imagem para float32. Se fizermos logaritmo em inteiros de 8 bits (0-255) perderemos as casas decimais e o cálculo ficará totalmente impreciso
    img_float = np.float32(imagem)
    
    # np.log1p calcula log(1 + x) de forma segura, evitando erro caso algum pixel seja 0
    img_log = c * (np.log1p(img_float))
    
    # o resultado não estará mais na escala 0-255
    # usamos cv2.normalize para "esticar" os valores resultantes de volta para o intervalo [0, 255]
    img_log = cv2.normalize(img_log, None, 0, 255, cv2.NORM_MINMAX)
    
    # converte de volta para inteiros de 8 bits sem sinal 
    return np.uint8(img_log)

def transformacao_logaritmica(imagem, c: float):
    """Aplica a fórmula s = c * log(1 + r), expande tons escuros e comprime tons claros"""
    # precisamos converter a imagem para float32. Se fizermos logaritmo em inteiros de 8 bits (0-255) perderemos as casas decimais e o cálculo ficará totalmente impreciso
    img_float = np.float32(imagem)
    
    # np.log1p calcula log(1 + x) de forma segura, evitando erro caso algum pixel seja 0
    img_log = c * (np.log1p(img_float))
    
    # ATENÇÃO: antes usávamos cv2.normalize(..., NORM_MINMAX) aqui, mas isso sempre
    # "esticava" o resultado para ocupar 0-255 por completo, o que anulava totalmente
    # o efeito do parâmetro 'c' (mudar c não alterava a imagem final).
    # Agora usamos np.clip para respeitar de fato o valor calculado por 's = c * log(1+r)',
    # apenas cortando o que ultrapassar os limites válidos da imagem (0 a 255).
    img_log = np.clip(img_log, 0, 255)
    
    # converte de volta para inteiros de 8 bits sem sinal 
    return np.uint8(img_log)
 
def equalizacao_histograma(imagem):
    """Redistribui as intensidades dos pixels para melhorar o contraste geral"""
    # essa operação só é habilitada para imagens em escala de cinza
    if not verificar_escala_cinza(imagem):
        raise ValueError("Equalização de histograma só está habilitada para imagens em escala de cinza")
    
    # função do OpenCV que equaliza a imagem, ela achata o histograma e aumenta o contraste
    img_eq = cv2.equalizeHist(imagem)
    return img_eq

def fatiamento_intensidade(imagem, A: int, B: int, preservar_fundo: bool):
    """Destaca uma faixa específica de pixels entre A e B"""
    # cria uma cópia independente da imagem para não alterar a original sem querer
    resultado = np.copy(imagem)
    
    # cria uma máscara booleana (uma matriz de Verdadeiros e Falsos)
    # verdadeiro onde o pixel estiver entre A e B; Falso onde estiver fora
    mascara = (imagem >= A) & (imagem <= B)
    
    if not preservar_fundo:
        # se não for para preservar o fundo, tudo que for Falso na máscara vira 0 
        resultado[~mascara] = 0
        # o que estava dentro da faixa desejada (Verdadeiro na máscara) ganha brilho máximo (255)
        resultado[mascara] = 255
    else:
        # se for para preservar o fundo, deixamos os falsos quietos e alteramos apenas o destaque
        resultado[mascara] = 255
        
    return resultado

def adicionar_ruido_gaussiano(img, media=0, sigma=20):
    # Gera ruído com base numa distribuição normal (gaussiana)
    ruido = np.random.normal(media, sigma, img.shape)
    
    # Soma o ruído e usa clip para garantir que não passe de 255 ou fique abaixo de 0
    resultado = np.clip(img.astype(np.float32) + ruido, 0, 255)
    
    return resultado.astype(np.uint8)

def aplicar_ruido_aleatorio(img, porcentagem, valor):
    """Função auxiliar para aplicar ruídos impulsivos (sal ou pimenta)."""
    resultado = img.copy()
    
    # Calcula a quantidade total de pixels que vão ser afetados pela porcentagem
    quantidade = int(porcentagem * img.size)
    
    # Escolhe posições aleatórias x e y na matriz
    x = np.random.randint(0, img.shape[0], quantidade)
    y = np.random.randint(0, img.shape[1], quantidade)
    
    # Aplica o valor (0 para pimenta, 255 para sal) nas coordenadas escolhidas
    resultado[x, y] = valor
    
    return resultado

def adicionar_ruido_sal(img, porcentagem=0.02):
    # Ruído sal aplica pixels brancos (255)
    return aplicar_ruido_aleatorio(img, porcentagem, 255)

def adicionar_ruido_pimenta(img, porcentagem=0.02):
    # Ruído pimenta aplica pixels pretos (0)
    return aplicar_ruido_aleatorio(img, porcentagem, 0)

def adicionar_ruido_sal_pimenta(img, porcentagem=0.02):
    # Metade da porcentagem vai para sal (branco), e a outra metade para pimenta (preto)
    resultado = adicionar_ruido_sal(img, porcentagem / 2)
    return adicionar_ruido_pimenta(resultado, porcentagem / 2)
