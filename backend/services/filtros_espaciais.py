import cv2

def filtro_media_gaussiana(imagem, tamanho_janela: int, sigma: float):
    """Borra a imagem usando uma distribuição gaussiana"""
    
    # iltros espaciais de convolução exigem que o tamanho da janela (kernel) seja um número ímpar, ex: 3x3, 5x5, 7x7 
    # isso acontece porque o filtro precisa ter um "pixel central" definido de forma simétrica.
    if tamanho_janela % 2 == 0:
        tamanho_janela += 1 # ex: se o usuário mandar 4, ajustamos para 5
        
    # aplica o desfoque gaussiano, passamos a tupla (largura, altura) da janela 
    # e os desvios padrões (sigmaX e sigmaY). Se sigmaY for igual a sigmaX, podemos passar só sigmaX.
    img_blur = cv2.GaussianBlur(imagem, (tamanho_janela, tamanho_janela), sigmaX=sigma, sigmaY=sigma)
    
    return img_blur