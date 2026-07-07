import cv2
import numpy as np

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

def filtro_minimo(imagem):

    kernel = np.ones((3, 3), dtype=np.uint8)
    minimo = cv2.erode(imagem, kernel)

    return minimo

def filtro_maximo(imagem):

    kernel = np.ones((3, 3), dtype=np.uint8)
    maximo = cv2.dilate(imagem, kernel)

    return maximo

def mascara_agucamento(imagem):

    kernel = np.array([
    [ 0,-1, 0],
    [-1, 5,-1],
    [ 0,-1, 0]
    ])

    aguecamento = cv2.filter2D(imagem, -1, kernel)
    return aguecamento

def filtro_passa_alta_gaussiano(img, ksize=5, sigma=1.5):
    blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
    resultado = cv2.subtract(img, blur)
    return resultado

def filtro_mediana_adaptativo(img, tamanho_max=7):

    resultado = img.copy()

    linhas, colunas = img.shape

    for i in range(linhas):
        for j in range(colunas):

            tamanho = 3

            while tamanho <= tamanho_max:

                r = tamanho // 2

                x1 = max(i-r,0)
                x2 = min(i+r+1,linhas)

                y1 = max(j-r,0)
                y2 = min(j+r+1,colunas)

                janela = img[x1:x2,y1:y2]

                zmin = np.min(janela)
                zmax = np.max(janela)
                zmed = np.median(janela)
                zxy = img[i,j]

                if zmin < zmed < zmax:

                    if zmin < zxy < zmax:
                        resultado[i,j] = zxy
                    else:
                        resultado[i,j] = zmed

                    break

                tamanho += 2

    return resultado