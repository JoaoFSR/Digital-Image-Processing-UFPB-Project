import cv2
import numpy as np

def filtro_media_gaussiana(imagem, tamanho_janela: int, sigma: float):
    """
    Borra a imagem usando uma distribuição gaussiana (forma de sino).
    Diferente da média simples, os pixels mais próximos do centro têm mais "peso",
    preservando melhor as bordas da imagem enquanto reduz o ruído.
    """
    # Filtros espaciais de convolução exigem que o tamanho da janela (kernel) seja um número ímpar (ex: 3, 5, 7).
    # Isso acontece porque o filtro precisa ter um "pixel central" bem definido matematicamente.
    if tamanho_janela % 2 == 0:
        tamanho_janela += 1 
        
    # Aplica o desfoque passando as dimensões da janela e o desvio padrão (sigma)
    return cv2.GaussianBlur(imagem, (tamanho_janela, tamanho_janela), sigmaX=sigma, sigmaY=sigma)

def filtro_mediana(imagem, tamanho_janela: int):
    """
    Filtro não-linear que substitui o pixel central pela mediana da sua vizinhança.
    É excelente para remover ruídos impulsivos (como o sal e pimenta) sem borrar as bordas.
    """
    # Garante que a janela seja ímpar
    if tamanho_janela % 2 == 0: 
        tamanho_janela += 1
        
    return cv2.medianBlur(imagem, tamanho_janela)

def filtro_minimo(imagem, tamanho_janela: int):
    """
    Filtro de mínimo (Erosão na morfologia matemática).
    Substitui o pixel central pelo menor valor da vizinhança. 
    Na prática, expande as áreas escuras da imagem e "come" as áreas claras.
    """
    # Cria uma matriz preenchida com 1s para definir o formato da vizinhança
    kernel = np.ones((tamanho_janela, tamanho_janela), dtype=np.uint8)
    return cv2.erode(imagem, kernel)

def filtro_maximo(imagem, tamanho_janela: int):
    """
    Filtro de máximo (Dilatação na morfologia matemática).
    Substitui o pixel central pelo maior valor da vizinhança.
    Na prática, expande as áreas claras e "engole" detalhes escuros.
    """
    # Cria a matriz da vizinhança
    kernel = np.ones((tamanho_janela, tamanho_janela), dtype=np.uint8)
    return cv2.dilate(imagem, kernel)

def mascara_agucamento(imagem, ganho: float, tamanho_janela: int = 3):
    """
    Realça as bordas da imagem usando a técnica de "Unsharp Masking".
    A lógica é: se subtrairmos uma versão borrada da imagem original, 
    o que sobra são apenas as bordas (a máscara). Depois somamos isso de volta.
    """
    if tamanho_janela % 2 == 0: 
        tamanho_janela += 1
        
    # 1. Cria a versão borrada da imagem
    borrada = cv2.GaussianBlur(imagem, (tamanho_janela, tamanho_janela), 0)
    
    # 2. Subtrai a borrada da original para isolar as bordas (criar a máscara)
    # IMPORTANTE: Usamos cv2.CV_16S (inteiro de 16 bits com sinal) porque ao subtrair,
    # podemos gerar valores negativos. Se usarmos 8 bits (0-255), os negativos viram 0.
    mascara = cv2.subtract(imagem, borrada, dtype=cv2.CV_16S)
    
    # 3. Multiplica a máscara pelo ganho desejado e soma de volta à imagem original
    # Usamos np.clip para garantir que os valores finais fiquem espremidos entre 0 e 255
    agucada = np.clip(imagem + ganho * mascara, 0, 255).astype(np.uint8)
    return agucada

def realce_laplaciano(imagem):
    """
    Aplica o filtro Laplaciano (derivada de segunda ordem). 
    Detecta bordas capturando variações bruscas de intensidade.
    """
    # Calcula o Laplaciano. 
    # Usamos cv2.CV_64F (float de 64 bits) pois as derivadas geram muitos valores negativos (transições de claro para escuro).
    laplaciano = cv2.Laplacian(imagem, cv2.CV_64F)
    
    # O Laplaciano puro tem valores fora da escala normal de imagem.
    # cv2.convertScaleAbs tira o valor absoluto e converte para 8 bits para podermos visualizar como imagem.
    lap_ajustado = cv2.convertScaleAbs(laplaciano)
    
    # O realce é feito subtraindo o Laplaciano (com sinal) da imagem original.
    realcada = np.clip(imagem - laplaciano, 0, 255).astype(np.uint8)
    
    return realcada, lap_ajustado

def gradiente_sobel(imagem):
    """
    Calcula a magnitude do gradiente usando o operador de Sobel (derivada de primeira ordem).
    Ele detecta bordas focando na direção em que a intensidade da imagem mais muda.
    """
    # Calcula a derivada na direção X (horizontal) - detecta bordas verticais
    sobelx = cv2.Sobel(imagem, cv2.CV_64F, 1, 0, ksize=3)
    
    # Calcula a derivada na direção Y (vertical) - detecta bordas horizontais
    sobely = cv2.Sobel(imagem, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combina as direções X e Y calculando a magnitude geométrica: raiz(X^2 + Y^2)
    magnitude = cv2.magnitude(sobelx, sobely)
    
    # Converte os resultados numéricos para a escala de imagem visível (0-255)
    return cv2.convertScaleAbs(magnitude)


def filtro_mediana_adaptativo(img, tamanho_max=7):
    """
    Filtro de mediana inteligente. 
    Se a própria mediana for um ruído, ele aumenta o tamanho da janela de busca.
    Ele só altera o pixel central se tiver certeza de que ele é um ruído impulsivo,
    preservando muito mais os detalhes da imagem do que a mediana comum.
 
    IMPORTANTE: esta função foi reescrita para ser vetorizada. A versão anterior
    percorria a imagem pixel a pixel em Python puro (dois 'for' aninhados + um
    'while' interno), o que é extremamente lento em imagens reais (podia levar
    minutos e parecer que o app estava travado). Agora calculamos min/máx/mediana
    para a imagem INTEIRA de uma vez, para cada tamanho de janela, usando
    operações nativas do OpenCV/NumPy. O resultado matemático é o mesmo, mas a
    execução passa a ser quase instantânea.
    """
    img = img.astype(np.uint8)
    resultado = img.copy()
 
    # Marca quais pixels já tiveram seu valor final decidido (Nível A satisfeito)
    decidido = np.zeros(img.shape, dtype=bool)
 
    tamanho = 3
    zmed = None
    while tamanho <= tamanho_max:
        kernel = np.ones((tamanho, tamanho), dtype=np.uint8)
 
        # zmin/zmax equivalem a erosão/dilatação (mínimo/máximo da vizinhança)
        # zmed é a mediana da vizinhança — cv2.medianBlur já faz isso pra imagem toda
        zmin = cv2.erode(img, kernel)
        zmax = cv2.dilate(img, kernel)
        zmed = cv2.medianBlur(img, tamanho)
 
        # Nível A: a mediana da janela não é ela mesma um ruído (não é o mín nem o máx)
        nivel_a = (zmin < zmed) & (zmed < zmax)
 
        # Só processa pixels que ainda não foram decididos em uma janela menor
        aplicavel = nivel_a & ~decidido
 
        # Nível B: o pixel central é ruído (igual ao mín ou ao máx da janela)?
        nivel_b = (zmin < img) & (img < zmax)
 
        mantem_original = aplicavel & nivel_b
        troca_por_mediana = aplicavel & ~nivel_b
 
        resultado[mantem_original] = img[mantem_original]
        resultado[troca_por_mediana] = zmed[troca_por_mediana]
 
        decidido |= aplicavel
        tamanho += 2
 
    # Pixels que nunca satisfizeram o Nível A mesmo na maior janela testada:
    # usamos a mediana da maior janela como valor de segurança (fallback razoável)
    if zmed is not None:
        resultado[~decidido] = zmed[~decidido]
 
    return resultado

