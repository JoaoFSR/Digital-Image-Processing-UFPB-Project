import numpy as np

def base_frequencia(img):
    """
    Função auxiliar para centralizar a Transformada de Fourier e calcular 
    a matriz de distâncias D(u, v) a partir do centro do espectro.
    """
    linhas, colunas = img.shape
    
    # Cria as coordenadas de frequência para a grade numérica
    u = np.arange(linhas)
    v = np.arange(colunas)
    
    # np.meshgrid gera as matrizes de coordenadas bidimensionais (U, V) necessárias para os cálculos
    V, U = np.meshgrid(v, u)
    
    # Calcula a distância euclidiana de cada ponto (u, v) até o centro da imagem (linhas/2, colunas/2)
    D = np.sqrt((U - linhas/2)**2 + (V - colunas/2)**2)
    
    # np.fft.fft2 aplica a Transformada Rápida de Fourier Bidimensional na imagem
    # np.fft.fftshift move a frequência zero (componente DC) dos cantos para o centro da matriz
    F = np.fft.fftshift(np.fft.fft2(img))
    
    return D, F

def butterworth_passa_baixa(img, D0=30, n=2):
    """
    Filtro passa-baixa de Butterworth.
    Atenua as altas frequências de forma suave, reduzindo ruídos e suavizando a imagem.
    """
    D, F = base_frequencia(img)
    
    # Fórmula matemática padrão do filtro de Butterworth Passa-Baixa: H = 1 / [1 + (D / D0)^(2n)]
    H = 1 / (1 + (D / D0)**(2*n))
    
    # Aplica o filtro multiplicando o espectro da imagem (F) pela função de transferência (H)
    G = F * H
    
    # np.fft.ifftshift desfaz a centralização do espectro (devolve para os cantos) antes da inversão
    G = np.fft.ifftshift(G)
    
    # np.fft.ifft2 aplica a Transformada Inversa de Fourier para trazer a imagem de volta para o domínio do espaço
    resultado = np.fft.ifft2(G)
    
    # Como o resultado da inversa gera números complexos, extraímos apenas a magnitude real com np.abs 
    # e convertemos de volta para inteiros de 8 bits sem sinal (0-255)
    return np.abs(resultado).astype(np.uint8)

def butterworth_passa_alta(img, D0=30, n=2):
    """
    Filtro passa-alta de Butterworth.
    Atenua as baixas frequências e preserva as altas, sendo ideal para realçar bordas e detalhes.
    """
    D, F = base_frequencia(img)
    
    # Fórmula matemática do filtro de Butterworth Passa-Alta: H = 1 - Passa-Baixa
    H = 1 - 1 / (1 + (D / D0)**(2*n))
    
    # Multiplica no domínio da frequência
    G = F * H
    
    # Desfaz a centralização e reconverte para o domínio do espaço
    G = np.fft.ifftshift(G)
    resultado = np.fft.ifft2(G)
    
    return np.abs(resultado).astype(np.uint8)

def gaussiano_passa_baixa(img, D0=30):
    """
    Filtro passa-baixa Gaussiano.
    Suaviza a imagem de forma natural sem gerar o efeito de ondulação (ringing) comum nos filtros ideais.
    """
    D, F = base_frequencia(img)
    
    # Fórmula matemática do filtro Gaussiano Passa-Baixa: H = e^(-D^2 / (2 * D0^2))
    H = np.exp(-(D**2) / (2 * (D0**2)))
    
    # Aplica o filtro multiplicando as matrizes
    G = F * H
    
    # Desfaz a centralização e aplica a Transformada Inversa
    G = np.fft.ifftshift(G)
    resultado = np.fft.ifft2(G)
    
    return np.abs(resultado).astype(np.uint8)

def gaussiano_passa_alta(img, D0=30):
    """
    Filtro passa-alta Gaussiano.
    Realça as bordas e os detalhes de alta frequência atenuando suavemente as baixas frequências.
    """
    D, F = base_frequencia(img)
    
    # Fórmula matemática do filtro Gaussiano Passa-Alta: H = 1 - Passa-Baixa
    H = 1 - np.exp(-(D**2) / (2 * (D0**2)))
    
    # Aplica o filtro multiplicando as matrizes
    G = F * H
    
    # Desfaz a centralização e aplica a Transformada Inversa
    G = np.fft.ifftshift(G)
    resultado = np.fft.ifft2(G)
    
    return np.abs(resultado).astype(np.uint8)