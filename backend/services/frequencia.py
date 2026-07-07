def butterworth_passa_baixa(img, D0=30, n=2):

    linhas, colunas = img.shape

    u = np.arange(linhas)
    v = np.arange(colunas)

    V, U = np.meshgrid(v, u)

    D = np.sqrt((U - linhas/2)**2 + (V - colunas/2)**2)

    H = 1 / (1 + (D / D0)**(2*n))

    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)

    G = F * H

    G = np.fft.ifftshift(G)
    resultado = np.fft.ifft2(G)

    return np.abs(resultado).astype(np.uint8)

def butterworth_passa_alta(img, D0=30, n=2):

    linhas, colunas = img.shape

    u = np.arange(linhas)
    v = np.arange(colunas)

    V, U = np.meshgrid(v, u)

    D = np.sqrt((U - linhas/2)**2 + (V - colunas/2)**2)

    H = 1 - 1 / (1 + (D / D0)**(2*n))

    F = np.fft.fft2(img)
    F = np.fft.fftshift(F)

    G = F * H

    G = np.fft.ifftshift(G)
    resultado = np.fft.ifft2(G)

    return np.abs(resultado).astype(np.uint8)