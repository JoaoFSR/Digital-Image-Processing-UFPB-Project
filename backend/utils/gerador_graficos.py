# Definimos o backend do matplotlib como 'Agg' ANTES de importar o pyplot.
# Isso é crucial para rodar em um servidor (back-end), pois diz ao matplotlib 
# para renderizar as imagens na memória e não tentar abrir uma janela gráfica na tela (isso mudará no futuro).
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import io
import base64

def gerar_grafico_histograma_base64(imagem, titulo="Histograma"):
    """Gera o histograma da imagem e o retorna como string Base64."""
    
    # Cria uma nova figura com o tamanho especificado (largura, altura)
    plt.figure(figsize=(6, 4))
    
    # A função plt.hist espera um array unidimensional (1D). 
    # O método .ravel() do NumPy "achata" a matriz de pixels da imagem 2D ou 3D em uma única lista.
    # bins=256 significa que queremos 256 barras (uma para cada nível de cinza de 0 a 255).
    plt.hist(imagem.ravel(), bins=256, range=[0, 256], color='black', alpha=0.7)
    
    # Configurações estéticas do gráfico
    plt.title(titulo)
    plt.xlabel('Intensidade de Pixel')
    plt.ylabel('Frequência')
    plt.grid(axis='y', alpha=0.75)
    
    # Em vez de salvar no disco rígido (ex: plt.savefig('grafico.png')), 
    # criamos um "arquivo em memória" usando io.BytesIO() para ser mais rápido.
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close() # Fecha a figura para liberar memória do servidor
    
    # Move o ponteiro de leitura do buffer de volta para o início (posição 0)
    buf.seek(0)
    
    # Lê os bytes gerados, converte para Base64 (formato texto) e decodifica para string
    imagem_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Retorna no formato de URL de dados (Data URI) para o front-end poder exibir diretamente na tag <img>
    return f"data:image/png;base64,{imagem_base64}"