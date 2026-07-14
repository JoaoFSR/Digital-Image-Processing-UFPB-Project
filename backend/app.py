import streamlit as st
import cv2
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

# ==========================================
# IMPORTS DOS SEUS SERVIÇOS (PASTA services)
# ==========================================
from services.espacos_cor import decompor_rgb, decompor_hsv
from services.intensidade import (limiarizacao, transformacao_logaritmica, transformacao_potencia, 
                                  equalizacao_histograma, fatiamento_intensidade, 
                                  adicionar_ruido_gaussiano, adicionar_ruido_sal, 
                                  adicionar_ruido_pimenta, adicionar_ruido_sal_pimenta)
from services.filtros_espaciais import (filtro_media_gaussiana, filtro_mediana, filtro_minimo, 
                                        filtro_maximo, mascara_agucamento, realce_laplaciano, 
                                        gradiente_sobel, filtro_mediana_adaptativo)
from services.frequencia import (butterworth_passa_baixa, butterworth_passa_alta, 
                                 gaussiano_passa_baixa, gaussiano_passa_alta)

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Processamento de Imagens", layout="wide")
st.title("📸 Processamento Digital de Imagens (PDI)")

# Variável de estado para manter a imagem atual na memória (Permite usar a saída como entrada)
if 'imagem_entrada' not in st.session_state:
    st.session_state['imagem_entrada'] = None

# ==========================================
# FUNÇÕES AUXILIARES
# ==========================================
def verificar_escala_cinza(imagem):
    return len(imagem.shape) == 2 or (len(imagem.shape) == 3 and imagem.shape[2] == 1)

def converter_para_download(imagem):
    """Converte a matriz OpenCV de volta para bytes de arquivo .png."""
    sucesso, buffer = cv2.imencode('.png', imagem)
    if sucesso:
        return BytesIO(buffer.tobytes())
    return None

def gerar_grafico_histograma(imagem, titulo):
    """Gera um gráfico do matplotlib de frequências de pixel."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    ax.hist(imagem.ravel(), bins=256, range=[0, 256], color='gray', alpha=0.8)
    ax.set_title(titulo)
    ax.set_xlabel('Intensidade')
    ax.set_ylabel('Frequência')
    st.pyplot(fig)

def preparar_exibicao(img):
    """Garante que imagens coloridas BGR do OpenCV sejam exibidas em RGB no Streamlit."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

# ==========================================
# MENU LATERAL - CARREGAR ARQUIVO
# ==========================================
st.sidebar.header("⚙️ Opções de Arquivo")

# REQUISITO DO PROJETO: Apenas formato .png é permitido
arquivo_imagem = st.sidebar.file_uploader("Carregue uma imagem (.png)", type=['png'])

# Lógica para gerenciar o upload de arquivos e o estado da imagem de entrada
if arquivo_imagem is not None:
    if st.session_state.get('ultimo_upload') != arquivo_imagem.name:
        file_bytes = np.asarray(bytearray(arquivo_imagem.read()), dtype=np.uint8)
        st.session_state['imagem_entrada'] = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        st.session_state['ultimo_upload'] = arquivo_imagem.name

# ==========================================
# CONTEÚDO PRINCIPAL DA APLICAÇÃO
# ==========================================
if st.session_state['imagem_entrada'] is not None:
    img_original = st.session_state['imagem_entrada']
    eh_cinza = verificar_escala_cinza(img_original)
    
    st.sidebar.divider()
    st.sidebar.header("🛠️ Seleção de Processos")
    
    # --- FILTRAGEM DINÂMICA DE PROCESSOS ---
    # Processos comuns a qualquer imagem (cinza ou colorida)
    processos_disponiveis = [
        "Nenhum (Original)", "Transf. Logarítmica", "Transf. Potência (Gama)",
        "Filtro Média Gaussiana", "Filtro Mediana", "Filtro Mínimo", "Filtro Máximo", 
        "Máscara de Aguçamento", "Realce por Laplaciano", "Gradiente de Sobel",
        "Ruído: Gaussiano", "Ruído: Sal", "Ruído: Pimenta", "Ruído: Sal e Pimenta"
    ]
    
    # REQUISITO DO PROJETO: Desabilitar/Ocultar processos se não forem aplicáveis
    if eh_cinza:
        # Processos que só fazem sentido matematicamente em tons de cinza
        processos_disponiveis.extend([
            "Limiarização", 
            "Fatiamento por Intensidade", 
            "Equalização de Histograma",
            "Filtro Mediana Adaptativo", 
            "Filtro Frequência: Passa-Baixa Butterworth", 
            "Filtro Frequência: Passa-Alta Butterworth",
            "Filtro Frequência: Passa-Baixa Gaussiano", 
            "Filtro Frequência: Passa-Alta Gaussiano"
        ])
    else:
        # Processos que só fazem sentido em imagens coloridas (RGB)
        processos_disponiveis.extend([
            "Decomposição RGB", 
            "Decomposição HSV"
        ])
        
    # Ordena as opções alfabeticamente para melhor visualização (mantendo o "Nenhum" no topo)
    processos_disponiveis.remove("Nenhum (Original)")
    processos_disponiveis = ["Nenhum (Original)"] + sorted(processos_disponiveis)
    
    processo_escolhido = st.sidebar.selectbox("Selecione o processo:", processos_disponiveis)
    
    # ==========================================
    # CONFIGURAÇÃO DE PARÂMETROS E EXECUÇÃO
    # ==========================================
    st.sidebar.subheader("🎛️ Parâmetros do Processo")
    img_processada = None
    dados_acessorios = None 
    
    try:
        if processo_escolhido == "Nenhum (Original)":
            img_processada = img_original.copy()
            
        # --- PROCESSOS DE INTENSIDADE ---
        elif processo_escolhido == "Limiarização":
            k = st.sidebar.slider("Valor do Limiar (k)", 0, 255, 127)
            img_processada = limiarizacao(img_original, k)
            
        elif processo_escolhido == "Transf. Logarítmica":
            c = st.sidebar.number_input("Constante de Ganho (c)", value=45.0)
            img_processada = transformacao_logaritmica(img_original, c)
            
        elif processo_escolhido == "Transf. Potência (Gama)":
            c = st.sidebar.number_input("Constante de Ganho (c)", value=1.0)
            gamma = st.sidebar.slider("Valor de Gama (γ)", 0.1, 5.0, 1.0, 0.1)
            img_processada = transformacao_potencia(img_original, c, gamma)
            
        elif processo_escolhido == "Fatiamento por Intensidade":
            colA, colB = st.sidebar.columns(2)
            A = colA.number_input("Início da Faixa (A)", 0, 255, 100)
            B = colB.number_input("Fim da Faixa (B)", 0, 255, 200)
            preservar = st.sidebar.checkbox("Preservar Fundo", value=False)
            img_processada = fatiamento_intensidade(img_original, A, B, preservar)
            
        elif processo_escolhido == "Equalização de Histograma":
            img_processada = equalizacao_histograma(img_original)
            dados_acessorios = "histograma" # Ativa exibição de histogramas
            
        # --- PROCESSOS DE DECOMPOSIÇÃO ---
        elif processo_escolhido == "Decomposição RGB":
            r, g, b = decompor_rgb(img_original)
            img_processada = img_original 
            dados_acessorios = {"tipo": "rgb", "canais": (r, g, b)}
            
        elif processo_escolhido == "Decomposição HSV":
            h, s, v = decompor_hsv(img_original)
            img_processada = img_original
            dados_acessorios = {"tipo": "hsv", "canais": (h, s, v)}

        # --- FILTROS ESPACIAIS ---
        elif processo_escolhido in ["Filtro Mediana", "Filtro Mínimo", "Filtro Máximo"]:
            janela = st.sidebar.slider("Tamanho da Janela de Convolução", 3, 31, 3, 2)
            if processo_escolhido == "Filtro Mediana": 
                img_processada = filtro_mediana(img_original, janela)
            elif processo_escolhido == "Filtro Mínimo": 
                img_processada = filtro_minimo(img_original, janela)
            elif processo_escolhido == "Filtro Máximo": 
                img_processada = filtro_maximo(img_original, janela)
            
        elif processo_escolhido == "Filtro Média Gaussiana":
            janela = st.sidebar.slider("Tamanho da Janela de Convolução", 3, 31, 5, 2)
            sigma = st.sidebar.slider("Desvio Padrão (σ)", 0.1, 10.0, 1.0)
            img_processada = filtro_media_gaussiana(img_original, janela, sigma)
            
        elif processo_escolhido == "Máscara de Aguçamento":
            janela = st.sidebar.slider("Janela de Convolução", 3, 31, 3, 2)
            ganho = st.sidebar.slider("Ganho de Aguçamento", 0.1, 5.0, 1.0, 0.1)
            img_processada = mascara_agucamento(img_original, ganho, janela)
            
        elif processo_escolhido == "Realce por Laplaciano":
            realcada, lap_ajustado = realce_laplaciano(img_original)
            img_processada = realcada
            dados_acessorios = {"tipo": "laplaciano", "imagem": lap_ajustado} # Ativa visualização do Laplaciano
            
        elif processo_escolhido == "Gradiente de Sobel":
            img_processada = gradiente_sobel(img_original)
            
        elif processo_escolhido == "Filtro Mediana Adaptativo":
            tmax = st.sidebar.slider("Tamanho Máximo da Janela", 3, 15, 7, 2)
            img_processada = filtro_mediana_adaptativo(img_original, tmax)
            
        # --- FILTROS EM FREQUÊNCIA ---
        elif "Frequência" in processo_escolhido:
            # Garante escala de cinza de 8 bits temporária caso a imagem de entrada não esteja convertida
            img_freq = img_original.copy() if eh_cinza else cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
            D0 = st.sidebar.slider("Frequência de Corte (D0)", 1, 200, 30)
            
            if "Butterworth" in processo_escolhido:
                n = st.sidebar.slider("Ordem do Filtro (n)", 1, 10, 2)
                if "Passa-Baixa" in processo_escolhido: 
                    img_processada = butterworth_passa_baixa(img_freq, D0, n)
                else: 
                    img_processada = butterworth_passa_alta(img_freq, D0, n)
            else: # Filtro Gaussiano em frequência
                if "Passa-Baixa" in processo_escolhido: 
                    img_processada = gaussiano_passa_baixa(img_freq, D0)
                else: 
                    img_processada = gaussiano_passa_alta(img_freq, D0)
                
        # --- RUÍDOS ---
        elif "Ruído" in processo_escolhido:
            if "Gaussiano" in processo_escolhido:
                media = st.sidebar.number_input("Média da Distribuição", value=0)
                sigma = st.sidebar.slider("Desvio Padrão do Ruído (Sigma)", 1, 100, 20)
                img_processada = adicionar_ruido_gaussiano(img_original, media, sigma)
            else:
                porcentagem = st.sidebar.slider("Distribuição / Intensidade do Ruído", 0.01, 0.50, 0.02, 0.01)
                if "Sal e Pimenta" in processo_escolhido: 
                    img_processada = adicionar_ruido_sal_pimenta(img_original, porcentagem)
                elif "Sal" in processo_escolhido: 
                    img_processada = adicionar_ruido_sal(img_original, porcentagem)
                elif "Pimenta" in processo_escolhido: 
                    img_processada = adicionar_ruido_pimenta(img_original, porcentagem)

    except Exception as e:
        st.error(f"Erro ao aplicar processo: {e}")

    # ==========================================
    # ÁREA PRINCIPAL: VISUALIZAÇÃO LADO A LADO
    # ==========================================
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Imagem de Entrada")
        st.image(preparar_exibicao(img_original), use_container_width=True)
        
    with col2:
        st.subheader("📤 Imagem de Saída")
        if img_processada is not None:
            st.image(preparar_exibicao(img_processada), use_container_width=True)
            
            # --- BOTÕES DE SALVAR E ENCADEAR PROCESSOS ---
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                img_download = converter_para_download(img_processada)
                st.download_button("💾 Salvar como .png", data=img_download, file_name="saida.png", mime="image/png")
            
            with btn_col2:
                if st.button("🔄 Usar Saída como Entrada"):
                    st.session_state['imagem_entrada'] = img_processada
                    st.rerun()

    # ==========================================
    # SEÇÃO: DADOS / IMAGENS ACESSÓRIAS
    # ==========================================
    if dados_acessorios:
        st.divider()
        st.subheader("📊 Dados / Imagens Acessórias")
        
        # Onde o gráfico do histograma é exibido
        if dados_acessorios == "histograma":
            colH1, colH2 = st.columns(2)
            with colH1: 
                gerar_grafico_histograma(img_original, "Histograma Original")
            with colH2: 
                gerar_grafico_histograma(img_processada, "Histograma Equalizado")
            
        elif isinstance(dados_acessorios, dict):
            # Exibe o Laplaciano para o realce
            if dados_acessorios["tipo"] == "laplaciano":
                st.image(preparar_exibicao(dados_acessorios["imagem"]), caption="Laplaciano Ajustado", width=450)
                
            # Exibe os 3 canais de cor separados (RGB ou HSV)
            elif dados_acessorios["tipo"] in ["rgb", "hsv"]:
                c1, c2, c3 = dados_acessorios["canais"]
                lbl1, lbl2, lbl3 = ("Vermelho (R)", "Verde (G)", "Azul (B)") if dados_acessorios["tipo"] == "rgb" else ("Matiz (H)", "Saturação (S)", "Valor (V)")
                
                colC1, colC2, colC3 = st.columns(3)
                with colC1: st.image(preparar_exibicao(c1), caption=lbl1, use_container_width=True)
                with colC2: st.image(preparar_exibicao(c2), caption=lbl2, use_container_width=True)
                with colC3: st.image(preparar_exibicao(c3), caption=lbl3, use_container_width=True)

else:
    st.info("👆 Faça o upload de uma imagem em formato .png no menu lateral para começar.")