# pip install streamlit
# cd C:\Projetos_VS_Code\BabyGrowth
# venv\Scripts\Activate (PowerShell)
# streamlit run main.py (python -m streamlit run main.py)

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Configuração da página
st.set_page_config(page_title="🎒 Peso da Mochila", layout="centered")

# Base de dados (Nome, Quantidade de livros, Peso da mochila)
dados = np.array([
    ["Felipe", 10, 8.5], ["Carlos", 1, 1.3], ["Igor", 3, 2.8],
    ["Alícia", 9, 7.9], ["Yasmin", 8, 7.3], ["Helena", 3, 3.1],
    ["Eduardo", 2, 2.0], ["Ana", 1, 1.2], ["Otávio", 5, 4.9],
    ["Davi", 10, 8.6], ["Fernanda", 2, 2.3],  ["Ester", 10, 8.8],
    ["Zeca", 8, 6.9], ["Benício", 9, 8.0], ["Vanessa", 7, 6.2],
    ["Daniela", 2, 2.1], ["Patrícia", 6, 5.5],
    ["Camila", 9, 7.7],
    ["Gabriel", 3, 2.9],
    ["Marcos", 5, 4.6],
    ["Bruno", 1, 1.0],
    ["Natália", 5, 4.5],
    ["Thiago", 7, 6.4],
    ["Xavier", 8, 7.1],
    ["Larissa", 4, 3.6],
    ["Ricardo", 6, 5.8],
    ["Kaio", 4, 4.0],
    ["Sabrina", 6, 5.3],
    ["Juliana", 4, 3.8],
    ["William", 7, 6.7]
])

# Separando X (quantidade de livros) e Y (peso da mochila)
X = dados[:, 0].reshape(-1, 1)  # Quantidade de livros
Y = dados[:, 1]  # Peso da mochila

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, Y)

# Obtendo coeficientes do modelo
m = modelo.coef_[0]  # Coeficiente angular (peso médio por livro)
b = modelo.intercept_  # Intercepto (peso base da mochila)

# Descrição do projeto
st.write(
    """
    🎒 **Estimativa do Peso da Mochila com Base na Quantidade de Livros**  
    Este projeto utiliza **Aprendizado de Máquina** para prever o peso aproximado 
    de uma mochila com base no número de livros carregados.  
    O modelo foi treinado usando **Regressão Linear Simples**, analisando dados reais e simulados. 📚📈
    """
)

# Criando as abas dos modelos
aba1, aba2 = st.tabs(["🤖 Modelo Supervisionado", "🔍 Modelo Não Supervisionado"])

# Aba Modelo Supervisionado
with aba1:
    # Descrição do Modelo Supervisionado
    st.write(
    """
    🎒 **Estimativa do Peso da Mochila com Base na Quantidade de Livros**  
    Este projeto utiliza **Aprendizado de Máquina** para prever o peso aproximado 
    de uma mochila com base no número de livros carregados.  
    O modelo foi treinado usando **Regressão Linear Simples**, analisando dados reais e simulados. 📚📈
    """
    )

    # Criando as abas do Modelo Supervisionado
    aba3, aba4 = st.tabs(["📊 Previsão", "🏗️ Parâmetros do Modelo"])

    # Aba Previsão
    with aba3:
        st.title("🎒 Estimativa do Peso da Mochila")
        st.write("📚 Informe a quantidade de livros na mochila para obter a estimativa de peso.")

        quantidade_livros = st.number_input("Quantidade de livros:", min_value=1, max_value=10, step=1)

        if st.button("Estimar Peso"):
            novo_valor = np.array([[quantidade_livros]])
            peso_estimado = modelo.predict(novo_valor)
            st.success(f"📖 Com {quantidade_livros} livros, a mochila deve pesar cerca de {peso_estimado[0]:.2f} kg.")

            # Gerando gráfico
            plt.scatter(X, Y, color='blue', label="Dados Reais")
            plt.plot(X, modelo.predict(X), color='red', label="Regressão Linear")
            plt.scatter(novo_valor, peso_estimado, color='green', marker='o', label="Previsão")
            plt.xlabel("Quantidade de Livros")
            plt.ylabel("Peso da Mochila (kg)")
            plt.legend()
            plt.title("Estimativa do Peso da Mochila")

            # Exibir gráfico no Streamlit
            st.pyplot(plt)

    # Aba Parâmetros do Modelo
    with aba4:
        st.title("🏗️ Parâmetros da Regressão Linear")
        st.write(f"**Coeficiente Angular (a)**: {m:.4f} (peso médio por livro)")
        st.write(f"**Intercepto (b)**: {b:.4f} (peso da mochila sem livros)")
        st.latex(r"y = a \cdot x + b")
        st.latex(rf"y = {m:.4f} \cdot x + {b:.4f}")
        st.title("📜 Dados Utilizados para Treino")
        df = pd.DataFrame(dados, columns=["Quantidade de Livros", "Peso da Mochila (kg)"])
        st.dataframe(df)

# Aba Modelo Não Supervisionado
with aba2:
    st.title("🏗️ Parâmetros da Regressão Linear")
    