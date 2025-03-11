import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Configuração da página
st.set_page_config(page_title="🎒 Peso da Mochila", layout="centered")

# Base de dados (Nome, Quantidade de livros, Peso da mochila)
dados = np.array([
    ["Felipe", 10, 8.5], ["Carlos", 1, 1.3], ["Igor", 3, 2.8],
    ["Alícia", 9, 7.9], ["Yasmin", 8, 7.3], ["Helena", 3, 3.1],
    ["Eduardo", 2, 2.0], ["Ana", 1, 1.2], ["Otávio", 5, 4.9],
    ["Davi", 10, 8.6], ["Fernanda", 2, 2.3],  ["Ester", 10, 8.8],
    ["Zeca", 8, 6.9], ["Benício", 9, 8.0], ["Vanessa", 7, 6.2],
    ["Daniela", 2, 2.1], ["Patrícia", 6, 5.5], ["Camila", 9, 7.7],
    ["Gabriel", 3, 2.9], ["Marcos", 5, 4.6], ["Bruno", 1, 1.0],
    ["Natália", 5, 4.5], ["Thiago", 7, 6.4], ["Xavier", 8, 7.1],
    ["Larissa", 4, 3.6], ["Ricardo", 6, 5.8], ["Kaio", 4, 4.0],
    ["Sabrina", 6, 5.3], ["Juliana", 4, 3.8], ["William", 7, 6.7]
])

# Separando X (quantidade de livros) e Y (peso da mochila)
X = dados[:, 1].astype(float).reshape(-1, 1)  # Quantidade de livros (convertido para float)
Y = dados[:, 2].astype(float)  # Peso da mochila (convertido para float)

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
        st.write("Informe a quantidade de livros na mochila para obter a estimativa de peso.")

        quantidade_livros = st.number_input("📚 Quantidade de livros:", min_value=1, max_value=50, step=1)

        if st.button("Estimar Peso"):
            novo_valor = np.array([[quantidade_livros]])
            peso_estimado = modelo.predict(novo_valor)
            st.info(f"📖 Com {quantidade_livros} livros, a mochila deve pesar cerca de {peso_estimado[0]:.2f} kg.")

            # Criando valores de X extendidos até a quantidade informada
            X_extendido = np.arange(1, quantidade_livros + 1).reshape(-1, 1)
            Y_predito = modelo.predict(X_extendido)

            # Criando o gráfico
            fig, ax = plt.subplots(figsize=(8, 5))

            ax.scatter(X, Y, color='blue', label="Dados Reais")
            ax.plot(X, modelo.predict(X), color='red', linestyle='dashed', label="Regressão Linear")
            ax.plot(X_extendido, Y_predito, color='red', linestyle='dashed', label="")
            ax.scatter(novo_valor, peso_estimado, color='green', marker='o', s=100, label="Previsão")

            ax.set_xlabel("Quantidade de Livros", fontsize=12)
            ax.set_ylabel("Peso da Mochila (kg)", fontsize=12)
            ax.set_title("Estimativa do Peso da Mochila", fontsize=14)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            fig.tight_layout()
            st.pyplot(fig)


    # Aba Parâmetros do Modelo
    with aba4:
        st.title("🏗️ Parâmetros da Regressão Linear")
        st.write(f"**Coeficiente Angular (a)**: {m:.4f} (peso médio por livro)")
        st.write(f"**Intercepto (b)**: {b:.4f} (peso da mochila sem livros)")
        st.latex(r"y = a \cdot x + b")
        st.latex(rf"y = {m:.4f} \cdot x + {b:.4f}")
        st.title("📜 Dados Utilizados para Treino")
        df = pd.DataFrame(dados, columns=["Nome", "Quantidade de Livros", "Peso da Mochila (kg)"])
        st.dataframe(df)

# Aba Modelo Não Supervisionado
with aba2:
    # Extraindo apenas o peso da mochila para treinar o modelo
    Y = dados[:, 2].astype(float).reshape(-1, 1)  # Peso da mochila como matriz

    # Criando um DataFrame para exibição no Streamlit
    df = pd.DataFrame(dados, columns=["Nome", "Quantidade de Livros", "Peso da Mochila (kg)"])

    # Criando o modelo K-Means com 3 clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(Y)

    # Criar um dicionário para ordenar os clusters corretamente
    clusters_ordenados = sorted(range(3), key=lambda i: kmeans.cluster_centers_[i, 0])

    # Criando o dicionário correto de cores e classificações
    cores = {
        clusters_ordenados[0]: "Leve 🟢",
        clusters_ordenados[1]: "Média 🟡",
        clusters_ordenados[2]: "Pesada 🔴"
    }

    # Aplicando os clusters corrigidos ao DataFrame
    df["Cluster"] = kmeans.labels_
    df["Classificação"] = df["Cluster"].map(cores)

    # Criando Gráfico
    st.title("📊 Classificação das Mochilas por Peso 📚🎒")
    st.write("O modelo K-Means agrupa mochilas automaticamente com base na quantidade de livros e no peso total.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X, Y, c=kmeans.labels_, cmap="viridis", s=100)
    ax.set_xlabel("Quantidade de Livros")
    ax.set_ylabel("Peso da Mochila (kg)")
    ax.set_title("Clusterização das Mochilas")

    for i, txt in enumerate(dados[:, 0]):
        ax.annotate(txt, (X[i], Y[i]), fontsize=8, xytext=(5, 5), textcoords="offset points")

    st.pyplot(fig)

    # Exibir a tabela
    st.title("📜 Dados Classificados")
    st.dataframe(df.drop(columns=["Cluster"]))

    # Previsão para um novo dado
    st.title("🔍 Classifique uma Nova Mochila")
    num_livros = st.number_input("📚 Quantidade de Livros:", min_value=1, max_value=50, step=1)

    if st.button("Classificar"):
        novo_dado = np.array([[num_livros]])
        cluster_predito = kmeans.predict(novo_dado)[0]
        classificacao = cores[cluster_predito]

        if classificacao == "Leve 🟢":
            st.success(f"A mochila inserida foi classificada como: **{classificacao}**")
        elif classificacao == "Média 🟡":
            st.warning(f"A mochila inserida foi classificada como: **{classificacao}**")
        else:
            st.error(f"A mochila inserida foi classificada como: **{classificacao}**")