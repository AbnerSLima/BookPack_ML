import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

# Título da página
st.set_page_config(page_title="🎒 Peso da Mochila", layout="centered")

# Dicionário para converter Tipo de Livro para número
tipo_livro_dict = {"Revista": 1, "Literatura": 2, "Acadêmico": 3}

# Base de dados (Nome, Quantidade de livros, Peso da mochila, Tipo de livro, Volume da mochila)
dados = np.array([
    ["Felipe", 10, 8.5, "Acadêmico", 35],
    ["Carlos", 1, 1.3, "Revista", 15],
    ["Igor", 3, 2.8, "Literatura", 22],
    ["Alícia", 9, 7.9, "Acadêmico", 30],
    ["Yasmin", 8, 7.3, "Literatura", 28],
    ["Helena", 3, 3.1, "Revista", 20],
    ["Eduardo", 2, 2.0, "Literatura", 18],
    ["Ana", 1, 1.2, "Revista", 16],
    ["Otávio", 5, 4.9, "Acadêmico", 25],
    ["Davi", 10, 8.6, "Acadêmico", 40]
])

# Separando X (quantidade de livros) e Y (peso da mochila)
X = dados[:, 1].astype(float).reshape(-1, 1)
Y = dados[:, 2].astype(float)

# Criando e treinando o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X, Y)

# Obtendo coeficientes do modelo
a = modelo.coef_[0]
b = modelo.intercept_

# Descrição do projeto
st.title("🎒 Bem-vindo ao BookPack ML!")
st.write(
    """  
    Você já se perguntou **quanto pesa sua mochila cheia de livros?** 📚🎒 
    Ou gostaria de saber se está carregando peso demais? 🤔

    O **BookPack ML** usa inteligência artificial para te ajudar a **prever o peso da mochila** e **classificá-la automaticamente**!
    """
)

# Criando as abas dos modelos
aba1, aba2 = st.tabs(["🤖 **Modelo Supervisionado**", "🔍 **Modelo Não Supervisionado**"])

# Aba Modelo Supervisionado
with aba1:
    # Descrição do Modelo Supervisionado
    st.write(
    """
    ## 📊 Como funciona o Modelo Supervisionado?

    O Modelo Supervisionado do BookPack ML utiliza um método chamado **Regressão Linear Simples** para prever o peso da sua mochila com base na quantidade de livros que você está carregando. 📚🎒

    🧠 **Como ele aprende?**

    O modelo foi treinado com dados simulados, onde cada entrada contém: _número de livros_ e _peso total da mochila_.
    Ele analisou os padrões nesses dados e descobriu uma relação matemática entre a quantidade de livros e o peso total.    
    Agora, sempre que você informa um número de livros, ele calcula automaticamente o peso estimado da mochila! 📈
    """
    )

    # Criando as abas do Modelo Supervisionado
    aba3, aba4 = st.tabs(["📊 Previsão", "🏗️ Parâmetros do Modelo Supervisionado"])

    # Aba Previsão
    with aba3:
        st.title("🎒 Estimativa do Peso da Mochila")
        st.write(
        """
        1️⃣ Insira a **quantidade de livros** que deseja carregar.
        """
        )

        quantidade_livros = st.number_input("📚 Quantidade de livros:", min_value=1, max_value=50, step=1)

        st.write(
        """
        2️⃣ Veja **quanto sua mochila deve pesar**!
        """
        )

        if st.button("Estimar Peso"):
            novo_valor = np.array([[quantidade_livros]])
            peso_estimado = modelo.predict(novo_valor)
            st.info(f"📖 Com {quantidade_livros} livros, a mochila deve pesar cerca de {peso_estimado[0]:.2f} kg.")

            # Criando valores de X extendidos até a quantidade informada
            X_extendido = np.arange(1, quantidade_livros + 1).reshape(-1, 1)
            Y_predito = modelo.predict(X_extendido)

            st.write(
            """
            3️⃣ Visualize o gráfico mostrando a **relação entre livros e peso**.
            """
            )

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


    # Aba Parâmetros do Modelo Supervisionado
    with aba4:
        st.title("🏗️ Parâmetros da Regressão Linear")
        st.write(f"**Coeficiente Angular (a)**: {a:.4f} (peso médio por livro)")
        st.write(f"**Intercepto (b)**: {b:.4f} (peso da mochila sem livros)")
        st.latex(r"y = a \cdot x + b")
        st.latex(rf"y = {a:.4f} \cdot x + {b:.4f}")
        st.write("Abaixo está a tabela com os dados utilizados:")
        st.title("📜 Dados Utilizados para Treino")

        df = pd.DataFrame(dados, columns=["Nome", "Quantidade de Livros", "Peso da Mochila (kg)", "Tipo de Livro", "Volume da Mochila"])
        st.dataframe(df)

# Aba Modelo Não Supervisionado
with aba2:
    
    # Descrição do Modelo Não Supervisionado
    st.write(
    """
    ## 📊 Como funciona o Modelo Não Supervisionado?

    O Modelo Não Supervisionado do BookPack ML utiliza um método chamado **K-Means**, que **agrupa mochilas automaticamente** de acordo com o peso total.

    🧠 **Como ele aprende?**    
    
    O modelo aprende sozinho **quais mochilas pertencem a cada grupo**, sem precisar de regras pré-definidas!  
    Agora, basta inserir um novo dado e ver em qual grupo sua mochila se encaixa! 🎒🚀
    """
    )

    # Criando as abas do Modelo Não Supervisionado
    aba5, aba6 = st.tabs(["📊 Agrupamento", "🏗️ Parâmetros do Modelo Não Supervisionado"])
    
    # Aba Agrupamento
    with aba5:
        # Extraindo apenas o peso da mochila para treinar o modelo
        Y = dados[:, 2].astype(float).reshape(-1, 1)

        # Criando um DataFrame para exibição no Streamlit
        df = pd.DataFrame(dados, columns=["Nome", "Quantidade de Livros", "Peso da Mochila (kg)", "Tipo de Livro", "Volume da Mochila"])

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


        # Classificar para um nova mochila
        st.title("🔍 Classifique uma Nova Mochila")
        st.write(
        """
        1️⃣ Insira a **quantidade de livros** que deseja carregar.
        """
        )
        num_livros = st.number_input("📚 Quantidade de Livros:", min_value=1, max_value=50, step=1)

        st.write(
        """
        2️⃣ Veja **qual grupo sua mochila se encaixa**!
        """
        )

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

    # Parâmetros do Modelo Não Supervisionado
    with aba6:
        st.title("🏗️ Parâmetros do K-Means")
        st.write(
        """
        O **K-Means** divide os dados em grupos chamados **clusters**, baseando-se na similaridade entre os pesos das mochilas.  
        Cada cluster tem um **centroide**, que representa o peso médio das mochilas dentro daquele grupo.

        **📌 Como os grupos são organizados?**  
        - O algoritmo agrupa mochilas com **pesos semelhantes** automaticamente.  
        - Nós ajustamos a ordem para que os clusters sejam classificados corretamente como **Leve, Média e Pesada**.

        Abaixo está a tabela com os dados agrupados:
        """
        )
        # Exibir a tabela com os dados agrupados
        st.title("📜 Dados Classificados")
        st.dataframe(df.drop(columns=["Cluster"]))