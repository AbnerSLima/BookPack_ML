# 🎒 BookPack ML

BookPack ML é um projeto de **Machine Learning** que analisa e prevê o peso de mochilas com base na quantidade de livros carregados. Ele utiliza dois modelos de **Aprendizado de Máquina**:

## ✅ Modelos Utilizados

### 📊 **Modelo Supervisionado**
Usando **Regressão Linear Simples**, este modelo aprende a relação entre a **quantidade de livros** e o **peso total da mochila**. Com isso, é possível estimar **quanto uma mochila irá pesar** com base no número de livros transportados.

### 🔍 **Modelo Não Supervisionado**
Através do algoritmo de **K-Means**, o modelo classifica automaticamente as mochilas em **três categorias**:
- 🟢 **Leve** → Mochilas de peso reduzido.
- 🟡 **Média** → Mochilas com peso intermediário.
- 🔴 **Pesada** → Mochilas mais carregadas.

Dessa forma, mesmo sem informações pré-definidas, o sistema identifica padrões e agrupa os dados de forma inteligente.

## 📚 **Como Rodar o Projeto?**

### 🔧 **1. Instale as Dependências**
Antes de rodar o projeto, certifique-se de ter o Python instalado e execute os seguintes comandos:

```bash
# Clone este repositório
git clone https://github.com/AbnerSLima/BookPack_ML.git

# Acesse a pasta do projeto
cd BookPack_ML

# Crie e ative o ambiente virtual (Windows)
python -m venv venv
venv\Scripts\activate

# (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Instale as dependências
pip install numpy pandas streamlit matplotlib scikit-learn
```

### 🔧 **2. Execute o Projeto**
Após instalar as dependências, rode o seguinte comando:

```bash
streamlit run main.py
```

Agora você pode acessar a aplicação no navegador pelo link:

📎 http://localhost:8501

## 🛠 Tecnologias Utilizadas

- 🐍 **Python**  
- 📊 **Streamlit** (Interface gráfica)  
- 🧮 **NumPy & Pandas** (Manipulação de dados)  
- 🤖 **Scikit-Learn** (Machine Learning)  
- 📈 **Matplotlib** (Gráficos)  

---

Feito com ❤️ por [Abner Silva](https://github.com/AbnerSLima)