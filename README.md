# 🤖 Chatbot Intelligence Pro: RAG + Vision Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-blueviolet.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-red.svg)](https://github.com/facebookresearch/faiss)

Uma plataforma avançada de inteligência artificial que combina **RAG (Retrieval-Augmented Generation)** com **Computer Vision**. Analise documentos PDFs complexos e imagens de forma proativa, com uma interface moderna e fluida.

---

## ✨ Funcionalidades Principais

- 📄 **Análise de PDFs Multifuncional**: Carregue manuais, contratos, livros ou guias e obtenha respostas precisas baseadas no contexto real dos documentos.
- 🖼️ **Visão Computacional (GPT-4 Vision)**: Analise imagens, gráficos e diagramas diretamente no chat.
- 🌊 **Streaming & Typewriter Effect**: Respostas que fluem em tempo real, palavra por palavra, com um efeito visual de digitação premium.
- 🧠 **Memória de Longo Prazo**: Histórico de sessões persistente. Alterne entre conversas sem perder o contexto.
- 🌗 **Modo Dark/Light Dinâmico**: Interface otimizada para produtividade em qualquer ambiente.
- ⚡ **Backend em FastAPI**: Motor assíncrono de alta performance para processamento paralelo de uploads e chat.

## �️ Stack Tecnológica

| Camada | Tecnologias |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Inteligência** | LangChain, OpenAI (GPT-3.5/GPT-4o), HuggingFace Embeddings |
| **Banco Vetorial** | FAISS (Facebook AI Similarity Search) |
| **Frontend** | Vanilla JS, HTML5, CSS3, Marked.js (Markdown), Highlight.js |

---

## 🚀 Instalação Rápida

### 1. Preparação do Ambiente
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/chatbot_with_rag.git
cd chatbot_with_rag

# Crie o ambiente virtual
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 2. Dependências e Configuração
```bash
# Instale os pacotes necessários
pip install -r requirements.txt

# Crie o arquivo de ambiente na raiz
# Adicione: OPENAI_API_KEY=sk-sua-chave
echo "OPENAI_API_KEY=sua_chave_aqui" > .env
```

### 3. Execução
```bash
# Inicie o servidor unificado
python backend/app.py
```
Acesse: **[http://localhost:5000](http://localhost:5000)**

---

## � Arquitetura do Projeto

```text
chatbot_with_rag/
├── backend/
│   ├── app.py           # Servidor FastAPI (API + Static Files)
│   ├── pdf_processor.py # Motor de Embeddings e FAISS
│   ├── chatbot.py       # Lógica RAG e Vision Integration
│   └── image_analyzer.py# Interface com GPT-4 Vision
├── frontend/
│   ├── index.html       # UI moderna e responsiva
│   ├── script.js        # Lógica de streaming e typewriter
│   └── style.css        # Design system e animações
├── data/
│   ├── pdfs/            # Armazenamento físico de documentos
│   ├── images/          # Cache de imagens analisadas
│   └── vectorstore/     # Índices vetoriais persistentes
└── .env                 # Variáveis de ambiente
```

---

## 🧠 Como o RAG funciona aqui?

O sistema utiliza a arquitetura **Retrieve-and-Generate**:
1. **Indexação**: O PDF é quebrado em chunks semânticos usando `RecursiveCharacterTextSplitter`.
2. **Vetores**: Geramos embeddings usando `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace) para suporte a múltiplos idiomas.
3. **Busca**: Quando você pergunta, o FAISS localiza os fragmentos mais relevantes do documento.
4. **Contexto**: Os fragmentos são injetados em um **Prompt Proativo** de alta performance, garantindo que o GPT responda com base nos fatos do arquivo.

---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---
**Desenvolvido com ❤️ por [@josetorquato]**
