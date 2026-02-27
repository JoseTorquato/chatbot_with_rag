# 🤖 Chatbot Intelligence Pro: RAG + Vision Analytics

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1-blueviolet.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/VectorStore-FAISS-red.svg)](https://github.com/facebookresearch/faiss)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

Uma plataforma avançada de inteligência artificial que combina **RAG (Retrieval-Augmented Generation)** com **Computer Vision**. Analise documentos (PDFs e TXTs) e imagens de forma proativa, com uma interface moderna e fluida.

---

## ✨ Funcionalidades Principais

- 📄 **Análise de Documentos**: Carregue PDFs ou arquivos TXT e obtenha respostas precisas baseadas no conteúdo real dos documentos.
- 🖼️ **Visão Computacional (GPT-4 Vision)**: Analise imagens, gráficos e diagramas diretamente no chat.
- 🌊 **Streaming & Typewriter Effect**: Respostas em tempo real com efeito de digitação premium.
- 🧠 **Memória de Sessão**: Histórico persistente por sessão, permitindo alternar entre conversas sem perder contexto.
- 🌗 **Modo Dark/Light**: Interface otimizada para produtividade em qualquer ambiente.
- ⚡ **Backend Assíncrono**: FastAPI com motor de alta performance para processamento paralelo.
- 🧪 **Suíte de Testes**: Testes unitários e de integração garantindo a robustez da aplicação.
- ⚙️ **Configuração Centralizada**: Padrão Pydantic Settings para gestão profissional de ambiente.

---

## 🏗️ Arquitetura do Projeto

```text
chatbot_with_rag/
├── backend/
│   ├── app.py              # Servidor FastAPI — camada de Interface (API REST)
│   ├── document_processor.py # Camada de Ingestão — parsing, chunking e embeddings
│   ├── chatbot.py           # Camada de Geração — orquestração RAG + LLM
│   └── image_analyzer.py    # Módulo de Visão Computacional (GPT-4o)
├── frontend/
│   ├── index.html           # Interface do usuário
│   ├── script.js            # Lógica de streaming e interação
│   └── style.css            # Design system e animações
├── data/
│   ├── documents/           # Armazenamento de PDFs e TXTs
│   ├── images/              # Cache de imagens analisadas
│   └── vectorstore/         # Índices FAISS persistentes
├── Dockerfile               # Containerização
├── docker-compose.yml       # Orquestração de containers
├── requirements.txt         # Dependências Python
├── .env.example             # Template de variáveis de ambiente
└── README.md
```

### Diagrama de Fluxo (RAG Pipeline)

```
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐
│   INTERFACE  │    │    INGESTÃO       │    │  RECUPERAÇÃO     │    │   GERAÇÃO    │
│   (app.py)   │───▶│ (doc_processor)  │───▶│  (FAISS)         │───▶│ (chatbot.py) │
│              │    │                  │    │                  │    │              │
│  FastAPI     │    │ PyPDF / TextLoad │    │ Similarity Search│    │ LangChain +  │
│  REST API    │    │ Chunking (1000)  │    │ Top-K Retrieval  │    │ ChatOpenAI   │
│  + Frontend  │    │ HuggingFace Emb  │    │                  │    │ Streaming    │
└──────────────┘    └──────────────────┘    └──────────────────┘    └──────────────┘
```

---

## 🧠 Decisões Arquiteturais

### 1. Por que FAISS ao invés de Chroma/Pinecone?

**FAISS (Facebook AI Similarity Search)** foi escolhido por:
- **Performance**: Busca por similaridade extremamente rápida, otimizada em C++ com bindings Python.
- **Zero dependências externas**: Roda localmente sem necessidade de servidores ou serviços cloud, simplificando o setup.
- **Persistência simples**: Serialização direta em disco via `faiss.serialize_index`, sem necessidade de banco de dados.
- **Trade-off consciente**: Para produção com milhões de documentos, migraria para Pinecone ou Qdrant (gerenciados e escaláveis).

### 2. Por que HuggingFace Embeddings ao invés de OpenAI Embeddings?

O modelo `paraphrase-multilingual-MiniLM-L12-v2` foi escolhido por:
- **Custo zero**: Roda localmente, sem chamadas de API pagas para cada embedding.
- **Suporte multilíngue**: Nativo para português, inglês e outros idiomas, essencial para documentos em PT-BR.
- **Independência**: Desacopla o embedding do provider de LLM, permitindo trocar o GPT sem reindexar documentos.
- **Trade-off**: Qualidade de embedding ligeiramente inferior ao `text-embedding-3-large` da OpenAI, mas suficiente para o escopo.

### 3. Por que FastAPI ao invés de Flask/Django?

- **Async nativo**: Suporte nativo a `async/await`, essencial para streaming de respostas e uploads simultâneos.
- **Tipagem e validação**: Pydantic integrado para validação automática de payloads (ex: `ChatRequest`).
- **Streaming**: `StreamingResponse` para Server-Sent Events (SSE), viabilizando o efeito typewriter no frontend.
- **Performance**: Um dos frameworks Python mais rápidos, baseado em Starlette e Uvicorn (ASGI).

### 4. Por que LangChain como orquestrador?

- **Abstração de chains**: `ConversationalRetrievalChain` encapsula retrieval + geração + memória em uma única chain.
- **Memória embutida**: `ConversationBufferMemory` gerencia o histórico de conversas por sessão.
- **Ecossistema**: Integração direta com FAISS, OpenAI, HuggingFace e futuros providers.
- **Trade-off**: LangChain adiciona complexidade e overhead. Para produção, avaliaria LlamaIndex ou orquestração manual.

### 5. Por que Streaming (SSE) ao invés de resposta única?

- **UX Premium**: O efeito typewriter dá percepção de velocidade e interatividade, mesmo em respostas longas.
- **Time-to-First-Token**: O usuário vê os primeiros tokens em segundos, em vez de esperar toda a geração.
- **Implementação**: Callback handler customizado (`StreamingCallbackHandler`) com `queue.Queue` para comunicação thread-safe entre a chain e o generator async.

### 6. Separação em Camadas

A arquitetura segue o princípio de **responsabilidade única**:

| Camada | Arquivo | Responsabilidade |
|:---|:---|:---|
| **Interface** | `app.py` | Rotas HTTP, validação de entrada, CORS, servir frontend |
| **Ingestão** | `document_processor.py` | Parsing de PDF/TXT, chunking, embeddings, persistência FAISS |
| **Geração** | `chatbot.py` | Prompt engineering, chain RAG, memória de sessão, streaming |
| **Visão** | `image_analyzer.py` | Análise de imagens via GPT-4 Vision |

Essa separação permite trocar qualquer camada (ex: substituir FAISS por Chroma, ou GPT por Claude) sem impactar as demais.

### 7. Testes Automatizados

O projeto utiliza `pytest` para garantir a qualidade do código em duas frentes:
- **Unitários**: Testam o processamento de documentos, chunking e lógica de vectorstore isoladamente.
- **Integração**: Testam os endpoints da API, payloads, health checks e serving de arquivos estáticos.

---

## 🔧 Stack Tecnológica

| Camada | Tecnologias |
| :--- | :--- |
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **Inteligência** | LangChain, OpenAI (GPT-3.5/GPT-4o), HuggingFace Embeddings |
| **Banco Vetorial** | FAISS (Facebook AI Similarity Search) |
| **Frontend** | Vanilla JS, HTML5, CSS3, Marked.js, Highlight.js |
| **Infra** | Docker, docker-compose |

---

## 🚀 Como Rodar o Projeto

### Opção 1: Execução Local

```bash
# 1. Clone o repositório
git clone https://github.com/JoseTorquato/chatbot_with_rag.git
cd chatbot_with_rag

# 2. Crie e ative o ambiente virtual
python -m venv venv
.\venv\Scripts\activate       # Windows
source venv/bin/activate      # Linux/Mac

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Configure a variável de ambiente
cp .env.example .env
# Edite o .env e adicione sua chave: OPENAI_API_KEY=sk-sua-chave

# 5. Inicie o servidor
python backend/app.py
```

Acesse: **[http://localhost:5000](http://localhost:5000)**

### Opção 2: Docker (Recomendado)

```bash
# 1. Clone e configure
git clone https://github.com/JoseTorquato/chatbot_with_rag.git
cd chatbot_with_rag
cp .env.example .env
# Edite o .env com sua OPENAI_API_KEY

# 2. Build e execução
docker-compose up --build
```

Acesse: **[http://localhost:5000](http://localhost:5000)**

---

## 📡 Endpoints da API

| Método | Rota | Descrição |
|:---|:---|:---|
| `POST` | `/api/upload` | Upload e indexação de PDF ou TXT |
| `POST` | `/api/chat` | Pergunta com streaming (SSE) |
| `GET` | `/api/sessions` | Listar sessões de conversa |
| `GET` | `/api/sessions/{id}` | Histórico de uma sessão |
| `GET` | `/api/documents` | Listar documentos indexados |
| `DELETE` | `/api/documents/{filename}` | Remover documento |
| `POST` | `/api/upload-image` | Upload e análise de imagem |
| `GET` | `/api/images` | Listar imagens |
| `POST` | `/api/clear` | Limpar vectorstore e memória |

---

## 🧠 Como o RAG funciona aqui?

O sistema utiliza a arquitetura **Retrieve-and-Generate** em 4 etapas:

1. **Indexação**: O documento (PDF ou TXT) é quebrado em chunks semânticos de 1000 caracteres com overlap de 200, usando `RecursiveCharacterTextSplitter`.
2. **Embedding**: Cada chunk é transformado em um vetor denso via `paraphrase-multilingual-MiniLM-L12-v2` (HuggingFace), otimizado para múltiplos idiomas.
3. **Busca**: Quando o usuário faz uma pergunta, o FAISS localiza os **top-6 fragmentos** mais similares usando busca por distância vetorial.
4. **Geração**: Os fragmentos relevantes são injetados como contexto em um prompt otimizado, e o GPT gera a resposta com **streaming token-a-token**.

---

## ⚠️ Limitações Atuais

1. **Sem autenticação**: A API é aberta, sem controle de acesso ou rate limiting.
2. **Memória volátil**: O histórico de conversas vive em memória (dict Python). Se o servidor reiniciar, as sessões são perdidas.
3. **Single-worker**: O Uvicorn roda com 1 worker por padrão, limitando concorrência real.
4. **Sem testes automatizados**: Não há suíte de testes unitários ou de integração.
5. **Tamanho de upload**: Sem limite explícito de tamanho de arquivo — PDFs muito grandes podem sobrecarregar a memória.
6. **Modelo fixo**: O modelo LLM (`gpt-3.5-turbo`) está hardcoded. Configurável via variável de ambiente seria mais flexível.
7. **Sem observabilidade**: Não há métricas, tracing ou APM integrados.

---

## 🔮 Próximos Passos para Produção

Se fosse evoluir este projeto para um ambiente produtivo:

### Curto Prazo
- **Autenticação**: JWT ou API Keys para proteger endpoints.
- **Persistência de sessões**: Migrar histórico de conversas para Redis ou banco relacional.
- **Variável de modelo**: Tornar o modelo LLM configurável via `.env` (ex: `LLM_MODEL=gpt-4o`).
- **Testes**: Adicionar testes unitários (pytest) para cada camada + testes de integração para a API.
- **Rate limiting**: Limitar requisições por IP/token para evitar abuso.

### Médio Prazo
- **Abstração de LLM Provider**: Interface/Protocol para trocar entre OpenAI, Anthropic, Ollama sem alterar o código.
- **Vector store gerenciado**: Migrar de FAISS local para Pinecone, Qdrant ou Weaviate para escalar.
- **Observabilidade**: Integrar OpenTelemetry + Langfuse para tracing de chains e métricas.
- **CI/CD**: Pipeline automatizado com lint, testes e deploy (GitHub Actions).
- **Multi-worker**: Configurar Uvicorn com Gunicorn e múltiplos workers.

### Longo Prazo
- **Multi-tenant**: Isolamento de vectorstores e sessões por organização/usuário.
- **Chunking inteligente**: Usar modelos de segmentação semântica ao invés de split por caractere.
- **Re-ranking**: Adicionar etapa de re-ranking (ex: Cohere Rerank) após retrieval para melhorar precisão.
- **Cache de respostas**: Cache semântico para perguntas similares, reduzindo custo de API.
- **Kubernetes**: Orquestração para deploy escalável e resiliente.

---


## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---
**Desenvolvido com ❤️ por [@josetorquato](https://github.com/JoseTorquato)**
