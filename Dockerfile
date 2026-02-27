FROM python:3.11-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copiar e instalar dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Criar diretórios de dados
RUN mkdir -p data/documents data/images data/vectorstore

EXPOSE 5000

CMD ["python", "backend/app.py"]
