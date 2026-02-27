"""
Configuração centralizada da aplicação usando Pydantic Settings.

Todas as variáveis de ambiente são validadas e tipadas aqui.
Para trocar de LLM ou ajustar parâmetros, basta alterar o .env.
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Configurações da aplicação carregadas de variáveis de ambiente."""

    # --- OpenAI ---
    openai_api_key: str = Field(..., description="Chave de API da OpenAI")

    # --- LLM ---
    llm_model: str = Field(default="gpt-3.5-turbo", description="Modelo LLM para chat (ex: gpt-3.5-turbo, gpt-4o)")
    llm_temperature: float = Field(default=0.3, description="Temperatura do LLM (0.0 = determinístico, 1.0 = criativo)")
    llm_max_tokens: int = Field(default=1500, description="Máximo de tokens por resposta")

    # --- Vision ---
    vision_model: str = Field(default="gpt-4o", description="Modelo para análise de imagens")

    # --- Embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Modelo de embeddings HuggingFace"
    )

    # --- RAG ---
    chunk_size: int = Field(default=1000, description="Tamanho dos chunks de texto")
    chunk_overlap: int = Field(default=200, description="Overlap entre chunks")
    retriever_k: int = Field(default=6, description="Número de documentos retornados pelo retriever")

    # --- Server ---
    host: str = Field(default="0.0.0.0", description="Host do servidor")
    port: int = Field(default=5000, description="Porta do servidor")
    max_upload_size_mb: int = Field(default=50, description="Tamanho máximo de upload em MB")

    # --- Paths ---
    base_dir: str = Field(
        default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )

    @property
    def documents_folder(self) -> str:
        return os.path.join(self.base_dir, 'data', 'documents')

    @property
    def images_folder(self) -> str:
        return os.path.join(self.base_dir, 'data', 'images')

    @property
    def vectorstore_path(self) -> str:
        return os.path.join(self.base_dir, 'data', 'vectorstore')

    @property
    def frontend_path(self) -> str:
        return os.path.join(self.base_dir, 'frontend')

    model_config = {
        "env_file": os.path.join(os.path.dirname(__file__), ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — importar de qualquer módulo
settings = Settings()
