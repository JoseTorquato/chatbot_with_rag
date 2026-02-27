import logging
import os
import shutil
import pickle

import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as FAISSVectorStore
from config import settings

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {'.pdf', '.txt'}


class DocumentProcessor:
    """Camada de Ingestão: parsing de documentos, chunking, embedding e persistência."""

    def __init__(self, documents_folder: str, vectorstore_path: str):
        self.documents_folder = os.path.abspath(documents_folder)
        self.vectorstore_path = os.path.abspath(vectorstore_path)

        os.makedirs(self.documents_folder, exist_ok=True)
        os.makedirs(self.vectorstore_path, exist_ok=True)

        logger.info("DocumentProcessor inicializado. Store: %s", self.vectorstore_path)

        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.vectorstore = None
        self._load_or_create_vectorstore()

    # ------------------------------------------------------------------
    # Persistência do Vector Store
    # ------------------------------------------------------------------

    def _load_or_create_vectorstore(self):
        index_faiss_path = os.path.join(self.vectorstore_path, 'index.faiss')
        index_pkl_path = os.path.join(self.vectorstore_path, 'index.pkl')

        if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
            try:
                logger.info("Carregando vector store existente...")

                with open(index_faiss_path, "rb") as f:
                    index_data = f.read()
                index = faiss.deserialize_index(np.frombuffer(index_data, dtype='uint8'))

                with open(index_pkl_path, "rb") as f:
                    metadata = pickle.load(f)

                self.vectorstore = FAISSVectorStore(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=metadata['docstore'],
                    index_to_docstore_id=metadata['index_to_docstore_id']
                )
                logger.info("Vector store carregado com sucesso.")
            except Exception:
                logger.exception("Erro ao carregar vector store.")
                self.vectorstore = None
        else:
            logger.info("Nenhum vector store encontrado. Será criado no primeiro upload.")

    def _save_vectorstore(self):
        """Serializa o FAISS index + metadados em disco."""
        os.makedirs(self.vectorstore_path, exist_ok=True)

        index_data = faiss.serialize_index(self.vectorstore.index)
        with open(os.path.join(self.vectorstore_path, "index.faiss"), "wb") as f:
            f.write(index_data.tobytes())

        metadata = {
            "docstore": self.vectorstore.docstore,
            "index_to_docstore_id": self.vectorstore.index_to_docstore_id
        }
        with open(os.path.join(self.vectorstore_path, "index.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        logger.info("Vector store salvo em disco.")

    # ------------------------------------------------------------------
    # Processamento de Documentos
    # ------------------------------------------------------------------

    def _get_loader(self, filepath: str):
        """Retorna o loader adequado baseado na extensão do arquivo."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext == '.pdf':
            return PyPDFLoader(filepath)
        elif ext == '.txt':
            return TextLoader(filepath, encoding='utf-8')
        else:
            raise ValueError(f"Formato não suportado: {ext}. Use: {SUPPORTED_EXTENSIONS}")

    def process_document(self, filepath: str):
        """Processa um documento (PDF ou TXT): carrega, divide em chunks e indexa."""
        filepath = os.path.abspath(filepath)
        filename = os.path.basename(filepath)
        logger.info("Processando documento: %s", filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

        # 1. Carregar documento com o loader adequado
        loader = self._get_loader(filepath)
        documents = loader.load()

        # 2. Dividir em chunks semânticos
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        texts = text_splitter.split_documents(documents)

        if not texts:
            raise ValueError(f"Não foi possível extrair texto de: {filename}")

        logger.info("Documento dividido em %d chunks.", len(texts))

        # 3. Gerar embeddings e indexar no FAISS
        try:
            if self.vectorstore is None:
                self.vectorstore = FAISSVectorStore.from_documents(texts, self.embeddings)
            else:
                new_vs = FAISSVectorStore.from_documents(texts, self.embeddings)
                self.vectorstore.merge_from(new_vs)
                # Garantir que o docstore interno está sincronizado
                self.vectorstore.index_to_docstore_id = self.vectorstore.index_to_docstore_id

            # 4. Persistir em disco imediatamente
            self._save_vectorstore()

            logger.info("Documento '%s' indexado com sucesso. Total de chunks: %d", 
                        filename, self.vectorstore.index.ntotal)
        except Exception:
            logger.exception("Erro ao processar documento: %s", filename)
            raise

    # ------------------------------------------------------------------
    # Recuperação (Retriever)
    # ------------------------------------------------------------------

    def get_retriever(self, k: int = 4):
        """Retorna um retriever FAISS para busca por similaridade."""
        if self.vectorstore is None:
            self._load_or_create_vectorstore()
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def clear_vectorstore(self):
        """Remove todos os dados do vector store."""
        self.vectorstore = None
        if os.path.exists(self.vectorstore_path):
            shutil.rmtree(self.vectorstore_path)
        os.makedirs(self.vectorstore_path, exist_ok=True)
        logger.info("Vector store limpo.")
