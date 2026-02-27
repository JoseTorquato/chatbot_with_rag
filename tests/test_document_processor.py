"""
Testes unitários para a camada de ingestão de documentos.
"""

import os
import tempfile
import pytest

# Adicionar o diretório backend ao path para imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS


class TestSupportedExtensions:
    """Testa as extensões de arquivo suportadas."""

    def test_pdf_is_supported(self):
        assert '.pdf' in SUPPORTED_EXTENSIONS

    def test_txt_is_supported(self):
        assert '.txt' in SUPPORTED_EXTENSIONS

    def test_unsupported_extension(self):
        assert '.docx' not in SUPPORTED_EXTENSIONS
        assert '.csv' not in SUPPORTED_EXTENSIONS


class TestDocumentProcessor:
    """Testa o processamento de documentos."""

    @pytest.fixture
    def processor(self, tmp_path):
        """Cria uma instância do DocumentProcessor com diretórios temporários."""
        docs_folder = tmp_path / "documents"
        vectorstore_path = tmp_path / "vectorstore"
        docs_folder.mkdir()
        vectorstore_path.mkdir()
        return DocumentProcessor(str(docs_folder), str(vectorstore_path))

    def test_initialization(self, processor):
        """Verifica que o processor inicializa corretamente."""
        assert processor.vectorstore is None
        assert processor.embeddings is not None

    def test_get_retriever_without_documents(self, processor):
        """Retriever deve retornar None quando não há documentos indexados."""
        retriever = processor.get_retriever()
        assert retriever is None

    def test_process_txt_document(self, processor):
        """Testa o processamento de um arquivo TXT."""
        # Criar arquivo TXT de teste
        txt_path = os.path.join(processor.documents_folder, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Este é um documento de teste para o sistema RAG. " * 50)

        processor.process_document(txt_path)

        assert processor.vectorstore is not None
        retriever = processor.get_retriever()
        assert retriever is not None

    def test_process_nonexistent_file(self, processor):
        """Deve lançar FileNotFoundError para arquivo inexistente."""
        with pytest.raises(FileNotFoundError):
            processor.process_document("/caminho/inexistente/arquivo.txt")

    def test_process_unsupported_format(self, processor):
        """Deve lançar ValueError para formato não suportado."""
        unsupported_path = os.path.join(processor.documents_folder, "test.docx")
        with open(unsupported_path, "w") as f:
            f.write("conteudo")

        with pytest.raises(ValueError, match="Formato não suportado"):
            processor.process_document(unsupported_path)

    def test_clear_vectorstore(self, processor):
        """Testa a limpeza do vector store."""
        # Indexar documento
        txt_path = os.path.join(processor.documents_folder, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Documento de teste para limpeza. " * 50)

        processor.process_document(txt_path)
        assert processor.vectorstore is not None

        # Limpar
        processor.clear_vectorstore()
        assert processor.vectorstore is None

    def test_retriever_returns_results(self, processor):
        """Testa que o retriever retorna resultados relevantes."""
        txt_path = os.path.join(processor.documents_folder, "test.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(
                "Python é uma linguagem de programação muito popular. "
                "Foi criada por Guido van Rossum em 1991. "
                "Python é usado para web, ciência de dados e inteligência artificial. " * 20
            )

        processor.process_document(txt_path)
        retriever = processor.get_retriever(k=2)
        docs = retriever.invoke("O que é Python?")

        assert len(docs) > 0
        assert "Python" in docs[0].page_content
