from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
import numpy as np
import os
import shutil
import pickle

class PDFProcessor:
    def __init__(self, pdf_folder, vectorstore_path):
        self.pdf_folder = os.path.abspath(pdf_folder)
        self.vectorstore_path = os.path.abspath(vectorstore_path)
        
        os.makedirs(self.pdf_folder, exist_ok=True)
        os.makedirs(self.vectorstore_path, exist_ok=True)
        
        print(f"🛠️ PDFProcessor inicializado.")
        print(f"   Store Path: {self.vectorstore_path}")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.vectorstore = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        index_faiss_path = os.path.join(self.vectorstore_path, 'index.faiss')
        index_pkl_path = os.path.join(self.vectorstore_path, 'index.pkl')
        
        if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
            try:
                print(f"📂 Carregando vector store via buffer (Unicode safe)...")
                
                # Carregar o index via buffer para evitar erro de encoding no C++ do FAISS
                with open(index_faiss_path, "rb") as f:
                    index_data = f.read()
                index = faiss.deserialize_index(np.frombuffer(index_data, dtype='uint8'))
                
                # Carregar o docstore e metadados
                with open(index_pkl_path, "rb") as f:
                    metadata = pickle.load(f)
                
                # Reconstruir o objeto FAISS do LangChain
                self.vectorstore = FAISS(
                    embedding_function=self.embeddings,
                    index=index,
                    docstore=metadata['docstore'],
                    index_to_docstore_id=metadata['index_to_docstore_id']
                )
                print("✅ Vector store carregado com sucesso!")
            except Exception as e:
                print(f"❌ Erro ao carregar vector store: {e}")
                self.vectorstore = None
        else:
            print(f"ℹ️ Nenhum vector store encontrado.")

    def process_pdf(self, pdf_path):
        pdf_path = os.path.abspath(pdf_path)
        print(f"📄 Processando: {os.path.basename(pdf_path)}")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        if not texts:
            raise ValueError("Não foi possível extrair texto do PDF.")

        try:
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(texts, self.embeddings)
            else:
                new_vs = FAISS.from_documents(texts, self.embeddings)
                self.vectorstore.merge_from(new_vs)

            # Garantir pasta
            os.makedirs(self.vectorstore_path, exist_ok=True)
            
            # 1. Salvar o INDEX via serialize
            index_data = faiss.serialize_index(self.vectorstore.index)
            with open(os.path.join(self.vectorstore_path, "index.faiss"), "wb") as f:
                f.write(index_data.tobytes())
                
            # 2. Salvar os metadados (PKL)
            metadata = {
                "docstore": self.vectorstore.docstore,
                "index_to_docstore_id": self.vectorstore.index_to_docstore_id
            }
            with open(os.path.join(self.vectorstore_path, "index.pkl"), "wb") as f:
                pickle.dump(metadata, f)

            print(f"✅ Documento indexado com sucesso!")

        except Exception as e:
            print(f"❌ Erro no processamento: {e}")
            raise e

    def clear_vectorstore(self):
        self.vectorstore = None
        if os.path.exists(self.vectorstore_path):
            shutil.rmtree(self.vectorstore_path)
        os.makedirs(self.vectorstore_path, exist_ok=True)

    def get_retriever(self, k=4):
        if self.vectorstore is None:
            self._load_or_create_vectorstore()
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": k})
