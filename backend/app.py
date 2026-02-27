import logging
import os
import shutil
import time

from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import uvicorn

from config import settings
from document_processor import DocumentProcessor, SUPPORTED_EXTENSIONS
from chatbot import Chatbot
from image_analyzer import ImageAnalyzer

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Inicialização ---
START_TIME = time.time()

app = FastAPI(
    title="Chatbot with RAG API",
    description="API REST para chatbot com RAG (Retrieval-Augmented Generation) e análise de imagens.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura exceções não tratadas e retorna JSON padronizado."""
    logger.exception("Erro não tratado em %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Erro interno do servidor",
            "detail": str(exc) if not isinstance(exc, HTTPException) else exc.detail
        }
    )


# --- Caminhos e instâncias ---
os.makedirs(settings.documents_folder, exist_ok=True)
os.makedirs(settings.images_folder, exist_ok=True)
os.makedirs(settings.vectorstore_path, exist_ok=True)

doc_processor = DocumentProcessor(settings.documents_folder, settings.vectorstore_path)
chatbot = Chatbot(doc_processor)
image_analyzer = ImageAnalyzer()


# --- Modelos Pydantic (Request & Response) ---
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

class MessageResponse(BaseModel):
    message: str

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    vectorstore_loaded: bool
    documents_count: int
    model: str


# =====================================================================
# ROTAS DA API
# =====================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check: status do serviço, uptime e estado do vectorstore."""
    doc_count = len([
        f for f in os.listdir(settings.documents_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]) if os.path.exists(settings.documents_folder) else 0

    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime_seconds=round(time.time() - START_TIME, 1),
        vectorstore_loaded=doc_processor.vectorstore is not None,
        documents_count=doc_count,
        model=settings.llm_model
    )


@app.post("/api/upload", response_model=MessageResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload e indexação de documento (PDF ou TXT)."""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Formato não suportado: {ext}. Formatos aceitos: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # Verificar tamanho do arquivo
    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande ({size_mb:.1f}MB). Máximo: {settings.max_upload_size_mb}MB"
        )

    filepath = os.path.join(settings.documents_folder, file.filename)

    try:
        with open(filepath, "wb") as buffer:
            buffer.write(contents)

        doc_processor.process_document(filepath)

        logger.info("Documento '%s' processado (%.1fMB).", file.filename, size_mb)
        return MessageResponse(message=f"Documento '{file.filename}' processado com sucesso!")

    except Exception as e:
        logger.exception("Erro no processamento de '%s'.", file.filename)
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    """Enviar pergunta e receber resposta via streaming (SSE)."""
    if not request.question:
        raise HTTPException(status_code=400, detail="Pergunta não fornecida")

    logger.info("Chat | session=%s | pergunta='%s'", request.session_id, request.question[:80])

    async def event_generator():
        for token in chatbot.get_response_stream(request.question, request.session_id):
            yield token

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Content-Type-Options": "nosniff"
        }
    )


@app.get("/api/sessions")
async def list_sessions():
    """Listar todas as sessões de conversa."""
    sessions = []
    for sid in chatbot.memories.keys():
        memory = chatbot.memories[sid]
        vars = memory.load_memory_variables({})
        history = vars.get("chat_history", [])
        title = history[0].content[:30] + "..." if history else "Nova Conversa"
        sessions.append({'id': sid, 'title': title})
    return {"sessions": sessions}


@app.get("/api/sessions/{session_id}")
async def get_session_history(session_id: str):
    """Retornar histórico de uma sessão específica."""
    history = chatbot.get_history(session_id)
    return {"history": history}


@app.get("/api/documents")
async def list_documents():
    """Listar documentos indexados (PDFs e TXTs)."""
    docs = [
        f for f in os.listdir(settings.documents_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return {"documents": docs}


@app.get("/api/pdfs")
async def list_pdfs():
    """Retrocompatibilidade: listar documentos."""
    docs = [
        f for f in os.listdir(settings.documents_folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    return {"pdfs": docs}


@app.delete("/api/documents/{filename}", response_model=MessageResponse)
async def delete_document(filename: str):
    """Remover um documento."""
    filepath = os.path.join(settings.documents_folder, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        logger.info("Documento '%s' removido.", filename)
        return MessageResponse(message=f"Documento '{filename}' deletado.")
    raise HTTPException(status_code=404, detail="Arquivo não encontrado")


@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    """Retrocompatibilidade: remover via rota antiga."""
    return await delete_document(filename)


@app.get("/api/images")
async def list_images():
    """Listar imagens analisadas."""
    images = [f for f in os.listdir(settings.images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    return {"images": images}


@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """Upload e análise de imagem via GPT-4 Vision."""
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Formato de imagem inválido")

    filepath = os.path.join(settings.images_folder, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    analysis = image_analyzer.analyze_image(filepath)
    chatbot.set_last_image(filepath)
    logger.info("Imagem '%s' analisada.", file.filename)
    return {"analysis": analysis, "filename": file.filename}


@app.post("/api/clear", response_model=MessageResponse)
async def clear_all():
    """Limpar vector store e memória de sessões."""
    doc_processor.clear_vectorstore()
    chatbot.clear_memory()
    logger.info("Banco de dados e memória limpos.")
    return MessageResponse(message="Banco de dados limpo!")


# =====================================================================
# SERVIR FRONTEND
# =====================================================================

app.mount("/static", StaticFiles(directory=settings.frontend_path), name="static")


@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(settings.frontend_path, "index.html"))


@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    local_file = os.path.join(settings.frontend_path, full_path)
    if os.path.exists(local_file) and os.path.isfile(local_file):
        return FileResponse(local_file)
    return FileResponse(os.path.join(settings.frontend_path, "index.html"))


if __name__ == "__main__":
    logger.info("Iniciando servidor em %s:%d...", settings.host, settings.port)
    uvicorn.run(app, host=settings.host, port=settings.port)
