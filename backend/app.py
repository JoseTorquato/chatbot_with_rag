import os
import shutil
import base64
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import uvicorn
import datetime

from pdf_processor import PDFProcessor
from chatbot import Chatbot
from image_analyzer import ImageAnalyzer

# Inicialização
app = FastAPI(title="Chatbot with RAG API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_FOLDER = os.path.join(BASE_DIR, 'data', 'pdfs')
IMAGE_FOLDER = os.path.join(BASE_DIR, 'data', 'images')
VECTORSTORE_PATH = os.path.join(BASE_DIR, 'data', 'vectorstore')

os.makedirs(PDF_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(VECTORSTORE_PATH, exist_ok=True)

# Instâncias
pdf_processor = PDFProcessor(PDF_FOLDER, VECTORSTORE_PATH)
chatbot = Chatbot(pdf_processor)
image_analyzer = ImageAnalyzer()

# Tipos de dados (Pydantic)
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"

# --- ROTAS DA API ---

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são permitidos")

    filepath = os.path.join(PDF_FOLDER, file.filename)
    
    try:
        # Salva o arquivo de forma assíncrona
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Processamento (RAG)
        pdf_processor.process_pdf(filepath)
        
        return {"message": f"PDF {file.filename} processado com sucesso!"}
    except Exception as e:
        print(f"❌ Erro no processamento: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    if not request.question:
        raise HTTPException(status_code=400, detail="Pergunta não fornecida")
    
    # Gerador para Streaming
    async def event_generator():
        for token in chatbot.get_response_stream(request.question, request.session_id):
            # Log apenas para debug no terminal
            # print(f"📤 Token enviado: {token[:20]}...") 
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
    history = chatbot.get_history(session_id)
    return {"history": history}

@app.get("/api/pdfs")
async def list_pdfs():
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
    return {"pdfs": pdfs}

@app.delete("/api/pdfs/{filename}")
async def delete_pdf(filename: str):
    filepath = os.path.join(PDF_FOLDER, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return {"message": f"PDF {filename} deletado."}
    raise HTTPException(status_code=404, detail="Arquivo não encontrado")

@app.get("/api/images")
async def list_images():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    return {"images": images}

@app.post("/api/upload-image")
async def upload_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
        raise HTTPException(status_code=400, detail="Formato de imagem inválido")

    filepath = os.path.join(IMAGE_FOLDER, file.filename)
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    analysis = image_analyzer.analyze_image(filepath)
    chatbot.set_last_image(filepath)
    return {"analysis": analysis, "filename": file.filename}

@app.post("/api/clear")
async def clear_all():
    pdf_processor.clear_vectorstore()
    chatbot.clear_memory()
    return {"message": "Banco de dados limpo!"}

# --- SERVIR FRONTEND ---

# Monta a pasta de arquivos estáticos (CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "frontend")), name="static")

@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

# Fallback para o index.html em rotas não encontradas (útil para SPAs)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    # Se o arquivo existir na pasta frontend, serve ele (ex: style.css)
    local_file = os.path.join(BASE_DIR, "frontend", full_path)
    if os.path.exists(local_file) and os.path.isfile(local_file):
        return FileResponse(local_file)
    # Caso contrário, volta para o index.html
    return FileResponse(os.path.join(BASE_DIR, "frontend", "index.html"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
