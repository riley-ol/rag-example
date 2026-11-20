import os
import uuid
from typing import Dict, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from .model import pdf_chatbot

load_dotenv(dotenv_path=".env", override=True)  # .env (riley-demo) wins over shell vars

app = FastAPI(
    title="PDF RAG Chatbot",
    description="A RAG chatbot that answers questions about uploaded PDFs using LangChain and OpenAI",
    version="1.0.0",
)

# Add session middleware
app.add_middleware(
    SessionMiddleware, 
    secret_key=os.environ.get('SECRET_KEY', 'your-secret-key-here')
)

# Mount static files and templates
current_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(current_dir, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))

# Folder to store uploaded files temporarily
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Store chatbot instances per session
chatbots: Dict[str, pdf_chatbot.PDFChatbotModel] = {}


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    sources: Optional[list[str]] = None


class ErrorResponse(BaseModel):
    error: str


def get_or_create_chatbot(session_id: str) -> pdf_chatbot.PDFChatbotModel:
    """Get existing chatbot for session or create a new one."""
    if session_id not in chatbots:
        chatbots[session_id] = pdf_chatbot.PDFChatbotModel()
    return chatbots[session_id]


@app.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Serve the upload page."""
    return templates.TemplateResponse(request, "upload.html", {"request": request})


@app.post("/upload-file")
async def upload_file(
    request: Request,
    file: UploadFile = File(...)
):
    """Handle file upload and process PDF."""
    # Initialize session if not exists
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    session_id = request.session['session_id']
    chatbot = get_or_create_chatbot(session_id)
    
    if not file.filename or not file.filename.endswith('.pdf'):
        return templates.TemplateResponse(
            request,
            "upload.html", 
            {"request": request, "error": "Please upload a valid PDF file"}
        )
    
    try:
        # Save the file with a unique name
        filename = str(uuid.uuid4()) + ".pdf"
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Load the PDF into the chatbot
        chunks = chatbot.load_pdf(file_path)
        request.session['current_pdf'] = file_path
        request.session['pdf_filename'] = file.filename
        
        return templates.TemplateResponse(
            request,
            "chat.html", 
            {
                "request": request,
                "pdf_loaded": True, 
                "filename": file.filename,
                "chunk_count": len(chunks)
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "upload.html", 
            {"request": request, "error": f"Error loading PDF: {str(e)}"}
        )


@app.post("/upload-sample")
async def upload_sample(
    request: Request,
    sample_pdf: str = Form(...)
):
    """Handle sample PDF selection."""
    # Initialize session if not exists
    if 'session_id' not in request.session:
        request.session['session_id'] = str(uuid.uuid4())
    
    session_id = request.session['session_id']
    chatbot = get_or_create_chatbot(session_id)
    
    try:
        chunks = chatbot.load_pdf(sample_pdf)
        request.session['current_pdf'] = sample_pdf
        request.session['pdf_filename'] = os.path.basename(sample_pdf)
        
        return templates.TemplateResponse(
            request,
            "chat.html", 
            {
                "request": request,
                "pdf_loaded": True, 
                "filename": os.path.basename(sample_pdf),
                "chunk_count": len(chunks)
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            request,
            "upload.html", 
            {"request": request, "error": f"Error loading PDF: {str(e)}"}
        )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest):
    """Handle chat messages."""
    if 'session_id' not in request.session:
        raise HTTPException(status_code=400, detail="No session found")
    
    session_id = request.session['session_id']
    
    if session_id not in chatbots:
        raise HTTPException(status_code=400, detail="No PDF loaded")
    
    chatbot = chatbots[session_id]
    question = chat_request.question.strip()
    
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    
    try:
        response = chatbot.ask_question(question)
        return ChatResponse(
            answer=response.answer,
            sources=response.sources
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.get("/uploads/{filename}")
async def get_uploaded_file(filename: str):
    """Serve uploaded files."""
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=filename
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "PDF RAG Chatbot"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, log_level="info")
