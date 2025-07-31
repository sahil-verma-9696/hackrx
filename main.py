from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import time
import os
import requests
import PyPDF2
import io
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from urllib.parse import urlparse
import re

# Load environment variables
load_dotenv()

app = FastAPI(
    title="PDF Q&A API with Gemini", 
    version="1.0.0",
    description="PDF Question Answering API using Google Gemini AI"
)
security = HTTPBearer()

# Configuration
API_KEY_PREFIX = os.getenv("API_KEY_PREFIX", "hackrx_")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Your provided API key
MAX_TIMEOUT = int(os.getenv("MAX_TIMEOUT", "30"))
DEBUG = os.getenv("DEBUG", "True").lower() == "true"

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# Models
class PDFRequest(BaseModel):
    documents: str
    questions: List[str]

class PDFResponse(BaseModel):
    answers: List[str]
    processing_time: float
    document_length: int

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str
    gemini_configured: bool

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if not credentials.credentials.startswith(API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Utility functions
def is_valid_url(url: str) -> bool:
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def download_pdf_from_url(url: str) -> bytes:
    """Download PDF from URL"""
    try:
        if not is_valid_url(url):
            raise ValueError("Invalid URL format")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=MAX_TIMEOUT)
        response.raise_for_status()
        
        # Check if content type is PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type and not url.lower().endswith('.pdf'):
            # Try to detect PDF by content
            if not response.content.startswith(b'%PDF'):
                raise ValueError("URL does not point to a valid PDF file")
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to download PDF: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing PDF URL: {str(e)}"
        )

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF")
        
        # Clean up the text
        text = re.sub(r'\n+', '\n', text)  # Multiple newlines to single
        text = re.sub(r'\s+', ' ', text)   # Multiple spaces to single
        
        return text.strip()
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to extract text from PDF: {str(e)}"
        )

def chunk_text(text: str, max_chunk_size: int = 4000) -> List[str]:
    """Split text into manageable chunks"""
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    sentences = text.split('. ')
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk + sentence + '. ') <= max_chunk_size:
            current_chunk += sentence + '. '
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def answer_question_with_gemini(question: str, context: str) -> str:
    """Use Gemini to answer question based on context"""
    try:
        prompt = f"""
    Based on the following document content, please answer the question accurately and concisely.

    Document Content:
    {context}

    Question: {question}

    Instructions:
    - Provide a clear, direct answer based only on the information in the document
    - If the answer is not found in the document, say "The information is not available in the provided document"
    - Keep the answer concise but complete
    - Use specific details from the document when possible

    Answer:"""

        response = model.generate_content(prompt)
        
        if response and response.text:
            return response.text.strip()
        else:
            return "Unable to generate answer due to API response issue"
            
    except Exception as e:
        if DEBUG:
            print(f"Gemini API Error: {str(e)}")
        return f"Error processing question with AI: {str(e)}"

def find_relevant_chunks(question: str, chunks: List[str], max_chunks: int = 3) -> str:
    """Find most relevant chunks for the question (simple keyword matching)"""
    if not chunks:
        return ""
    
    if len(chunks) == 1:
        return chunks[0]
    
    # Simple relevance scoring based on keyword overlap
    question_words = set(question.lower().split())
    chunk_scores = []
    
    for i, chunk in enumerate(chunks):
        chunk_words = set(chunk.lower().split())
        overlap = len(question_words.intersection(chunk_words))
        chunk_scores.append((i, overlap, chunk))
    
    # Sort by relevance score
    chunk_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top chunks
    relevant_chunks = [chunk for _, _, chunk in chunk_scores[:max_chunks]]
    
    return "\n\n".join(relevant_chunks)

# API Endpoints
@app.post("/hackrx/run", response_model=PDFResponse)
async def pdf_question_answering(
    request: PDFRequest,
    token: str = Depends(verify_token)
):
    """Main endpoint for PDF Question Answering"""
    start_time = time.time()
    
    try:
        if DEBUG:
            print(f"🔍 Processing {len(request.questions)} questions")
            print(f"📄 PDF URL: {request.documents}")
        
        # Download PDF
        pdf_content = download_pdf_from_url(request.documents)
        
        # Extract text
        document_text = extract_text_from_pdf(pdf_content)
        
        if DEBUG:
            print(f"📊 Document extracted: {len(document_text)} characters")
        
        # Split into chunks for better processing
        chunks = chunk_text(document_text)
        
        if DEBUG:
            print(f"📑 Document split into {len(chunks)} chunks")
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            if DEBUG:
                print(f"❓ Question {i+1}: {question[:60]}...")
            
            # Find relevant context
            relevant_context = find_relevant_chunks(question, chunks)
            
            # Get answer from Gemini
            answer = answer_question_with_gemini(question, relevant_context)
            answers.append(answer)
            
            if DEBUG:
                print(f"✅ Answer {i+1}: {answer[:60]}...")
        
        processing_time = time.time() - start_time
        
        if DEBUG:
            print(f"⏱️ Completed in {processing_time:.2f} seconds")
        
        return PDFResponse(
            answers=answers,
            processing_time=round(processing_time, 2),
            document_length=len(document_text)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/pdf/text")
async def extract_pdf_text(
    request: dict,
    token: str = Depends(verify_token)
):
    """Extract text from PDF URL"""
    try:
        pdf_url = request.get("documents")
        if not pdf_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="documents is required"
            )
        
        # Download and extract
        pdf_content = download_pdf_from_url(pdf_url)
        document_text = extract_text_from_pdf(pdf_content)
        
        chunks = chunk_text(document_text)
        
        return {
            "text": document_text,
            "length": len(document_text),
            "chunks": len(chunks),
            "preview": document_text[:500] + "..." if len(document_text) > 500 else document_text
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error extracting PDF text: {str(e)}"
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        service="PDF Q&A API with Gemini",
        version="1.0.0",
        gemini_configured=bool(GEMINI_API_KEY)
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 PDF Q&A API with Google Gemini is running!",
        "endpoints": {
            "main": "/pdf/qa",
            "extract": "/pdf/extract", 
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "version": "1.0.0",
        "features": [
            "PDF Text Extraction",
            "Google Gemini AI Integration",
            "Multi-Question Processing",
            "Automatic Text Chunking",
            "Bearer Token Authentication"
        ],
        "usage": {
            "authentication": "Bearer token with prefix 'hackrx_'",
            "main_endpoint": "POST /pdf/qa",
            "request_format": {
                "documents": "URL to PDF file",
                "questions": ["Array of questions"]
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    print(f"🚀 Starting PDF Q&A API with Gemini on port {port}")
    print(f"🔑 Gemini API configured: {bool(GEMINI_API_KEY)}")
    print(f"🐛 Debug mode: {DEBUG}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=DEBUG
    )