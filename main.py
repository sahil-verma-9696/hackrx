from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List
import uvicorn
import time
import os
from datetime import datetime
from dotenv import load_dotenv

from utility.download_pdf import download_pdf
from utility.process_question_with_document import process_question_with_document
from utility.semantic_chunk_text import semantic_chunk_text

# Load environment variables
load_dotenv()

app = FastAPI(
    title="HackRx 6.0 API", 
    version="1.0.0",
    description="HackRx 6.0 Document Question Answering API"
)
security = HTTPBearer()

# Configuration
API_KEY_PREFIX = os.getenv("API_KEY_PREFIX", "hackrx_")
MAX_TIMEOUT = int(os.getenv("MAX_TIMEOUT", "30"))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Models
class HackRxRequest(BaseModel):
    documents: str  # Changed from HttpUrl to str for compatibility
    questions: List[str]

class HackRxResponse(BaseModel):
    answers: List[str]

# Authentication
def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify bearer token"""
    if not credentials.credentials.startswith(API_KEY_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return credentials.credentials

# Main API Endpoints
@app.post("/hackrx/run", response_model=HackRxResponse)
async def hackrx_run(
    request: HackRxRequest,
    token: str = Depends(verify_token)
):
    """Main HackRx endpoint for document Q&A"""
    start_time = time.time()
    
    try:
        # Log request
        if DEBUG:
            print(f"🔍 Processing {len(request.questions)} questions")
            print(f"📄 Document URL: {request.documents}")
        
        # Download and extract document
        document_text = await download_pdf(str(request.documents))
        
        chunks = semantic_chunk_text(document_text, return_text_only=False)

        print(chunks)

        if not document_text:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not extract text from document"
            )
        
        if DEBUG:
            print(f"📊 Document extracted: {len(document_text)} characters")
        
        # Process each question
        answers = []
        for i, question in enumerate(request.questions):
            if DEBUG:
                print(f"❓ Question {i+1}: {question[:60]}...")
            
            answer = process_question_with_document(document_text, question)
            answers.append(answer)
            
            if DEBUG:
                print(f"✅ Answer {i+1}: {answer[:60]}...")
        
        # Log completion
        processing_time = time.time() - start_time
        if DEBUG:
            print(f"⏱️ Completed in {processing_time:.2f} seconds")
        
        return HackRxResponse(answers=chunks)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "HackRx 6.0 API",
        "version": "1.0.0"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 HackRx 6.0 API is running!",
        "endpoints": {
            "main": "/hackrx/run",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "version": "1.0.0"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=DEBUG
    )