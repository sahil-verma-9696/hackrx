import os
import httpx
import pypdf
import io
from fastapi import FastAPI, HTTPException, Depends, status


MAX_TIMEOUT = int(os.getenv("MAX_TIMEOUT", "30"))


async def download_pdf(url: str) -> str:
    """Download and extract text from PDF"""
    try:
        async with httpx.AsyncClient(timeout=MAX_TIMEOUT) as client:
            response = await client.get(str(url))
            response.raise_for_status()
            
            # Extract text from PDF
            pdf_content = io.BytesIO(response.content)
            pdf_reader = pypdf.PdfReader(pdf_content)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error downloading/processing PDF: {str(e)}"
        )
