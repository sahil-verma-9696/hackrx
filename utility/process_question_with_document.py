def process_question_with_document(document_text: str, question: str) -> str:
    """
    Process a single question against the document text
    
    🚨 IMPORTANT: Replace this with your AI/ML logic!
    Current implementation is just a basic demo.
    """
    
    # Simple keyword-based matching (REPLACE WITH YOUR AI LOGIC)
    question_lower = question.lower()
    document_lower = document_text.lower()
    
    # Basic pattern matching examples
    if "grace period" in question_lower:
        sentences = document_text.split('.')
        for sentence in sentences:
            if "grace period" in sentence.lower() and ("thirty" in sentence.lower() or "30" in sentence):
                return sentence.strip()
                
    elif "waiting period" in question_lower and "pre-existing" in question_lower:
        sentences = document_text.split('.')
        for sentence in sentences:
            if "waiting period" in sentence.lower() and ("thirty-six" in sentence.lower() or "36" in sentence):
                return sentence.strip()
                
    elif "maternity" in question_lower:
        sentences = document_text.split('.')
        for sentence in sentences:
            if "maternity" in sentence.lower() and "24" in sentence:
                return sentence.strip()
    
    # Default fallback response
    return f"Based on document analysis, specific information about '{question}' requires detailed policy review."
