import os
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from src.schemas.response import FinalOutputSchema

class GenerationPipeline:
    def __init__(self, model_name: str = "llama3"):
        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        
        # Initialize the Ollama Chat Model
        self.llm = ChatOllama(
            model=model_name,
            base_url=host,
            temperature=0.0  # 0.0 guarantees deterministic, factual extraction
        )
        
        # Enforce the Pydantic schema
        self.structured_llm = self.llm.with_structured_output(FinalOutputSchema)

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> FinalOutputSchema:
        # 1. Format the retrieved chunks into readable context blocks
        formatted_context = []
        for i, chunk in enumerate(context_chunks, 1):
            source_info = f"Source ID: {chunk.get('id', 'N/A')} | Title: {chunk.get('metadata', {}).get('title', 'N/A')}"
            formatted_context.append(f"--- [Document {i}] ---\n{source_info}\nContent:\n{chunk['text']}")
            
        full_context_str = "\n\n".join(formatted_context)

        # 2. Build the Grounded Prompt
        prompt = f"""You are an expert AI perception and automotive systems researcher.
Answer the user query strictly using the provided context documents below. 
Do not extrapolate or assume details that are not present in the context.

User Query:
{query}

Context:
{full_context_str}
"""
        print("Sending context to LLM for structured extraction...")
        # 3. Invoke LLM and get validated Pydantic model
        result: FinalOutputSchema = self.structured_llm.invoke(prompt)
        return result