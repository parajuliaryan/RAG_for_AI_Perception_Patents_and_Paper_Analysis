import io
import re
import requests
import pymupdf
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)

class PDFParser:
    """
    Downloads and extracts text from PDFs directly into memory.
    """
    
    def extract_text_from_url(self, url: str, doc_id: str, timeout: int = 15) -> Optional[str]:
        """
        Downloads a PDF from a URL, saves it to disk, and extracts its text.
        
        Args:
            url: The URL to the PDF file.
            doc_id: The document ID (used for the filename).
            timeout: Request timeout in seconds.
            
        Returns:
            The extracted text as a string, or None if extraction failed.
        """
        if not url:
            return None
            
        try:
            logger.info(f"Attempting to download and parse PDF from: {url}")
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning(f"URL returned HTML instead of a PDF (likely a landing page): {url}")
                return None
                
            safe_id = doc_id.replace('/', '_')
            
            import src.config as cfg
            pdf_path = cfg.RAW_PDF_DIR / f"{safe_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            logger.debug(f"Saved PDF to disk: {pdf_path}")

            doc = pymupdf.Document(stream=response.content, filetype="pdf")
            
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text())
                
            full_text = "\n".join(text_blocks).strip()
            
            # --- INJECT MARKDOWN HEADERS HERE ---
            full_text = self._format_markdown_headers(full_text)
            
            if not full_text:
                logger.warning(f"Successfully downloaded PDF, but extracted text was empty: {url}")
                return None
                
            logger.debug(f"Successfully extracted {len(full_text)} characters from {url}")
            return full_text
            
        except requests.HTTPError as e:
            logger.warning(f"HTTP error downloading PDF from {url}: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Failed to parse PDF from {url}: {e}")
            return None

    def _format_markdown_headers(self, text: str) -> str:
        """
        Uses heuristics to convert academic section titles into Markdown headers.
        """
        lines = text.split('\n')
        formatted_lines = []
        
        exact_headers = {
            "abstract", "introduction", "methodology", "methods", "results", 
            "discussion", "conclusion", "conclusions", "references", "background", "related work"
        }
        
        # Regex to catch "1 Introduction", "1. Introduction", "I. INTRODUCTION", "A. BACKGROUND"
        header_pattern = re.compile(r'^(?:(?:I{1,3}|IV|V|VI{1,3}|IX|X|[A-Z]|\d+)\.?\s+)?([A-Z][a-zA-Z\s]+)$')
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append(line)
                continue
                
            # If the line is just a stray number (like '1'), skip it and append,
            # because the next line might be the actual title ("Introduction").
            if stripped.isdigit() and len(stripped) < 3:
                formatted_lines.append(line)
                continue
                
            is_all_caps = stripped.isupper() and len(stripped) > 3
            is_exact = stripped.lower() in exact_headers
            match = header_pattern.match(stripped)
            
            if len(stripped) < 60:
                if is_all_caps or is_exact or match:
                    clean_title = match.group(1).title() if match else stripped.title()
                    
                    if not stripped.startswith('#'):
                        formatted_lines.append(f"\n# {clean_title}\n")
                        continue
                        
            formatted_lines.append(line)
            
        return "\n".join(formatted_lines)
