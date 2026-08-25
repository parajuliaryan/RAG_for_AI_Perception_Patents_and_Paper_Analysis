import io
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
            
            # Use a standard browser User-Agent to avoid basic blocking
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            # A lot of DOIs redirect to HTML pages instead of PDFs. Check the content type.
            content_type = response.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                logger.warning(f"URL returned HTML instead of a PDF (likely a landing page): {url}")
                return None
                
            # Safe filename
            safe_id = doc_id.replace('/', '_')
            
            # Save PDF to disk
            import src.config as cfg
            pdf_path = cfg.RAW_PDF_DIR / f"{safe_id}.pdf"
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            logger.debug(f"Saved PDF to disk: {pdf_path}")

            # Load the PDF from bytes in memory for extraction
            doc = pymupdf.Document(stream=response.content, filetype="pdf")
            
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text())
                
            full_text = "\n".join(text_blocks).strip()
            
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

