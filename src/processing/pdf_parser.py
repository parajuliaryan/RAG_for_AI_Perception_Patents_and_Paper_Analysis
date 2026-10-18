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
    
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """
        Loads a PDF from a local file path and extracts its text with markdown formatting.
        """
        try:
            logger.info(f"Parsing local PDF from: {file_path}")
            doc = pymupdf.Document(file_path)
            
            text_blocks = []
            for page in doc:
                text_blocks.append(self._extract_page_text(page))
                
            full_text = "\n".join(text_blocks).strip()
            full_text = self._normalize_ligatures(full_text)
            full_text = self._format_markdown_headers(full_text)
            
            if not full_text:
                logger.warning(f"Successfully loaded local PDF, but extracted text was empty: {file_path}")
                return None
                
            logger.debug(f"Successfully extracted {len(full_text)} characters from {file_path}")
            return full_text
            
        except Exception as e:
            logger.warning(f"Failed to parse local PDF from {file_path}: {e}")
            return None

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
                text_blocks.append(self._extract_page_text(page))
                
            full_text = "\n".join(text_blocks).strip()
            full_text = self._normalize_ligatures(full_text)
            
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

    def _extract_page_text(self, page) -> str:
        """
        Extracts text from a PDF page using block coordinates to prevent two-column
        interleaving (e.g., on two-column patent cover pages and academic papers).
        """
        try:
            blocks = page.get_text("blocks")
        except Exception:
            return page.get_text()

        text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0 and b[4].strip()]
        if not text_blocks:
            return ""

        page_width = page.rect.width if page.rect.width > 0 else 612.0
        left_bound = page_width * 0.45
        right_bound = page_width * 0.55

        # Group consecutive non-spanning blocks so left-column blocks precede right-column blocks
        ordered_texts = []
        col_group = []

        def flush_col_group(group):
            if not group:
                return
            left_col = []
            right_col = []
            for b in group:
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                x_center = (x0 + x1) / 2.0
                if x_center < page_width * 0.5:
                    left_col.append((y0, x0, txt))
                else:
                    right_col.append((y0, x0, txt))
            left_col.sort(key=lambda item: (round(item[0], 1), item[1]))
            right_col.sort(key=lambda item: (round(item[0], 1), item[1]))
            for _, _, txt in left_col:
                ordered_texts.append(txt.rstrip("\n"))
            for _, _, txt in right_col:
                ordered_texts.append(txt.rstrip("\n"))

        # First sort all blocks roughly top-to-bottom so full-width dividers act as band boundaries
        for b in text_blocks:
            x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
            is_spanning = (x0 < left_bound) and (x1 > right_bound)
            if is_spanning:
                flush_col_group(col_group)
                col_group = []
                ordered_texts.append(txt.rstrip("\n"))
            else:
                col_group.append(b)

        flush_col_group(col_group)
        return "\n".join(ordered_texts)

    @staticmethod
    def _normalize_ligatures(text: str) -> str:
        """Replaces common PDF Unicode ligatures with standard ASCII characters."""
        ligatures = {
            "\ufb00": "ff",
            "\ufb01": "fi",
            "\ufb02": "fl",
            "\ufb03": "ffi",
            "\ufb04": "ffl",
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)
        return text

    @staticmethod
    def _collapse_spaced_header(line: str) -> str:
        """
        Normalizes spaced-out all-caps headers common in Elsevier/academic PDFs,
        e.g. 'A B S T R A C T' -> 'ABSTRACT', 'A R T I C L E   I N F O' -> 'ARTICLE INFO'.
        """
        stripped = line.strip()
        if re.match(r'^[A-Z](?:\s+[A-Z]){4,}$', stripped):
            # Split by 2+ spaces to preserve word boundaries if present
            words = re.split(r'\s{2,}', stripped)
            collapsed = " ".join(re.sub(r'\s+', '', w) for w in words)
            if collapsed == "ARTICLEINFO":
                return "ARTICLE INFO"
            return collapsed
        return stripped

    def _format_markdown_headers(self, text: str) -> str:
        """
        Uses heuristics to convert academic and patent section titles into Markdown headers
        without corrupting normal prose lines inside abstracts or paragraphs.
        """
        lines = text.split('\n')
        formatted_lines = []
        
        exact_headers = {
            # Academic Paper Headers
            "abstract", "introduction", "methodology", "methods", "method", "results", 
            "discussion", "conclusion", "conclusions", "references", "background",
            "related work", "related works", "experiments", "experimental setup",
            "evaluation", "limitations", "future work", "acknowledgments",
            "acknowledgements", "motivation", "keywords", "index terms",
            "article info", "history", "system overview", "problem statement",
            # Tech Patent Headers
            "claims", "what is claimed is", "it is claimed", "we claim", "the invention claimed is",
            "detailed description", "summary", "summary of the invention", "brief summary",
            "background of the invention", "field of the invention", "technical field",
            "description of embodiments", "description", "brief description of the drawings",
            "cross-reference to related applications", "cross reference to related applications"
        }
        
        # Require an explicit section number/Roman numeral prefix with a period (or multi-level 1.1)
        # so margin line numbers (5, 10, 15...) and normal prose lines are NEVER turned into # Headers.
        header_pattern = re.compile(
            r'^(?:(?:I{1,3}|IV|V|VI{1,3}|IX|X|[A-Z])\.\s+|\d{1,2}(?:\.\d+)+\.?\s+|\d{1,2}\.\s+)([A-Z][a-zA-Z0-9\s\-]{2,50})$'
        )
        # Exclude patent claims and prose sentences from matching header_pattern
        non_header_body = re.compile(
            r'^(?:a|an|the|one\s+or\s+more|at\s+least\s+one|each|wherein|in\s+claim|according\s+to|'
            r'method\s+of|system\s+of|device\s+of|apparatus\s+of|non-transitory|computer[\s\-]+implemented|'
            r'computer[\s\-]+readable|vehicular|humanoid|electronic\s+control)\b'
            r'|\b(?:comprising|comprises|configured|includes|including|having|storing|wherein|claim|claims|'
            r'instructions|plurality|operable|coupled|connected|disposed|executing|executed|may\s+be|can\s+be|'
            r'is\s+a|are\s+a)\b',
            re.IGNORECASE
        )
        
        seen_claims_header = False
        last_para_idx = -1
        for i, l in enumerate(lines):
            s_l = l.strip()
            if re.match(r'^\[\d{4}\]', s_l) and int(s_l[1:5]) >= 20:
                last_para_idx = i
        
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                formatted_lines.append(line)
                continue
                
            # Collapse spaced-out headers like "A B S T R A C T" -> "ABSTRACT"
            stripped = self._collapse_spaced_header(stripped)
            
            # Detect unheadered start of patent Claims section after the last [0xxx] paragraph
            if not seen_claims_header and last_para_idx != -1 and idx > last_para_idx:
                if re.match(r'^1\.\s+(?:A|An|One\s+or\s+more)\s+[a-zA-Z]', stripped):
                    formatted_lines.append("\n# Claims\n")
                    seen_claims_header = True
                elif stripped == "1." and idx + 1 < len(lines) and re.match(r'^(?:A|An|One\s+or\s+more)\s+[a-zA-Z]', lines[idx + 1].strip()):
                    formatted_lines.append("\n# Claims\n")
                    seen_claims_header = True
                
            # If the line is just a stray number (like '1'), skip header formatting and append
            if stripped.isdigit() and len(stripped) < 3:
                formatted_lines.append(line)
                continue
                
            # Check for exact known headers (optionally preceded by INID code like "(57) Abstract")
            inid_stripped = re.sub(r'^\(?\d{2}\)?\s*', '', stripped).strip(':.-‘’\'" ')
            clean_exact_probe = stripped.lower().strip(':.-‘’\'" ')
            is_exact = clean_exact_probe in exact_headers or inid_stripped.lower() in exact_headers
            
            # All-caps check: must contain at least one 4+ letter word and not be an IPC/CPC code
            is_all_caps = (
                stripped.isupper()
                and 3 < len(stripped) < 60
                and bool(re.search(r'[A-Z]{4,}', stripped))
                and not bool(re.search(r'\(\d{4}\.\d{2}\)', stripped))
            )
            match = None
            if not seen_claims_header:
                m = header_pattern.match(stripped)
                if m:
                    body = m.group(1).strip()
                    if not body.endswith('-') and not non_header_body.search(body):
                        match = m
            
            if len(stripped) < 60:
                if is_exact or match or is_all_caps:
                    if is_exact and inid_stripped.lower() in exact_headers:
                        clean_title = inid_stripped.title()
                    elif is_exact:
                        clean_title = clean_exact_probe.title()
                    elif match:
                        clean_title = match.group(1).title()
                    else:
                        clean_title = stripped.title()
                    
                    if clean_title.lower() in {"claims", "what is claimed is", "it is claimed", "we claim", "the invention claimed is"}:
                        seen_claims_header = True
                        clean_title = "Claims"
                    
                    if not stripped.startswith('#'):
                        formatted_lines.append(f"\n# {clean_title}\n")
                        continue
                        
            formatted_lines.append(line)
            
        return "\n".join(formatted_lines)

