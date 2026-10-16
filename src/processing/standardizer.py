import re
from typing import List, Dict, Any, Optional
from src.schemas.document import DocumentSchema
from src.processing.pdf_parser import PDFParser
from src.utils.logger import get_logger

logger = get_logger(__name__)

KNOWN_HYPHEN_PREFIXES = {
    "end", "real", "state", "self", "high", "low", "closed", "open", "multi",
    "co", "three", "two", "one", "object", "event", "ray", "spatial", "geo",
    "human", "non", "tool", "sensor", "purpose", "cost", "user", "long",
    "fine", "chain", "pre", "post", "cross", "range", "time", "frame",
    "data", "model", "system", "vehicle", "driver", "scene", "task", "world",
    "action", "rule", "large", "small", "full", "semi", "auto", "inter",
    "intra", "sub", "super", "hyper", "ultra", "micro", "macro", "meta",
    "well", "first", "second", "third", "fourth", "3D", "4D", "2D"
}

ACRONYM_MAP = {
    "3d": "3D", "4d": "4D", "2d": "2D", "3dgs": "3DGS", "4dgs": "4DGS",
    "av": "AV", "avs": "AVs", "cav": "CAV", "cavs": "CAVs", "adas": "ADAS",
    "ads": "ADS", "bam": "BAM", "ecu": "ECU", "hil": "HIL", "sil": "SIL",
    "radar": "radar", "fmcw": "FMCW", "gan": "GAN", "kpi": "KPI",
    "map": "mAP", "nerf": "NeRF", "ros": "ROS", "carla": "CARLA",
    "navsim": "NAVSIM", "fov": "FOV", "led": "LED", "cpu": "CPU",
    "gpu": "GPU", "rcs": "RCS", "fft": "FFT", "ffts": "FFTs", "cpi": "CPI",
    "mimo": "MIMO", "nci": "NCI", "rdm": "RDM", "rdms": "RDMs",
    "cfar": "CFAR", "dbscan": "DBSCAN", "pn": "PN", "rf": "RF", "pi": "PI",
    "im3": "IM3", "nf": "NF", "rm": "RM", "mape": "MAPE", "fmi": "FMI",
    "osi": "OSI", "fmu": "FMU", "ugv": "UGV", "dvs": "DVS", "fvd": "FVD",
    "psnr": "PSNR", "nvs": "NVS", "sotif": "SOTIF", "dcas": "DCAS"
}


class DocumentStandardizer:
    def __init__(self, use_full_text: bool = True):
        self.use_full_text = use_full_text
        self.pdf_parser = PDFParser()

    @staticmethod
    def _restore_header_line_case(line_text: str, is_first_line: bool) -> str:
        """Restores normal sentence case to a prose line that was previously turned into # Title Case."""
        words = line_text.split()
        if not words:
            return ""
        out = []
        for i, w in enumerate(words):
            clean_w = re.sub(r'[^a-zA-Z0-9]', '', w).lower()
            if clean_w in ACRONYM_MAP:
                prefix = re.match(r'^[^a-zA-Z0-9]*', w).group(0)
                suffix = re.search(r'[^a-zA-Z0-9]*$', w).group(0)
                out.append(f"{prefix}{ACRONYM_MAP[clean_w]}{suffix}")
            elif i == 0 and is_first_line:
                out.append(w[0].upper() + w[1:].lower() if len(w) > 1 else w.upper())
            else:
                out.append(w.lower())
        res = " ".join(out)
        res = re.sub(r'^Arule\b', 'A rule', res)
        return res

    def _clean_abstract_lines(self, raw_block: str, doc_title: str = "") -> str:
        """
        Cleans an extracted abstract block by removing interleaved patent bibliographic lines,
        academic footnotes, conference copyright footers, and re-joining hyphenated line wraps.
        """
        lines = raw_block.split('\n')
        cleaned_lines = []

        title_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', doc_title) if len(w) > 2]

        for line in lines:
            s = line.strip()
            if not s:
                continue

            current_len = sum(len(x) for x in cleaned_lines)
            if current_len > 120:
                # Stop at standalone figure/claim/page numbers or patent publication banners
                if re.match(r'^(?:\d{1,6}\s*[~a-zA-Z]?|#\s*A\)[\'‘]?\d+.*|\d+\s*,\s*\d+|\d+\s*\|\s*)$', s):
                    break
                if re.match(r'^(?:#+\s*)?(?:Patent\s+Application\s+Publication|Sheet\s+\d+\s+of\s+\d+|Fig\.\s*\d+|FIG\.\s*\d+|Processed\s+by\s+Luminess|Codes?\s+Will\s+Be\s+Accessed\s+At|https?://)', s, re.I):
                    break
                if re.match(r'^(?:#+\s*)?(?:\*\s*)?(?:EP|US|WO)\s*\d[\d\s,/]*[A-Z]\d\b', s, re.I):
                    break
                # Stop if the paper's own title repeats immediately after the abstract (e.g. SAE papers)
                if len(title_words) >= 4:
                    line_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', s) if len(w) > 2]
                    if len(line_words) >= 3 and line_words[:3] == title_words[:3]:
                        break

            # Skip OCR stray symbols
            if re.match(r'^[\)\(\.\|=\-~\\\/\*†‡§¶#]+$', s):
                continue

            # Skip academic footnotes & IEEE/journal footers
            if re.match(r'^[\*†‡§¶]\s*(?:Equal\s+contribution|These\s+authors|Project\s+advisor|Project\s+lead)', s, re.I):
                continue
            if re.match(r'^(?:Corresponding\s+author|20\d{2}\s+IEEE|Authorized\s+licensed\s+use|(?:#\s*)?\d{4}-\d{4}/\d{2}/|(?:#\s*)?Doi\s+10\.)', s, re.I):
                continue

            # Skip interleaved patent bibliographic lines (left column INID codes (21), (22), (60), (63), (71), (72), (73))
            if re.match(r'^\(?(?:21|22|60|63|71|72|73)\)?$', s):
                continue
            if re.match(r'^(?:\(?(?:21|22|60|63|71|72|73)\)?\s*)?(?:Appl\.\s*No\.:|Filed:|Provisional\s+application\s+No\.|Continuation(?:-in-part)?\s+of\s+application|Related\s+U\.S\.\s+Application|Applicant:|Inventors?:|Tnventors?:|Assignee:)', s, re.I):
                continue
            if re.search(r'(?:provisional\s+application\s+No\.|provisional\s+ap-|plication\s+No\.\s*\d{2}/|filed\s+on\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|63/\d{3},\d{3}|now\s+Pat\.\s*No\.|Mobileye\s+Vision\s+Technologies|Jerusalem\s*\(IL\))', s, re.I):
                continue
            if re.match(r'^(?:Dec\.\s*\d+,\s*20\d{2}|\d{1,2},\s*20\d{2},?\s*provisional|el|pplication|#\s*Dat|#\s*Data)$', s, re.I):
                continue

            # Handle false markdown headers inside abstract
            if s.startswith('#'):
                header_body = re.sub(r'^#+\s*', '', s).strip()
                if not header_body:
                    continue
                if current_len > 200 and len(header_body.split()) <= 3:
                    break
                s = self._restore_header_line_case(header_body, is_first_line=(len(cleaned_lines) == 0))

            cleaned_lines.append(s)

        # Join lines with smart hyphenation handling
        result = ""
        for line in cleaned_lines:
            if not result:
                result = line
                continue
            if result.endswith('-'):
                prefix_match = re.search(r'([a-zA-Z0-9]+)-$', result)
                prefix = prefix_match.group(1) if prefix_match else ""
                if prefix in KNOWN_HYPHEN_PREFIXES or (line and line[0].isupper()):
                    result = result + line
                else:
                    result = result[:-1] + line
            else:
                result = result + " " + line

        result = PDFParser._normalize_ligatures(result)
        result = re.sub(r'(?<=\s)[‘\'`´](?=[a-z])', '', result)
        result = re.sub(r'\s*See here for our project page\.?\s*$', '', result, flags=re.I)
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    def extract_abstract(self, full_text: str, doc_title: str = "") -> Optional[str]:
        """
        Multi-stage foolproof abstract extractor supporting academic papers and global patent formats
        (US, EP A1/B1, WO PCT), including fallback to Summary/[0001] for granted B1 specifications.
        """
        if not full_text:
            return None

        if "\nFull Text:\n" in full_text:
            search_area = full_text.split("\nFull Text:\n", 1)[1][:15000]
        else:
            search_area = full_text[:15000]

        # Line-anchored stop pattern so ordinary words like "description" inside sentences never truncate the abstract
        stop_pattern = re.compile(
            r'(?:'
            r'(?:^|\n)\s*(?:#+\s*)?(?:(?:1|I)[\.\s\|]+)?(?:introduction|background(?:\s+of\s+the\s+invention)?|motivation|related\s+works?|preliminaries|methodology|methods|system\s+overview|problem\s+statement|description|detailed\s+description|summary(?:\s+of\s+the\s+invention)?|technical\s+field|field\s+of\s+the\s+invention|cross[\s\-]+reference\s+to\s+related\s+applications|brief\s+description\s+of\s+the\s+drawings|claims)\s*(?:\n|$)'
            r'|(?:^|\n)\s*(?:#+\s*)?(?:keywords|index\s+terms|key\s+words|history|article\s+info|citation|funding)\b'
            r'|(?:^|\n)\s*\d+\s+Claims?\s*,?\s*\d+\s+Drawing\s+Sheets?'
            r'|(?:^|\n)\s*(?:#+\s*)?(?:Patent\s+Application\s+Publication|Sheet\s+\d+\s+of\s+\d+|Fig\.\s*\d+|FIG\.\s*\d+|Processed\s+by\s+Luminess)'
            r'|(?:^|\n)\s*\[0001\]'
            r')',
            re.IGNORECASE
        )

        start_patterns = [
            # Stage 1: Explicit Abstract header (including (57) Abstract, spaced A B S T R A C T, or # B S T R A C T)
            re.compile(r'(?:^|\n)\s*(?:#+\s*)?(?:\(\s*57\s*\)\s*|57\s*[\)\n]\s*)?(?:#+\s*)?(?:abstract|a\s+b\s+s\s+t\s+r\s+a\s+c\s+t|b\s+s\s+t\s+r\s+a\s+c\s+t)\b[\s\n:\-\)—]*', re.IGNORECASE),
            # Stage 2: WIPO INID (57) without the word Abstract (e.g. EP A1 applications)
            re.compile(r'(?:^|\n)\s*\(\s*57\s*\)[\s\n:\-\)]*', re.IGNORECASE),
        ]

        for sp in start_patterns:
            for match in sp.finditer(search_area):
                start_idx = match.end()
                window = search_area[start_idx:start_idx + 4500]
                stop_match = stop_pattern.search(window)
                raw_block = window[:stop_match.start()] if stop_match else window
                candidate = self._clean_abstract_lines(raw_block, doc_title=doc_title)
                if len(candidate) > 80:
                    return candidate

        # Stage 3: Fallback for Granted Patent Specifications (e.g. EP B1) that lack a cover-page (57) Abstract
        summary_match = re.search(
            r'(?:^|\n)\s*(?:#+\s*)?Summary(?:\s+of\s+the\s+Invention)?\s*\n+(.*?)(?=(?:^|\n)\s*\[0009\]|(?:^|\n)\s*(?:#+\s*)?(?:Brief\s+Description|Detailed\s+Description|Description\s+of\s+Embodiments))',
            search_area,
            re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            raw_sum = re.sub(r'\[\d{4}\]', '', summary_match.group(1))
            candidate = self._clean_abstract_lines(raw_sum, doc_title=doc_title)
            if len(candidate) > 80:
                logger.info("Extracted fallback abstract from Patent Summary section.")
                return candidate

        # Stage 4: Fallback to opening paragraph [0001] in patents
        p0001_match = re.search(r'\[0001\]\s*(.*?)(?=\[0002\]|\n\s*#)', search_area, re.DOTALL)
        if p0001_match:
            candidate = self._clean_abstract_lines(p0001_match.group(1), doc_title=doc_title)
            if len(candidate) > 50:
                logger.info("Extracted fallback abstract from Patent paragraph [0001].")
                return candidate

        return None

    def process(self, documents: List[DocumentSchema]) -> List[Dict[str, Any]]:
        standardized_docs = []
        for doc in documents:
            
            content_body = f"# Abstract\n{doc.abstract}"
            
            # Try to fetch full text if enabled
            if self.use_full_text:
                full_text = getattr(doc, "full_text", None)
                if not full_text:
                    if getattr(doc, "local_path", None):
                        full_text = self.pdf_parser.extract_text_from_file(doc.local_path)
                    elif doc.pdf_url:
                        full_text = self.pdf_parser.extract_text_from_url(doc.pdf_url, doc_id=doc.id)
                    
                if full_text:
                    # Attempt to extract abstract if missing or defaulted
                    if not doc.abstract or "no abstract" in doc.abstract.lower() or len(doc.abstract.strip()) < 50:
                        extracted = self.extract_abstract(full_text, doc_title=doc.title)
                        if extracted:
                            doc.abstract = extracted
                            logger.info(f"Successfully extracted abstract for {doc.id} ({len(extracted)} chars)")
                        else:
                            logger.warning(f"Could not extract abstract for {doc.id}")

                    has_valid_abstract = bool(doc.abstract and "no abstract" not in doc.abstract.lower())
                    content_body = (
                        f"# Abstract\n{doc.abstract}\n\nFull Text:\n{full_text}"
                        if has_valid_abstract
                        else f"Full Text:\n{full_text}"
                    )
                    logger.info(f"Successfully incorporated full text for {doc.id}")
                else:
                    logger.debug(f"Falling back to abstract for {doc.id}")

            combined_text = (
                f"Title: {doc.title}\n"
                f"Authors: {', '.join(doc.authors)}\n"
                f"Published Date: {doc.published_date}\n\n"
                f"{content_body}"
            )
            
            # Save the final text representation to disk for transparency
            import src.config as cfg
            safe_id = doc.id.replace('/', '_')
            text_path = cfg.EXTRACTED_TEXT_DIR / f"{safe_id}.txt"
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(combined_text)
            
            standardized_docs.append({
                "id": doc.id,
                "text": combined_text,
                "metadata": {
                    "source": doc.source,
                    "title": doc.title,
                    "abstract": doc.abstract,
                    "published_date": doc.published_date,
                    "pdf_url": doc.pdf_url
                }
            })
        return standardized_docs