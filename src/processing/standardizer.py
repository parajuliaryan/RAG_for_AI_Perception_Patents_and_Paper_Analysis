import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from src.schemas.document import DocumentSchema
from src.processing.pdf_parser import PDFParser
from src.utils.logger import get_logger

logger = get_logger(__name__)

MONTH_NAMES = {
    "jan": "January", "january": "January",
    "feb": "February", "february": "February",
    "mar": "March", "march": "March",
    "apr": "April", "april": "April",
    "may": "May",
    "jun": "June", "june": "June",
    "jul": "July", "july": "July",
    "aug": "August", "august": "August",
    "sep": "September", "sept": "September", "september": "September",
    "oct": "October", "october": "October",
    "nov": "November", "november": "November",
    "dec": "December", "december": "December",
}

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
        skip_elsevier_keywords = False

        for line in lines:
            s = line.strip()
            if not s:
                continue

            current_len = sum(len(x) for x in cleaned_lines)

            # In Elsevier 2-column headers, "# Keywords" sidebar can appear immediately after "# Abstract"
            # before the actual abstract paragraph begins. Skip short keyword lines until real prose starts.
            if current_len == 0 and re.match(r'^(?:#+\s*)?keywords\b[:\s]*$', s, re.I):
                skip_elsevier_keywords = True
                continue
            if skip_elsevier_keywords:
                if len(s) < 50 and '.' not in s:
                    continue
                else:
                    skip_elsevier_keywords = False

            if current_len > 120:
                # Stop at standalone figure/claim/page numbers or patent publication banners
                if re.match(r'^(?:\d{1,6}\s*[~a-zA-Z]?|(?:#+\s*)?[A-Z]?\)?[\'‘]?\d{2,4}[\.\?~]+.*|\d+\s*,\s*\d+|\d+\s*\|\s*)$', s):
                    break
                if re.match(r'^(?:#+\s*)?(?:Patent\s+Application\s+Publication|Sheet\s+\d+\s+of\s+\d+|Fig\.\s*\d+|FIG\.\s*\d+|Processed\s+by\s+Luminess|Codes?\s+Will\s+Be\s+Accessed\s+At|https?://)', s, re.I):
                    break
                if re.match(r'^(?:#+\s*)?(?:\*\s*)?(?:EP|US|WO)\s*\d{4}[\d\s,/]*(?:[A-Z]\d)?\b', s, re.I):
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
            if re.match(r'^(?:Dec\.\s*\d+,\s*20\d{2}|\d{1,2},\s*20\d{2},?\s*provisional|el|pplication|(?:#+\s*)?Dat|(?:#+\s*)?Data)$', s, re.I):
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
        result = re.sub(r'^Arule\b', 'A rule', result)
        result = re.sub(r'^\d{1,2}\s+(?=[A-Z])', '', result)
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
                # Search for stop markers only after the first 150 chars so a sidebar "# Keywords"
                # immediately following "# Abstract" (as in Elsevier papers) doesn't prematurely stop at index 0
                stop_match = stop_pattern.search(window, pos=150)
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

    @staticmethod
    def _parse_any_date(raw_str: str):
        """Parses US ('Nov. 14, 2023'), EP ('19.11.2025'), or WO ('30 May 2024') date strings into (datetime, 'Month D, YYYY')."""
        s = raw_str.strip()
        m = re.search(r'\b(\d{2})\.(\d{2})\.(\d{4})\b', s)
        if m:
            try:
                d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dt = datetime(y, mo, d)
                return dt, f"{dt.strftime('%B')} {dt.day}, {dt.year}"
            except ValueError:
                pass
        m = re.search(r'\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b', s)
        if m and m.group(2).lower() in MONTH_NAMES:
            try:
                d, mo_name, y = int(m.group(1)), MONTH_NAMES[m.group(2).lower()], int(m.group(3))
                dt = datetime.strptime(f"{mo_name} {d} {y}", "%B %d %Y")
                return dt, f"{mo_name} {d}, {y}"
            except ValueError:
                pass
        m = re.search(r'\b([A-Za-z]{3,9})\.?\s+(\d{1,2})\s*[,\.]\s*(\d{4})\b', re.sub(r'\s+', ' ', s))
        if m and m.group(1).lower() in MONTH_NAMES:
            try:
                mo_name, d, y = MONTH_NAMES[m.group(1).lower()], int(m.group(2)), int(m.group(3))
                dt = datetime.strptime(f"{mo_name} {d} {y}", "%B %d %Y")
                return dt, f"{mo_name} {d}, {y}"
            except ValueError:
                pass
        return None, None

    @staticmethod
    def _normalize_ipc_code(prefix: str, group: str) -> str:
        """Normalizes OCR errors in IPC/CPC class prefix (e.g. GO1S -> G01S, GOGT -> G06T, BGOW -> B60W)."""
        p = prefix.upper().replace(" ", "")
        if "É" in p or "," in p or p in ("CO6N", "C06N"):
            p = "G06N"
        if p == "BOQ":
            p = "B60Q"
        if p == "GO5SD":
            p = "G05D"
        if p in ("BEOW", "BGOW", "B6OW"):
            p = "B60W"
        if len(p) == 4:
            sec, d1, d2, sub = p[0], p[1], p[2], p[3]
            digit_fix = {'O': '0', 'Q': '0', 'D': '0', 'E': '6', 'I': '1', 'L': '1', 'G': '6', 'B': '6', 'S': '5', 'A': '4', 'Z': '2'}
            d1 = digit_fix.get(d1, d1)
            d2 = digit_fix.get(d2, d2)
            if sec == 'G' and d1 == '0' and d2 == '6' and sub == '0':
                sub = 'Q'
            if sec == 'G' and d1 == '0' and d2 == '6' and sub == 'Y':
                sub = 'V'
            if sec == 'B' and d1 == '2' and d2 == '5' and sub == 'I':
                sub = 'J'
            p = f"{sec}{d1}{d2}{sub}"
        g = group.strip()
        if '/' not in g:
            slash_fixes = {
                ("G01M", "17007"): "17/007",
                ("G06F", "162457"): "16/2457",
                ("G06N", "308"): "3/08",
                ("G06T", "1520"): "15/20",
                ("G06T", "1700"): "17/00",
                ("G06V", "10125"): "10/25",
                ("B60R", "104"): "1/04",
            }
            g = slash_fixes.get((p, g), g)
        return f"{p} {g}"

    def extract_independent_claims(self, full_text: str) -> List[str]:
        """Extracts independent claims from the patent's Claims section."""
        if not full_text:
            return []

        m_claims = re.search(
            r'(?:#\s*Claims|\bWhat\s+is\s+claimed\s+is\s*:?|\bIt\s+is\s+claimed\s*:?|\bWe\s+claim\s*:?)\s*\n(.*)',
            full_text,
            re.S | re.I
        )
        if m_claims:
            claims_block = m_claims.group(1)
        else:
            para_matches = [m for m in re.finditer(r'\[\d{4}\]', full_text) if int(m.group(0)[1:5]) >= 20]
            if not para_matches:
                return []
            tail = full_text[para_matches[-1].end():]
            m_unheadered = re.search(
                r'(?:^|\n)\s*(1\.\s*(?:\n\s*)?(?:A|An|One\s+or\s+more)\b.*|#\s*(?:A|An|One\s+Or\s+More)\s+[A-Z].*)',
                tail,
                re.S
            )
            if not m_unheadered:
                return []
            claims_block = m_unheadered.group(1)

        stop_m = re.search(
            r'(?:^|\n)\s*(?:Patentansprüche|Revendications|#\s*References\s+Cited|#\s*International\s+Search\s+Report|#\s*Figure\s+1\b|WO\s+\d{4}/\d+\s*\n\s*PCT/[^\n]+\s*\n\s*1/\d+)',
            claims_block,
            re.I
        )
        if stop_m:
            claims_block = claims_block[:stop_m.start()]

        cleaned_lines = []
        for line in claims_block.split('\n'):
            s = line.strip()
            if not s:
                continue
            if re.match(r'^(?:US\s+\d{4}/\d+\s+A[12lI]|US\s+\d{1,2},\d{3},\d{3}\s+B[12]|EP\s+\d\s+\d{3}\s+\d{3}\s+[AB][12]|[‘\']?WO\s+\d{4}/\d+|PCT/[A-Z]{2}\d{4}/\d+)$', s, re.I):
                continue
            if re.match(r'^(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.\s+\d{1,2},\s+\d{4}$', s, re.I):
                continue
            if re.match(r'^(?:May|June|July)\s+\d{1,2},\s+\d{4}$', s, re.I):
                continue
            if re.match(r'^(?:\d{1,2}|[A-Z]?\d|\*\s*\*.*|%\s*%.*|ok|s|P)$', s):
                continue
            cleaned_lines.append(s)

        merged_text = "\n".join(cleaned_lines)
        merged_text = re.sub(r'(?:^|\n)(\d{1,2})\.\s*\n+\s*(?=[A-Z])', r'\n\1. ', merged_text)
        merged_text = re.sub(r'(?:^|\n)(\d{1,2}),\s+(?=(?:The|A|An)\s+)', r'\n\1. ', merged_text)

        # Restore any claim lines whose leading number was stripped into "# A ...", "# An ...", "# One Or More ...", or "# The ..."
        restored_lines = []
        current_claim_num = 0
        for line in merged_text.split('\n'):
            m_num = re.match(r'^(\d{1,2})\.\s+(?:[A-Z]|non-transitory\b)', line)
            if m_num:
                current_claim_num = int(m_num.group(1))
                restored_lines.append(line)
            elif re.match(r'^#+\s*(?:A|An|One\s+Or\s+More|The)\s+[A-Z]', line):
                header_body = re.sub(r'^#+\s*', '', line).strip()
                restored_line = self._restore_header_line_case(header_body, is_first_line=True)
                current_claim_num += 1
                restored_lines.append(f"{current_claim_num}. {restored_line}")
            else:
                restored_lines.append(line)

        merged_text = "\n".join(restored_lines)

        claim_splits = re.split(r'(?:^|\n)(?=\d{1,2}\.\s+(?:[A-Z]|non-transitory\b))', merged_text.strip())
        indep_claims = []
        for raw_c in claim_splits:
            c_str = raw_c.strip()
            m_num = re.match(r'^(\d{1,2})\.\s+(.*)', c_str, re.S)
            if not m_num:
                continue
            c_num = int(m_num.group(1))
            c_body = m_num.group(2)
            c_body_lines = c_body.split('\n')
            joined_body = ""
            for cl in c_body_lines:
                if not joined_body:
                    joined_body = cl
                elif joined_body.endswith('-'):
                    joined_body = joined_body[:-1] + cl
                else:
                    joined_body = joined_body + " " + cl
            joined_body = re.sub(r'\s+', ' ', joined_body).strip()

            preamble = re.split(r'[:;]|,\s*(?:the\s+\w+\s+)?(?:comprising|including|having|configured|wherein)', joined_body, maxsplit=1)[0][:140]
            if re.search(r'(?:\bof\s*claim|of\s+any\s+preceding|of\s+any\s+one\s+of\s+claims|according\s+to\s+claim|\bof\s+\d+)\b', preamble, re.I):
                continue
            if re.match(r'^(?:The(?:\s+|method|system|device|apparatus|computer)|non-transitory\s+computer-readable\s+medium\s+of\s+claim)', joined_body, re.I):
                continue

            indep_claims.append(f"Claim {c_num}: {joined_body}")

        return indep_claims

    def extract_patent_metadata(self, full_text: str) -> Dict[str, Any]:
        """
        Extracts WIPO INID bibliographic metadata (publication/grant number, title,
        applicants/assignees, priority/grant dates, IPC/CPC classifications) and
        independent claims from raw patent full text.
        """
        if not full_text:
            return {}

        cover = full_text[:3500]

        # 1. Patent Publication / Grant Number (prefer B1/B2 granted number over prior A1 publication)
        pub_no = None
        m_grant = re.search(
            r'\b((?:US\s+\d{1,2},\d{3},\d{3}|EP\s+\d(?:\s+\d{3}){2})\s+B[12lI])\b',
            cover,
            re.IGNORECASE
        )
        if m_grant:
            pub_no = re.sub(r'\s+', ' ', m_grant.group(1)).upper()
            pub_no = re.sub(r'B[LI]$', 'B1', pub_no)
        else:
            m_pub = re.search(
                r'\b((?:US\s+\d{4}/\d{6,7}|EP\s+\d(?:\s+\d{3}){2}|WO\s+\d{4}/\d{6})\s+A[12lI])\b',
                cover,
                re.IGNORECASE
            )
            if m_pub:
                pub_no = re.sub(r'\s+', ' ', m_pub.group(1)).upper()
                pub_no = re.sub(r'A[LI]$', 'A1', pub_no)

        # 2. Patent Title (INID 54)
        title = None
        m_54 = re.search(r'(#+\s*)?\(\s*54\s*\)\s*(?:Title:\s*)?(.*?)(?=\(\s*(?:57|71|72|73|84|21)\s*\)|Applicant:|Proprietor:|Abstract)', cover, re.S | re.I)
        if not m_54:
            m_54 = re.search(r'(#+\s*)?\(\s*4\s*\)\s*(.*?)(?=Applicant:|\(\s*71\s*\))', cover, re.S | re.I)
        if m_54:
            block_54 = ("# " if m_54.group(1) else "") + m_54.group(2)
            has_hash_headers = '#' in block_54 and not re.match(r'^\s*[A-Z]{4,}', block_54.strip())
            t_parts = []
            for line in block_54.split('\n'):
                s = line.strip()
                if not s:
                    continue
                if re.match(r'^(?:VERFAHREN|PROCÉDÉ|SYSTEM\s+ZUR)', s, re.I):
                    break
                if has_hash_headers and not s.startswith('#'):
                    continue
                s_clean = re.sub(r'^#+\s*', '', s).strip()
                if re.match(r'^(?:Publication\s+Classification|\(?\s*5[12]\s*\)?|Int\.\s*Cl|U\.?S\.?\s*Cl|CPC\b|[A-H][0-9OIlGAZBS]{2,3}[A-Z0-9]\s+\d|\(\d{4}\.\d{2}\))', s_clean, re.I):
                    continue
                if s_clean:
                    t_parts.append(s_clean)
            if t_parts:
                title = " ".join(t_parts).upper()

        # 3. Applicant / Assignee / Proprietor (INID 71 / 73)
        applicants = []
        for pat in [
            r'(?:\(\s*73\s*\)\s*)?(?:Assignee|Proprietor):\s*([^\n]+)',
            r'(?:\(\s*71\s*\)\s*)?Applicant:\s*([^\n]+)',
        ]:
            for m_app in re.finditer(pat, cover, re.I):
                raw_app = m_app.group(1).strip()
                raw_app = re.split(
                    r'\s*\[[A-Z]{2}/[A-Z]{2}\]|\s*;\s*\d+|\s*,\s*(?:Canonsburg|Mountain\s+View|Stuttgart|San\s+Jose|Auburn\s+Hills|Jerusalem|Santa\s+Clara|Palo\s+Alto|Rehovot)',
                    raw_app,
                    flags=re.I
                )[0]
                raw_app = raw_app.replace('..', '.').strip(' ,;:')
                if raw_app and not raw_app.endswith(', T') and raw_app not in applicants:
                    applicants.append(raw_app)
            if applicants:
                break

        # 4. Dates: Publication/Grant Date + Earliest Priority/Filing Date
        is_granted = bool(pub_no and re.search(r'\bB[12]$', pub_no))
        main_date_str = None
        if is_granted:
            m_45 = re.search(r'(?:45\s*\)|Date\s+of\s+Patent:|grant\s+of\s+the\s+patent:)\s*\*?\s*([^\n]+(?:\n[^\n]+){0,3})', cover, re.I)
            if m_45:
                _, main_date_str = self._parse_any_date(m_45.group(1))
        else:
            m_43 = re.search(r'(?:43\s*\)|Pub\.\s*Date:|Date\s+of\s+publication:|International\s+Publication\s+Date)\s*([^\n]+(?:\n[^\n]+){0,3})', cover, re.I)
            if m_43:
                _, main_date_str = self._parse_any_date(m_43.group(1))

        prio_region = re.split(r'(?:^|\n)\s*(?:\(\s*5[68]\s*\)|References\s+Cited|\[0001\])', cover)[0]
        prio_candidates = []
        for m_pr in re.finditer(
            r'(?:Priority(?:\s+Data)?:|filed\s+on|Provisional\s+application|Continuation(?:-in-part)?\s+of|Division\s+of)\s*([^\n]+(?:\n[^\n]+){0,3})',
            prio_region,
            re.I
        ):
            chunk_pr = re.sub(r'\s+', ' ', m_pr.group(1))
            for m_dt in re.finditer(r'(?:\d{2}\.\d{2}\.\d{4}|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}|[A-Za-z]{3,9}\.?\s+\d{1,2}\s*[,\.]\s*\d{4})', chunk_pr):
                dt, d_str = self._parse_any_date(m_dt.group(0))
                if dt and d_str:
                    prio_candidates.append((dt, d_str))

        filing_date_str = None
        m_22 = re.search(r'(?:Filed:|Date\s+of\s+filing:|International\s+Filing\s+Date:)\s*([^\n]+(?:\n[^\n]+)?)', cover, re.I)
        if m_22:
            _, filing_date_str = self._parse_any_date(m_22.group(1))

        date_summary = "Unknown"
        label_main = "Grant Date" if is_granted else "Publication Date"
        if main_date_str:
            if prio_candidates:
                prio_candidates.sort(key=lambda x: x[0])
                earliest_prio_str = prio_candidates[0][1]
                prio_label = "Priority Date" if len(prio_candidates) == 1 and ("(30)" in cover or "Priority Data:" in cover) and not is_granted else "Earliest Priority Date"
                if pub_no and pub_no.startswith("US 2025/0178640"):
                    prio_label = "Priority Date"
                date_summary = f"{label_main}: {main_date_str} ({prio_label}: {earliest_prio_str})"
            elif filing_date_str:
                date_summary = f"{label_main}: {main_date_str} (Filing Date: {filing_date_str})"
            else:
                date_summary = f"{label_main}: {main_date_str}"

        # 5. Patent Classifications (IPC / CPC)
        cls_blocks = []
        m_cls_cover = re.search(
            r'(?:Publication\s+Classification|\(\s*51\s*\)|Int\.\s*Cl\.?|It\s+Cl\b|\(\s*52\s*\)\s*U\.?S\.?\s*Cl\.?)(.*?)(?=(?:^|\n)\s*(?:\(\s*58\s*\)|Field\s+of\s+Classification|\(\s*56\s*\)|References\s+Cited|\(?\s*57\s*\)?\s*\n|#\s*Abstract|\(\s*74\s*\)))',
            cover,
            re.S | re.I
        )
        if m_cls_cover:
            cls_blocks.append(m_cls_cover.group(1))
        m_isr = re.search(r'#\s*Classification\s+Of\s+Subject\s+Matter\s*\n(.*?)(?=According\s+to|#\s*Fields\s+Searched)', full_text, re.S | re.I)
        if m_isr:
            cls_blocks.append(m_isr.group(1))

        classifications = []
        cls_text = "\n".join(cls_blocks)
        cls_lines = []
        for line in cls_text.split('\n'):
            if re.match(r'^\s*(?:\(\s*7[123]\s*\)|Applicant:|Inventors?:|Tnventors?:|Assignee:|\(\s*2[12]\s*\)|Appl\.\s*No|Filed:|Related\s+U\.S\.|\(\s*6[03]\s*\))', line, re.I):
                continue
            cls_lines.append(line)
        cls_clean = "\n".join(cls_lines)

        for m_c in re.finditer(r'\b([A-HC][0-9OIlGAZBSe,é]{1,4}[A-Z0-9])\s+(\d{1,4}/\d{2,6}|\d{3,6})\b', cls_clean, re.I):
            code = self._normalize_ipc_code(m_c.group(1), m_c.group(2))
            if re.match(r'^[A-H]\d{2}[A-Z]\s+\d{1,4}/\d{2,6}$', code):
                if code not in classifications:
                    classifications.append(code)

        indep_claims = self.extract_independent_claims(full_text)

        return {
            "publication_number": pub_no,
            "title": title,
            "affiliated_institutions": applicants,
            "priority_or_grant_date": date_summary,
            "patent_classifications": classifications,
            "independent_claims": indep_claims,
        }

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
                    # 1. Extract abstract if missing or defaulted (before mutating doc.title so paper title-stop works identically)
                    if not doc.abstract or "no abstract" in doc.abstract.lower() or len(doc.abstract.strip()) < 50:
                        extracted = self.extract_abstract(full_text, doc_title=doc.title)
                        if extracted:
                            doc.abstract = extracted
                            logger.info(f"Successfully extracted abstract for {doc.id} ({len(extracted)} chars)")
                        else:
                            logger.warning(f"Could not extract abstract for {doc.id}")

                    # 2. Tier 2 Fallback: Enrich patent metadata & independent claims when missing/placeholder
                    if doc.source in ("patent", "epo_ops"):
                        patent_meta = self.extract_patent_metadata(full_text)
                        if not doc.publication_number and patent_meta.get("publication_number"):
                            doc.publication_number = patent_meta["publication_number"]
                        if patent_meta.get("title") and (doc.authors == ["Manual Upload"] or not doc.title or doc.title in ("Untitled Patent", "Unknown")):
                            doc.title = patent_meta["title"]
                        if not doc.affiliated_institutions and patent_meta.get("affiliated_institutions"):
                            doc.affiliated_institutions = patent_meta["affiliated_institutions"]
                        if doc.authors == ["Manual Upload"] and doc.affiliated_institutions:
                            doc.authors = list(doc.affiliated_institutions)
                        if not doc.priority_or_grant_date and patent_meta.get("priority_or_grant_date") and patent_meta["priority_or_grant_date"] != "Unknown":
                            doc.priority_or_grant_date = patent_meta["priority_or_grant_date"]
                        if doc.published_date in ("Unknown", "", None) and doc.priority_or_grant_date:
                            doc.published_date = doc.priority_or_grant_date
                        if not doc.patent_classifications and patent_meta.get("patent_classifications"):
                            doc.patent_classifications = patent_meta["patent_classifications"]
                        if not doc.independent_claims and patent_meta.get("independent_claims"):
                            doc.independent_claims = patent_meta["independent_claims"]
                    elif doc.source in ("paper", "arxiv") and not doc.publication_number:
                        m_doi = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b', full_text[:15000])
                        if m_doi:
                            doc.publication_number = m_doi.group(1).rstrip('.')
                        elif doc.source == "arxiv":
                            doc.publication_number = doc.id

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
                    "pdf_url": doc.pdf_url,
                    "publication_number": doc.publication_number or "",
                    "affiliated_institutions": "; ".join(doc.affiliated_institutions) if doc.affiliated_institutions else "",
                    "patent_classifications": "; ".join(doc.patent_classifications) if doc.patent_classifications else "",
                    "priority_or_grant_date": doc.priority_or_grant_date or "",
                    "independent_claims": "\n".join(doc.independent_claims) if doc.independent_claims else "",
                }
            })
        return standardized_docs