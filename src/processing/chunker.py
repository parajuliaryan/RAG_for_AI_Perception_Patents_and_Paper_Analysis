import re
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from src.processing.base_chunker import BaseChunker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StructuralChunker(BaseChunker):
    """
    Hybrid structural chunker with foolproof pre-chunking sanitization:
    1. Sanitizes document text before splitting:
       - Strips redundant top metadata preamble (Title/Authors/Date) and duplicate Abstract blocks.
       - For patents: strips cover-page citation tables, rotated Drawing Sheets ('Sheet X of Y'),
         and post-Claims tail boilerplate (Search Reports, foreign claim translations, trailing figures).
       - For academic papers: enriches inline product/hardware citations [N] from online datasheet
         references (e.g., dSPACE DARTS 9040-G, Rohde & Schwarz R&S AREG800A) and strips '# References',
         running page headers/footers, NeurIPS checklists, Acknowledgments, and false table/acronym headers.
       - Recovers formatting-specific section headers (Wiley pipe headers, SAE inline headings,
         NeurIPS two-line headers, and long numbered section titles).
    2. Splits structurally based on Markdown headers and qualifies subsections with parent context.
    3. Coalesces short adjacent sections (< 400 chars) and tail sub-split remainders (< 400 chars)
       so zero tiny stub chunks are emitted.
    4. Uses RecursiveCharacterTextSplitter for oversized sections and injects document metadata.
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
        ]
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # ------------------------------------------------------------------
    # Pre-Chunking Sanitization & Structural Header Recovery
    # ------------------------------------------------------------------

    @classmethod
    def sanitize_and_clean_for_chunking(cls, text: str, source: str = "") -> str:
        """
        Cleans raw standardized text prior to Markdown header splitting.
        """
        if not text:
            return ""

        # Normalize combining circle trademark artifact (e.g., "R\u20ddAREG800A" -> "® AREG800A")
        text = re.sub(r'R\u20dd\s*', '® ', text)

        is_patent = source in ("patent", "epo_ops")

        # 1. Strip redundant top preamble (Title / Authors / Published Date) since Chunker injects Title & Date into every chunk
        text = re.sub(
            r'^Title:\s*[^\n]*\nAuthors:\s*[^\n]*\nPublished Date:\s*[^\n]*\n+',
            '',
            text,
            count=1,
        )

        # Split into top "# Abstract" part and "Full Text:" part if present
        top_abstract_part = ""
        full_text_body = text
        if "\nFull Text:\n" in text:
            top_abstract_part, full_text_body = text.split("\nFull Text:\n", 1)
            top_abstract_part = top_abstract_part.strip()
        elif text.startswith("Full Text:\n"):
            full_text_body = text[len("Full Text:\n"):]

        if is_patent:
            full_text_body = cls._sanitize_patent_body(full_text_body)
        else:
            full_text_body = cls._sanitize_paper_body(full_text_body, has_top_abstract=bool(top_abstract_part))

        if top_abstract_part:
            return f"{top_abstract_part}\n\n{full_text_body}".strip()
        return full_text_body.strip()

    @classmethod
    def _sanitize_patent_body(cls, body: str) -> str:
        # 1. Strip cover-page bibliographic tables and rotated Drawing Sheets before the Patent Specification starts
        spec_start_pat = re.compile(
            r'(?:^|\n)(?:'
            r'#\s*(?:Claims\b|Cross[\s\-]+Reference|Related\s+Application|Technical\s+Field|Field\s+Of\s+The\s+Invention|'
            r'Background|Summary|Brief\s+Summary|Description\s+Of\s+Embodiments|Description\s+Of\s+The\s+Preferred|'
            r'Detailed\s+Description|Brief\s+Description\s+Of\s+The\s+Drawings|Brief\s+Description\s+Of\s+The\s+Several|'
            r'Brief\s+Description\s+Of\s+The\n|Description\b)'
            r'|\s*\[0001\]'
            r')',
            re.IGNORECASE,
        )
        m_start = spec_start_pat.search(body)
        if m_start:
            body = body[m_start.start():]

        # 2. Strip post-Claims tail noise (International Search Report, References Cited in Description, Foreign Claims, Post-Claim Drawings)
        tail_stop_pat = re.compile(
            r'(?:^|\n)\s*(?:#+\s*)?(?:Patentanspr[üu]che|Revendications|References\s+Cited\b|'
            r'International\s+Search\s+Report|Classification\s+Of\s+Subject\s+Matter|'
            r'Figure\s+1\b|WO\s+\d{4}/\d+\s*\n\s*PCT/[^\n]+\s*\n\s*1/\d+)',
            re.IGNORECASE,
        )
        m_tail = tail_stop_pat.search(body)
        if m_tail:
            body = body[:m_tail.start()]

        # 3. Remove running patent page headers/footers and un-header false inline headers
        cleaned_lines = []
        for line in body.split("\n"):
            s = line.strip()
            if re.match(
                r'^(?:US\s+\d{4}/\d+\s+A[12lI]|US\s+\d{1,2},\d{3},\d{3}\s+B[12]|EP\s+\d\s+\d{3}\s+\d{3}\s+[AB][12]|'
                r'[‘\']?WO\s+\d{4}/\d+(?:\s+A[123])?|PCT/[A-Z]{2}\d{4}/\d+|Sheet\s+\d+\s+of\s+\d+)$',
                s,
                re.IGNORECASE,
            ):
                continue
            # Fix literal '#' without space in code/scenario tables (e.g. "#target vehicle" in Applied Intuition)
            if s.startswith("#") and not s.startswith("# ") and not s.startswith("##"):
                line = s.lstrip("#").strip()
                s = line
            if s.startswith("#"):
                h_title = re.sub(r'^#+\s*', '', s).strip()
                if re.match(
                    r'^(?:Figs?\b|Table\s+\d|Scenario\b|\(Pomdp\)|Lstm\b|Although\b|Fayo\)|20\d{2}\s*\(|Other\b|Examples\b)',
                    h_title,
                    re.IGNORECASE,
                ):
                    line = h_title
            cleaned_lines.append(line)

        body = "\n".join(cleaned_lines)
        body = cls._merge_consecutive_headers(body)
        body = re.sub(r'(#\s*Claims\s*\n+)+#\s*Claims\b', '# Claims', body, flags=re.I)
        return body

    @staticmethod
    def _enrich_online_product_citations(before_refs: str, refs_block: str) -> str:
        """
        Before stripping '# References', resolves inline citations [N] that point to
        product/hardware/simulator datasheets (entries with 'Accessed:' / '[Online]. Available:')
        so vendor and product names (e.g. dSPACE for DARTS 9040-G, Rohde & Schwarz for R&S AREG800A)
        are preserved right where the tool/hardware is used in the paper body.
        """
        entries = re.split(r'(?:^|\n)(?=\[\d+\])', refs_block)
        for entry in entries:
            m_ref = re.match(r'^\[(\d+)\]\s*(.*?)(?:Accessed:|\[Online\])', entry.strip(), re.S | re.I)
            if not m_ref:
                continue
            ref_num = m_ref.group(1)
            raw_desc = re.sub(r'#\s*', '', m_ref.group(2))
            clean_desc = re.sub(r'\s+', ' ', raw_desc).strip(' .,;')
            if 5 < len(clean_desc) < 160:
                before_refs = re.sub(
                    rf'\[{ref_num}\]',
                    f'[{ref_num} ({clean_desc})]',
                    before_refs,
                    count=1,
                )
        return before_refs

    @classmethod
    def _sanitize_paper_body(cls, body: str, has_top_abstract: bool = True) -> str:
        # 1. If we already have the standardized top "# Abstract", strip the duplicate pre-Introduction header/abstract block
        if has_top_abstract:
            first_body_sec = re.search(
                r'(?:^|\n)(?:#\s*(?:1\.?\s+|I\.?\s+)?(?:Introduction|Motivation|Survey\s+Of\s+Existing|Overview\s+Of\s+The\s+Testing)\b|1\s*\n\s*\|\s*\n+\s*#?\s*Introduction\b)',
                body,
                re.IGNORECASE,
            )
            if first_body_sec and first_body_sec.start() < 8000:
                pre_intro = body[:first_body_sec.start()]
                fig1_match = re.search(
                    r'(?:^|\n)(Figure\s+1\.\s+[^\n]+(?:\n(?!(?:#|Abstract\b|Keywords\b|Article\s+Info))[^\n]+)*)',
                    pre_intro,
                )
                body = body[first_body_sec.start():]
                if fig1_match:
                    intro_line_end = body.find("\n", 2)
                    if intro_line_end != -1:
                        body = body[:intro_line_end] + "\n\n" + fig1_match.group(1).strip() + "\n" + body[intro_line_end:]

        # 2. Enrich inline product/tool citations from '# References' and then strip '# References'
        ref_match = re.search(r'(?:^|\n)#\s*(?:References|Bibliography)\b', body, re.IGNORECASE)
        if ref_match:
            before_refs = body[:ref_match.start()]
            after_refs = body[ref_match.end():]
            before_refs = cls._enrich_online_product_citations(before_refs, after_refs)
            appendix_match = re.search(
                r'(?:^|\n)#\s*(?:Appendix\b|Implementation\s+Details\b|Additional\s+Architecture\s+Details\b|'
                r'Supplementary\b|Related\s+Work\b|Ijcb\s*-\s*Tpr@Fpr)',
                after_refs,
                re.IGNORECASE,
            )
            if appendix_match:
                body = before_refs + "\n\n" + after_refs[appendix_match.start():]
            else:
                body = before_refs

        # 3. Recover formatting-specific section headers missed by PDFParser:
        # 3a. Wiley pipe-formatted headers: "2\n|\nDriving Datasets" -> "\n# Driving Datasets\n"
        body = re.sub(
            r'(?:^|\n)\d+(?:\.\d+)*\s*\n\s*\|\s*\n+\s*(?:#\s*)?([A-Z][^\n]{2,80})',
            lambda m: f"\n# {m.group(1).strip().title()}\n",
            body,
        )
        # 3b. Wrapped two-line numbered section headers
        body = re.sub(
            r'(?:^|\n)(\d{1,2}(?:\.\d+)*\.\s+[A-Z][a-zA-Z0-9\s\-&/:,()]{25,75})\n([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})(?=\n)',
            lambda m: f"\n# {(m.group(1).strip() + ' ' + m.group(2).strip())}\n",
            body,
        )
        # 3c. NeurIPS two-line numbered headers
        body = re.sub(
            r'(?:^|\n)(?:[1-6]\.[1-6]|[2-6])\s*\n\s*([A-Z][a-zA-Z\-]+(?:\s+(?:of|and|for|from|in|with|to|[A-Z][a-zA-Z\-]+)){1,7})(?=\n)',
            lambda m: f"\n# {m.group(1).strip().title()}\n",
            body,
        )

        # 4. Line-by-line cleanup of running journal headers/footers, SAE inline headings, long numbered headers, and false markdown headers
        long_numbered_hdr_pat = re.compile(
            r'^(?:(?:I{1,3}|IV|V|VI{1,3}|IX|X)\.\s+|\d{1,2}(?:\.\d+)+\.?\s+|\d{1,2}\.\s+)([A-Z][a-zA-Z0-9\s\-&/:,()]{3,95})$'
        )
        non_header_prose_pat = re.compile(
            r'^(?:a|an|one\s+or\s+more|at\s+least\s+one|each|wherein|in\s+claim|according\s+to|'
            r'we\b|in\s+this|for\s+each|as\s+shown|to\s+evaluate|first|second|third|finally)\b'
            r'|\b(?:comprising|comprises|configured|includes|including|having|storing|wherein|claim|claims|'
            r'instructions|plurality|operable|coupled|connected|disposed|executing|executed|may\s+be|can\s+be|'
            r'is\s+a|are\s+a|was\s+a|were\s+a|shows?\s+that|demonstrates?)\b',
            re.IGNORECASE,
        )

        lines = body.split("\n")
        cleaned_lines = []
        seen_experiments = False

        for idx, line in enumerate(lines):
            s = line.strip()
            if not s:
                cleaned_lines.append(line)
                continue

            # Remove running page headers/footers & IEEE/MDPI/SAE copyright lines
            if re.match(
                r'^(?:#+\s*)?(?:'
                r'Volume\s+\d+,\s*20\d{2}|'
                r'IEEE\s+OPEN\s+JOURNAL\s+OF\s+INTELLIGENT\s+TRANSPORTATION\s+SYSTEMS|'
                r'Authorized\s+licensed\s+use\s+limited\s+to:.*|'
                r'\d{4}-\d{4}/\d{2}/\$31\.00\s*©\s*20\d{2}\s*IEEE|'
                r'Doi\s+10\.\d{4,9}/[^\s]+|'
                r'Issn:\s*\d{4}-\d{3}[\dX]|'
                r'Sensors\s+20\d{2},\s*\d+,\s*\d+|'
                r'Vehicles\s+20\d{2},\s*\d+,\s*\d+|'
                r'\d{1,2}\s+of\s+\d{1,2}|'
                r'Downloaded\s+from\s+http.*|'
                r'©\s*20\d{2}\s+The\s+Authors\..*'
                r')$',
                s,
                re.IGNORECASE,
            ):
                continue

            # Handle SAE inline headings separated by EN SPACE (\u2002)
            if "\u2002" in s:
                if s.startswith("#"):
                    h_part, rest_part = s.lstrip("#").strip().split("\u2002", 1)
                    cleaned_lines.append(f"\n# {h_part.strip().title()}\n")
                    if rest_part.strip():
                        cleaned_lines.append(rest_part.strip())
                    continue
                m_sae = re.match(r'^(\d+(?:\.\d+)*\.?\s+)([^\u2002]{2,45})\u2002+\s*(.*)$', s)
                if m_sae:
                    cleaned_lines.append(f"\n# {m_sae.group(2).strip().title()}\n")
                    if m_sae.group(3).strip():
                        cleaned_lines.append(m_sae.group(3).strip())
                    continue

            # Convert long numbered section headers missed by PDFParser
            if not s.startswith("#") and not s.endswith("."):
                m_num_hdr = long_numbered_hdr_pat.match(s)
                if m_num_hdr and not non_header_prose_pat.search(m_num_hdr.group(1)):
                    cleaned_lines.append(f"\n# {m_num_hdr.group(1).strip().title()}\n")
                    continue

            if s.startswith("#"):
                h_title = re.sub(r'^#+\s*', '', s).strip()
                h_title_clean = re.sub(r'^\d{1,2}(?:\.\d+)*\.?\s+', '', h_title).strip()
                if h_title_clean != h_title:
                    h_title = h_title_clean.title()
                    line = f"# {h_title}"

                if re.match(
                    r'^(?:Experiments?|Experimental\s+Setup|Experimental\s+Evaluation|Experiment\s+And\s+Results|Main\s+Results)\b',
                    h_title,
                    re.I,
                ):
                    seen_experiments = True

                if h_title.lower() in ("method", "results", "comparison"):
                    lookahead = []
                    for nxt_l in lines[idx + 1: idx + 12]:
                        if nxt_l.strip():
                            lookahead.append(nxt_l.strip())
                        if len(lookahead) >= 5:
                            break
                    lookahead_str = " ".join(lookahead)
                    is_table_header = bool(
                        re.search(r'(?:↑|↓|PSNR|SSIM|LPIPS|PDMS|EPDMS|FVMD|DISTS|samples|r@0\.|Venue\b|Method\b)', lookahead_str, re.I)
                        or (h_title.lower() == "method" and seen_experiments)
                    )
                    if is_table_header:
                        cleaned_lines.append(h_title)
                        continue

                is_false_header = (
                    bool(re.search(r'[↑↓=\[\]@×]', h_title))
                    or bool(re.match(r'^\(.*\)$', h_title))
                    or bool(re.match(r'^\d+\)\s+', h_title))
                    or bool(re.match(
                        r'^(?:Psnr|Ssim|Lpips|Pdms|Epdms|Fvmd|Dists|Par\.|Carla|Navsim\.?|Clip|Unchanged|'
                        r'Sdxl\s+Vae\.?|V-Jepa2|Interact|Nvidia|\(Nvidia|Idiada|Fmdc|Cgicvstb|Cuicvpg|'
                        r'Cvii[‐\-]Adtad|Hncvp|Nicvctz|Nitctsj|Nicvtssh|Niastsbh|Adas|Cavm|Cepra|Smdl|'
                        r'Unece|Srdm|Fmcw|Radar|Yolo|Davis|Dcae|Tron|Grand|Cdgnss|Orbits|Socp/Qp62|'
                        r'Aot\s*\+\s*Sedf|Spn\s*\+\s*Sedf|Cdgnss\s*\+\s*Iar\s*\+\s*Sedf|Antcom\s+1\.9G1215A|'
                        r'Ijcb\s*-\s*Tpr@Fpr)$',
                        h_title,
                        re.IGNORECASE,
                    ))
                )
                if is_false_header:
                    cleaned_lines.append(h_title)
                    continue

            cleaned_lines.append(line)

        body = "\n".join(cleaned_lines)
        body = cls._merge_consecutive_headers(body)
        body = cls._remove_boilerplate_sections(body)
        return body

    @staticmethod
    def _merge_consecutive_headers(text: str) -> str:
        """
        Handles consecutive '# Header' lines with no body text between them:
        - If 3+ headers appear back-to-back with no body text, they are diagram/flowchart labels -> un-header them.
        - If 2 headers appear back-to-back, merges wrapped titles or formats 'Parent — Subsection'.
        """
        lines = text.split("\n")
        out = []
        i = 0
        wrapped_end_pat = re.compile(
            r'(?:\b(?:Of|The|To|For|In|And|With|On|By|Related|Presented|Preferred|Several|Fmu|Impairments|Guided|Autonomous)|-)$',
            re.IGNORECASE,
        )
        while i < len(lines):
            s = lines[i].strip()
            if s.startswith("# "):
                run_indices = [i]
                j = i + 1
                while j < len(lines):
                    if not lines[j].strip():
                        j += 1
                        continue
                    if lines[j].strip().startswith("# "):
                        run_indices.append(j)
                        j += 1
                        continue
                    break

                if len(run_indices) >= 3:
                    for r_idx in run_indices:
                        out.append(lines[r_idx].strip()[2:].strip())
                    i = run_indices[-1] + 1
                    continue
                elif len(run_indices) == 2:
                    h1 = lines[run_indices[0]].strip()[2:].strip()
                    h2 = lines[run_indices[1]].strip()[2:].strip()
                    if h1.lower() == h2.lower():
                        merged_h = f"# {h1}"
                    elif h1.lower() == "description" and h2.lower().startswith("background"):
                        merged_h = f"# {h2}"
                    elif wrapped_end_pat.search(h1) or len(h2.split()) == 1:
                        merged_h = f"# {h1} {h2}"
                    else:
                        merged_h = f"# {h1} — {h2}"
                    out.append(merged_h)
                    i = run_indices[1] + 1
                    continue
            out.append(lines[i])
            i += 1
        return "\n".join(out)

    @staticmethod
    def _remove_boilerplate_sections(text: str) -> str:
        """Removes non-technical sections like Acknowledgments, Ethics/LLM Statements, and NeurIPS Checklists."""
        boilerplate_sec_pat = re.compile(
            r'^(?:'
            r'Acknowledgments?|Acknowledgements?|Aknowledgement|'
            r'Ai\s+Use\s+Statement|Declaration\s+Of\s+Llm\s+Usage|Use\s+Of\s+Generative\s+Ai\s+Tools|'
            r'Reproducibility\s+Statement|Code\s+Of\s+Ethics|Broader\s+Impacts|Safeguards|'
            r'Licenses\s+For\s+Existing\s+Assets|New\s+Assets|Crowdsourcing\s+And\s+Research\s+With\s+Human\s+Subjects|'
            r'Theory\s+Assumptions\s+And\s+Proofs|Experimental\s+Result\s+Reproducibility|'
            r'Open\s+Access\s+To\s+Data\s+And\s+Code|Experiment\s+Statistical\s+Significance|'
            r'Experiments\s+Compute\s+Resources|Article\s+Info|History|Keywords|Index\s+Terms'
            r')$',
            re.IGNORECASE,
        )

        parts = re.split(r'(?=(?:^|\n)#\s+)', text)
        kept_parts = []
        for part in parts:
            p_strip = part.strip()
            if not p_strip:
                continue
            if p_strip.startswith("# "):
                first_nl = p_strip.find("\n")
                sec_title = p_strip[2:first_nl].strip() if first_nl != -1 else p_strip[2:].strip()
                sec_body = p_strip[first_nl + 1:].strip() if first_nl != -1 else ""
                if boilerplate_sec_pat.match(sec_title):
                    continue
                if sec_body.startswith("Question:") and "Answer: [" in sec_body[:300]:
                    continue
            kept_parts.append(part)
        return "".join(kept_parts)

    @staticmethod
    def _qualify_and_coalesce_sections(
        sections: List[Tuple[str, str]],
        min_chars: int = 450,
    ) -> List[Tuple[str, str]]:
        """
        1. Prefixes generic subsection titles (e.g., 'Data Acquisition', 'First Iteration',
           'Second Iteration', 'Plausibility Verification', 'Validation Of Sensor-Specific Impairments Modeling')
           with their active parent section header so every chunk carries rich semantic section context.
        2. Coalesces short adjacent sections (< min_chars) so no tiny stub chunks are created.
        """
        if not sections:
            return []

        generic_subsections = re.compile(
            r'^(?:'
            r'Data\s+Acquisition|First\s+Iteration|Second\s+Iteration|'
            r'Plausibility\s+Verification|Validation\s+Of\s+Sensor-Specific\s+Impairments\s+Modeling|'
            r'Simulation\s+Controller|Sensor\s+Module|Phase\s+Noise|Rf\s+Group\s+Delay|'
            r'Phase\s+Imbalance|Mixer\s+Non-Linearity|Receiver\s+Noise\s+And\s+Noise\s+Figure|'
            r'Signal\s+Processing|Object\s+Recognition|Angles\s+Estimation|'
            r'Setup|Calibration|Testing|Analysis|Main\s+Results|Ablation\s+Study|Ablations|'
            r'Qualitative\s+Results|Implementation\s+Details|Implement\s+Details'
            r')$',
            re.IGNORECASE,
        )

        qualified: List[Tuple[str, str]] = []
        current_major = None
        for sec_name, sec_body in sections:
            if sec_name.lower() in ("abstract", "claims", "general context"):
                qualified.append((sec_name, sec_body))
                continue
            if "—" in sec_name:
                current_major = sec_name.split("—")[0].strip()
                qualified.append((sec_name, sec_body))
            elif generic_subsections.match(sec_name) and current_major:
                qualified.append((f"{current_major} — {sec_name}", sec_body))
            else:
                current_major = sec_name
                qualified.append((sec_name, sec_body))

        # Coalesce short sections (< min_chars) into the following section (except Abstract and Claims)
        merged: List[Tuple[str, str]] = []
        i = 0
        while i < len(qualified):
            sec_name, sec_body = qualified[i]
            if (
                len(sec_body.strip()) < min_chars
                and sec_name.lower() not in ("abstract", "claims")
                and i + 1 < len(qualified)
                and qualified[i + 1][0].lower() not in ("abstract", "claims")
            ):
                nxt_name, nxt_body = qualified[i + 1]
                if nxt_name.startswith(sec_name):
                    combined_name = nxt_name
                elif sec_name == "General Context":
                    combined_name = nxt_name
                else:
                    combined_name = f"{sec_name} / {nxt_name}"
                combined_body = f"{sec_body.strip()}\n\n{nxt_body.strip()}".strip()
                qualified[i + 1] = (combined_name, combined_body)
                i += 1
                continue
            merged.append((sec_name, sec_body))
            i += 1

        # Also coalesce a trailing short section (< min_chars) into the preceding section
        if (
            len(merged) >= 2
            and len(merged[-1][1].strip()) < min_chars
            and merged[-1][0].lower() not in ("abstract", "claims")
            and merged[-2][0].lower() not in ("abstract", "claims")
        ):
            prev_name, prev_body = merged[-2]
            last_name, last_body = merged[-1]
            merged[-2] = (
                f"{prev_name} / {last_name}",
                f"{prev_body.strip()}\n\n{last_body.strip()}".strip(),
            )
            merged.pop()

        return merged

    # ------------------------------------------------------------------
    # Main Chunking Entrypoint
    # ------------------------------------------------------------------

    def chunk(self, standardized_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunked_results = []
        for doc in standardized_docs:
            raw_text = doc["text"]
            metadata = doc["metadata"]
            source = metadata.get("source", "")

            title = metadata.get("title", "Unknown Title")
            pub_date = metadata.get("published_date", "Unknown Date")

            # Pass 0: Pre-Chunking Sanitization & Structural Header Recovery
            clean_text = self.sanitize_and_clean_for_chunking(raw_text, source=source)

            # Pass 1: Structural Split
            md_splits = self.markdown_splitter.split_text(clean_text)

            raw_sections: List[Tuple[str, str]] = []
            if not md_splits:
                if clean_text.strip():
                    raw_sections.append(("General Context", clean_text.strip()))
            else:
                for md_doc in md_splits:
                    sec_text = md_doc.page_content.strip()
                    if not sec_text:
                        continue
                    sec_meta = md_doc.metadata
                    sec_name = sec_meta.get("Header 1", sec_meta.get("Header 2", "General Context"))
                    raw_sections.append((sec_name, sec_text))

            sections = self._qualify_and_coalesce_sections(raw_sections, min_chars=450)

            global_chunk_idx = 0

            for section_name, section_text in sections:
                # Pass 2: Recursive Sub-splitting
                sub_splits = self.recursive_splitter.split_text(section_text)

                # Coalesce tiny final tail remainder (< 400 chars) into the preceding sub-split
                if len(sub_splits) >= 2 and len(sub_splits[-1]) < 400:
                    sub_splits[-2] = f"{sub_splits[-2].rstrip()}\n{sub_splits[-1].lstrip()}"
                    sub_splits.pop()

                for sub_text in sub_splits:
                    # Pass 3: Metadata Injection
                    injected_text = (
                        f"Document Title: {title}\n"
                        f"Publication Date: {pub_date}\n"
                        f"Section: {section_name}\n"
                        f"---\n"
                        f"{sub_text}"
                    )

                    chunk_record = {
                        "id": f"{doc['id']}_chunk_{global_chunk_idx}",
                        "parent_id": doc["id"],
                        "text": injected_text,
                        "metadata": {
                            **metadata,
                            "parent_id": doc["id"],
                            "section": section_name,
                            "chunk_index": global_chunk_idx,
                        },
                    }
                    chunked_results.append(chunk_record)
                    global_chunk_idx += 1

        return chunked_results
