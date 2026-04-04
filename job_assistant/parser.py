"""
parser.py
Ingest job spec and CV from text, .txt, .pdf, or .docx files.
Returns plain strings for downstream processing.
"""

from pathlib import Path


def read_file(path: str) -> str:
    """Read a file and return its text content regardless of format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = p.suffix.lower()

    if suffix == ".txt":
        return p.read_text(encoding="utf-8")

    elif suffix == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(str(p))
            return "\n".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
        except ImportError:
            raise ImportError("Install pypdf to read PDFs: pip install pypdf")

    elif suffix == ".docx":
        try:
            import docx
            doc = docx.Document(str(p))
            return "\n".join(
                para.text for para in doc.paragraphs if para.text.strip()
            )
        except ImportError:
            raise ImportError(
                "Install python-docx to read Word files: pip install python-docx"
            )

    else:
        # Try reading as plain text for unknown extensions
        return p.read_text(encoding="utf-8")


def parse_job_spec(source: str) -> str:
    """
    Return job spec text from a file path or raw string.
    If source looks like a file path and exists, read it.
    Otherwise treat as raw text.
    """
    p = Path(source)
    if p.exists() and p.is_file():
        return read_file(source)
    return source


def parse_cv(source: str) -> str:
    """
    Return CV text from a file path or raw string.
    If source looks like a file path and exists, read it.
    Otherwise treat as raw text.
    """
    p = Path(source)
    if p.exists() and p.is_file():
        return read_file(source)
    return source


def read_pdf_ocr(path: str) -> str:
    """OCR fallback for image-based PDFs."""
    from pdf2image import convert_from_path
    import pytesseract
    images = convert_from_path(path)
    return "\n".join(pytesseract.image_to_string(img) for img in images)
