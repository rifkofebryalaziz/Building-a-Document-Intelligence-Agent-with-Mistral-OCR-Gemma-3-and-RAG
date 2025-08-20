import streamlit as st
import os
import io
import tempfile
import requests

# Mistral 0.4.2 (legacy client) — TIDAK dipakai untuk OCR di versi ini
from mistralai.client import MistralClient

import google.generativeai as genai
from PIL import Image
from PyPDF2 import PdfReader

# --------------------- Page config ---------------------
st.set_page_config(page_title="Document Intelligence Agent", layout="wide")
st.title("Document Intelligence Agent")
st.markdown("Upload documents or images to extract information and ask questions")

# --------------------- Sidebar: API Keys ----------------
with st.sidebar:
    st.header("API Configuration")
    mistral_api_key = st.text_input("Mistral AI API Key (legacy 0.4.2)", type="password")
    google_api_key = st.text_input("Google API Key (Gemini)", type="password")

# MistralClient (legacy) — tidak untuk OCR di 0.4.2
mistral_client = None
if mistral_api_key:
    try:
        mistral_client = MistralClient(api_key=mistral_api_key)
        st.success("✅ Mistral API connected (legacy client 0.4.2) — OCR disabled")
    except Exception as e:
        st.error(f"Failed to initialize Mistral client: {e}")

# Gemini untuk OCR & QnA
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        st.success("✅ Google API connected")
    except Exception as e:
        st.error(f"Failed to initialize Google API: {e}")

# --------------------- Helpers (Gemini OCR) -------------------------
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Ekstrak teks dari PDF non-scan via PyPDF2 (cepat & lokal).
    Jika kosong (kemungkinan scan), nanti fallback ke Gemini OCR.
    """
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            parts.append(page.extract_text() or "")
        text = "\n".join(parts).strip()
        return text
    except Exception:
        return ""

def gemini_ocr_image(image_bytes: bytes) -> str:
    """OCR gambar (PNG/JPG) ke Markdown via Gemini."""
    img = Image.open(io.BytesIO(image_bytes))
    prompt = (
        "Convert this document image into clean Markdown. "
        "Preserve headings, lists, and tables (use Markdown tables). "
        "Maintain natural reading order."
    )
    model = genai.GenerativeModel("gemini-1.5-pro")
    resp = model.generate_content([prompt, img])
    return getattr(resp, "text", "").strip()

def gemini_ocr_pdf(pdf_bytes: bytes, filename: str = "upload.pdf") -> str:
    """OCR + ekstraksi PDF (termasuk scan) ke Markdown via Gemini."""
    # Simpan sementara agar bisa di-upload ke Gemini
    suffix = os.path.splitext(filename)[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        file_obj = genai.upload_file(
            path=tmp_path,
            mime_type="application/pdf",
            display_name=filename
        )
        prompt = (
            "Extract the full content of this PDF as clean Markdown. "
            "Preserve headings and tables. If pages are scanned, perform OCR first."
        )
        model = genai.GenerativeModel("gemini-1.5-pro")
        resp = model.generate_content([file_obj, prompt])
        return getattr(resp, "text", "").strip()
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

def process_document_with_gemini(kind: str, name: str, data: bytes) -> str:
    """
    Router sederhana:
    - PDF: coba PyPDF2 dulu (cepat). Jika terlalu pendek/kosong → fallback Gemini OCR.
    - Image: langsung pakai Gemini OCR.
    """
    if kind == "pdf":
        text = extract_text_from_pdf_bytes(data)
        # Jika hasil terlalu pendek (kemungkinan scan), fallback OCR via Gemini
        if len(text) >= 200:
            return text
        return gemini_ocr_pdf(data, filename=name)
    else:  # image
        return gemini_ocr_image(data)

def generate_response(context: str, query: str) -> str:
    """Jawab pertanyaan berdasarkan konteks dokumen menggunakan Gemini."""
    if not context or len(context) < 10:
        return "Error: Document context is empty or too short."
    try:
        prompt = f"""
You are a document analysis assistant with access to the following document content:

{context}

Answer the following question based solely on the document content above:
{query}

If the answer is not in the document, say so.
"""
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        resp = model.generate_content(prompt)
        return getattr(resp, "text", "No response text.")
    except Exception as e:
        return f"Error generating response: {e}"

# --------------------- UI Layout -----------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.header("Document Upload")
    upload_type = st.radio("Select upload type:", ["PDF", "Image", "URL"])

    uploaded_file = None
    url_input = None

    if upload_type in ["PDF", "Image"]:
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf"] if upload_type == "PDF" else ["png", "jpg", "jpeg"],
        )
    else:
        url_input = st.text_input("Enter document URL (PDF/PNG/JPG):")

    process_button = st.button("Process Document")

    if "ocr_content" not in st.session_state:
        st.session_state.ocr_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if process_button:
        # Karena OCR memakai Gemini, butuh Google API key
        if not google_api_key:
            st.error("Please provide a valid Google API Key for OCR/processing.")
        elif uploaded_file is not None:
            with st.spinner("Processing document..."):
                try:
                    ext = os.path.splitext(uploaded_file.name)[1].lower()
                    kind = "pdf" if ext == ".pdf" else "image"
                    st.session_state.ocr_content = process_document_with_gemini(
                        kind, uploaded_file.name, uploaded_file.getvalue()
                    )
                    if not st.session_state.ocr_content:
                        st.warning("No content extracted.")
                    else:
                        st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        elif url_input:
            with st.spinner("Downloading & processing from URL..."):
                try:
                    r = requests.get(url_input, timeout=30)
                    r.raise_for_status()
                    # Deteksi jenis konten berdasar ekstensi URL
                    clean_url = url_input.split("?")[0]
                    ext = os.path.splitext(clean_url)[1].lower()
                    kind = "pdf" if ext == ".pdf" else "image"
                    st.session_state.ocr_content = process_document_with_gemini(
                        kind, os.path.basename(clean_url) or "download", r.content
                    )
                    if not st.session_state.ocr_content:
                        st.warning("No content extracted.")
                    else:
                        st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        else:
            st.warning("Please upload a document or provide a URL.")

with col2:
    st.header("Document Q&A")

    if st.session_state.ocr_content:
        st.markdown("Document loaded. Ask questions about the content:")

        for m in st.session_state.chat_history:
            role = "You" if m["role"] == "user" else "Assistant"
            st.markdown(f"**{role}:** {m['content']}")

        user_q = st.text_input("Your question:")
        if st.button("Ask") and user_q:
            st.session_state.chat_history.append({"role": "user", "content": user_q})
            with st.spinner("Generating response..."):
                if not google_api_key:
                    ans = "Please provide a valid Google API Key."
                else:
                    ans = generate_response(st.session_state.ocr_content, user_q)
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()
    else:
        st.info("Please upload and process a document first.")

# Tampilkan konten hasil OCR/ekstraksi
if st.session_state.get("ocr_content"):
    with st.expander("View Extracted Document Content"):
        st.markdown(st.session_state.ocr_content)
