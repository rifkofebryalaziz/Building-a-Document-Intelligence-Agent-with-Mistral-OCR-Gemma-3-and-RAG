import streamlit as st
import os
import tempfile
from mistralai.client import MistralClient
import google.generativeai as genai

# optional deps: Pillow (jika perlu tampilkan gambar), io/base64 jika dipakai
from PIL import Image  

# --------------------- Page config ---------------------
st.set_page_config(page_title="Document Intelligence Agent", layout="wide")
st.title("Document Intelligence Agent")
st.markdown("Upload documents or images to extract information and ask questions")

# --------------------- Sidebar: API Keys ----------------
with st.sidebar:
    st.header("API Configuration")
    mistral_api_key = st.text_input("Mistral AI API Key", type="password")
    google_api_key = st.text_input("Google API Key", type="password")

mistral_client = None
if mistral_api_key:
    try:
        mistral_client = MistralClient(api_key=mistral_api_key)
        st.success("✅ Mistral API connected")
    except Exception as e:
        st.error(f"Failed to initialize Mistral client: {e}")

if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        st.success("✅ Google API connected")
    except Exception as e:
        st.error(f"Failed to initialize Google API: {e}")

# --------------------- Helpers -------------------------
def upload_to_mistral(client, file_content: bytes, file_name: str) -> str:
    """Upload a file to Mistral's API and return a signed URL."""
    if client is None:
        raise ValueError("Mistral client is not initialized. Provide a valid API key.")

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file_name)
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)

        with open(file_path, "rb") as f:
            resp = client.files.create(file=f, purpose="ocr")

        signed_url = client.files.retrieve_content(resp.id)
        return signed_url
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

def replace_images_in_markdown(markdown: str, images: dict) -> str:
    """Replace image placeholders in markdown with base64 images."""
    for image_id, image_base64 in images.items():
        markdown = markdown.replace(
            f"![](/images/{image_id})",
            f"![](data:image/png;base64,{image_base64})",
        )
    return markdown

def get_combined_markdown(ocr_response) -> str:
    """Merge markdown of all pages with inlined images."""
    markdowns = []
    for page in ocr_response.pages:
        imgs = {img.id: img.base64 for img in page.images}
        page_md = replace_images_in_markdown(page.markdown, imgs)
        markdowns.append(page_md)
    return "\n\n---\n\n".join(markdowns)

def process_ocr(client, source: str, source_type: str = "document"):
    """Call Mistral OCR for a document or image URL."""
    if client is None:
        raise ValueError("Mistral client is not initialized.")

    if source_type == "document":
        return client.ocr.process(
            document_url=source,
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
    elif source_type == "image":
        return client.ocr.process(
            image_url=source,
            model="mistral-ocr-latest",
            include_image_base64=True,
        )
    else:
        raise ValueError(f"Unsupported source type: {source_type}")

def generate_response(context: str, query: str) -> str:
    """Answer using Gemini based only on provided context."""
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
        url_input = st.text_input("Enter document URL:")

    process_button = st.button("Process Document")

    if "ocr_content" not in st.session_state:
        st.session_state.ocr_content = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if process_button:
        if not mistral_client:
            st.error("Please provide a valid Mistral API Key.")
        elif uploaded_file is not None:
            with st.spinner("Processing document..."):
                try:
                    signed_url = upload_to_mistral(
                        mistral_client,
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )
                    ocr_resp = process_ocr(mistral_client, signed_url)
                    st.session_state.ocr_content = get_combined_markdown(ocr_resp)
                    st.success("Document processed successfully!")
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        elif url_input:
            with st.spinner("Processing document from URL..."):
                try:
                    ocr_resp = process_ocr(mistral_client, url_input)
                    st.session_state.ocr_content = get_combined_markdown(ocr_resp)
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
            st.markdown(f"**{'You' if m['role']=='user' else 'Assistant'}:** {m['content']}")

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

if st.session_state.get("ocr_content"):
    with st.expander("View Extracted Document Content"):
        st.markdown(st.session_state.ocr_content)
