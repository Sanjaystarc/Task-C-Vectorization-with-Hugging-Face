import streamlit as st
from sentence_transformers import SentenceTransformer, util
from pypdf import PdfReader
import numpy as np

# ------------------------------------
# Load Embedding Model
# ------------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-small-v2")

model = load_model()

st.title("📚 Vector Search & Basic Similarity Test — E5-small-v2")
st.write("Upload a PDF or use sample sentences and run basic similarity tests.")

# ------------------------------------
# PDF extraction
# ------------------------------------
def extract_pdf_sentences(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    sentences = [line.strip() for line in text.split("\n") if line.strip()]
    return sentences


# ------------------------------------
# Sidebar Controls
# ------------------------------------
st.sidebar.header("📂 Input Source")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_pdf:
    sentences = extract_pdf_sentences(uploaded_pdf)
    st.success(f"Loaded {len(sentences)} sentences from PDF.")
else:
    st.info("Using default example sentences.")
    sentences = [
        "AI agents are transforming automation across the world.",
        "Hugging Face provides powerful tools for model hosting.",
        "Vector embeddings help find semantic similarity between texts.",
        "The E5-small-v2 model is lightweight and fast.",
        "Python is commonly used for machine learning development."
    ]

# Show sentences
st.write("### 📌 Sentences Loaded")
for s in sentences:
    st.write("- ", s)


# ------------------------------------
# Basic Similarity Test Section
# ------------------------------------
st.write("## 🔍 Basic Similarity Test")

query = st.text_input("Enter a search query:")

if query:
    # Embed query (E5 prefix required)
    query_emb = model.encode("query: " + query, normalize_embeddings=True)

    # Embed sentences
    sentence_embs = model.encode(sentences, normalize_embeddings=True)

    # Cosine similarity
    scores = util.cos_sim(query_emb, sentence_embs)[0]

    # Sort results
    ranked = sorted(
        zip(sentences, scores), 
        key=lambda x: x[1], 
        reverse=True
    )

    st.write("### 🏆 Top Matches")
    for sentence, score in ranked:
        st.write(f"**Sentence:** {sentence}")
        st.write(f"**Similarity Score:** {float(score):.4f}")
        st.write("---")
