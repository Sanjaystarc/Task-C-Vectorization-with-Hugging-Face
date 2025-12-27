import streamlit as st
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import numpy as np

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("intfloat/e5-small-v2")

model = load_model()

st.title("📚 Vector Search with intfloat/e5-small-v2")
st.write("Upload a PDF or use sample text, embed sentences, and run similarity search.")

# -------------------------------
# PDF Extraction
# -------------------------------
def extract_pdf_sentences(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    sentences = [line.strip() for line in text.split("\n") if line.strip()]
    return sentences


# -------------------------------
# Sidebar Input
# -------------------------------
st.sidebar.header("📄 Data Source")

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

st.write("### 📌 Sentences Loaded")
for s in sentences:
    st.write("- ", s)


# -------------------------------
# Embedding Sentences
# -------------------------------
@st.cache_resource
def embed_sentences(sentences):
    return model.encode(sentences, normalize_embeddings=True)

sentence_embeddings = embed_sentences(sentences)

# -------------------------------
# Query Input
# -------------------------------
st.write("## 🔍 Run Similarity Search")

query = st.text_input("Enter your search query:", "")

if query:
    # prefix required for E5 models
    query_emb = model.encode("query: " + query, normalize_embeddings=True)

    # Cosine similarity
    similarities = np.dot(sentence_embeddings, query_emb)

    # Top results
    top_k = st.slider("Number of results:", 1, 5, 3)
    top_idx = np.argsort(similarities)[::-1][:top_k]

    st.write("### 🏆 Top Matches")
    for idx in top_idx:
        st.write(f"**Sentence:** {sentences[idx]}")
        st.write(f"**Score:** {similarities[idx]:.4f}")
        st.write("---")
