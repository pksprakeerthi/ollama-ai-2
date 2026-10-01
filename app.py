import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

st.set_page_config(page_title="Mini RAG")
st.title("Mini RAG: Document Store + Retrieval")
st.caption("PDF Chunks Embeddings ChromaDB Retrieval Ollama")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-V2")


model = load_embedding_model()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="document")

st.sidebar.header("Settings")
ollama_model = st.sidebar.text_input("Ollama model", "llama3.2")
chunk_size = st.sidebar.slider("Chunk Size", 200, 1500, 500, 100)
top_k = st.sidebar.slider("Chunks to retrieve", 1, 5, 3)

st.header("Build Document Store")
uploaded_file = st.file_uploader("upload a text based PDF", type=["pdf"])

if uploaded_file and st.button("Process & store PDF"):
    reader = PdfReader(uploaded_file)
    text = ""
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        st.error("No readable text was found. Try a text-based PDF.")
        st.stop()

    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size].strip()
        if chunk:
            chunks.append(chunk)

    with st.spinner("Generating embeddings..."):
        embeddings = model.encode(chunks)

    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    collection.add(
        ids=[f"chunk_{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=[
            {"source": uploaded_file.name, "chunk": i}
            for i in range(len(chunks))
        ],
    )

    st.success(f"Stored {len(chunks)} chunks in ChromaDB")
    with st.expander("Preview stored chunks"):
        for i, chunk in enumerate(chunks[:5]):
            st.write(f"Chunk {i + 1}:")
            st.write(chunk)
            st.divider()

st.header("Ask questions")
question = st.text_input(
    "Ask a question about your uploaded document"
)
if st.button(" Retrieve & answer"):
    if not question.strip():
        st.warning("Please enter question")
        st.stop()

    count = collection.count()

    if count == 0:
        st.warning("please upload and process a PDF")
        st.stop()

    with st.spinner("searching document..."):
        question_embedding = model.encode([question])[0]
        results = collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(top_k, count),
        )

    retrieved_chunks = results["documents"][0]
    st.subheader("Retrieved Chunks")
    for i, chunk in enumerate(retrieved_chunks):
        with st.expander(f"Retrieved Chunk {i + 1}"):
            st.write(chunk)

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
You are a helpful question-answering assistant.

Answer the user's question using ONLY the context provided below.
If the answer is not present in the context, say:
"I could not find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    with st.spinner(f"Generate answer with {ollama_model}..."):
        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            answer = response["message"]["content"]
            st.subheader("Answer:")
            st.write(answer)
        except Exception as e:
            st.error(
                f"Could not connect to Ollama. Please make sure Ollama is running "
                f"and the model '{ollama_model}' is installed.\n\nError: {e}"
            )