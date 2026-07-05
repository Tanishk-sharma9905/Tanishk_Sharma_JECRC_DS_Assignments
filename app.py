import streamlit as st
import tempfile

from rag_backend import (
    process_document,
    answer_question
)


st.set_page_config(page_title="Simple RAG System",page_icon="📄",layout="wide")

st.title("📄 Retrieval-Augmented Generation (RAG)")
st.write("Upload a PDF document and ask questions based on its content.")


if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False


with st.sidebar:

    st.header("Upload Document")
    uploaded_pdf = st.file_uploader("Choose a PDF",type=["pdf"])

    if uploaded_pdf:

        with tempfile.NamedTemporaryFile(delete=False,suffix=".pdf") as tmp:

            tmp.write(uploaded_pdf.read())
            pdf_path = tmp.name
            st.success(f"Uploaded File : {uploaded_pdf.name}")
        with st.spinner("Processing document..."):
            vector_db, stats = process_document(pdf_path)


        st.session_state.vector_db = vector_db
        st.session_state.document_loaded = True
        st.success("Document indexed successfully!")
        st.markdown("### Document Statistics")
        col1, col2 = st.columns(2)
        col1.metric("Characters", stats["characters"])
        col2.metric("Chunks", stats["chunks"])
        col1.metric("Chunk Size", stats["chunk_size"])
        col2.metric("Overlap", stats["chunk_overlap"])

    st.markdown("---")
    st.markdown("### Project Info")
    st.write("Vector Database : FAISS")
    st.write("Embedding Model : all-MiniLM-L6-v2")
    st.write("LLM : SmolLM2-360M-Instruct")


if st.session_state.document_loaded:
    st.header("Ask Questions")
    question = st.text_input(
        "Enter your question"
    )
    if st.button("Generate Answer"):
        if question.strip() == "":
            st.warning("Please enter a question.")

        else:
            with st.spinner("Searching document..."):
                answer, retrieved_chunks = answer_question(
                    st.session_state.vector_db,
                    question
                )
            st.subheader("Answer")
            st.success(answer)
            with st.expander("Retrieved Context"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.markdown(f"### Chunk {i+1}")
                    st.write(chunk)

else:

    st.info("Please upload a PDF document to begin.")