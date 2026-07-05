from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from functools import lru_cache

@lru_cache(maxsize=1)
def load_models():
    embedding_model = HuggingFaceEmbeddings( model_name="sentence-transformers/all-MiniLM-L6-v2")
    generator = pipeline(task="text-generation",model="HuggingFaceTB/SmolLM2-360M-Instruct")
    
    return embedding_model, generator


embedding_model, generator = load_models()


def extract_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    return text



def split_text(text):
    splitter = RecursiveCharacterTextSplitter( chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)



def create_vector_db(chunks):
    return FAISS.from_texts( chunks, embedding_model)



def process_document(pdf_path):

    text = extract_text(pdf_path)
    chunks = split_text(text)
    db = create_vector_db(chunks)
    stats = {
        "characters": len(text),
        "chunks": len(chunks),
        "chunk_size": 500,
        "chunk_overlap": 100,
        "embedding_dimension": 384,
        "embedding_model": "all-MiniLM-L6-v2",
        "vector_database": "FAISS"
    }
    return db, stats


def retrieve_context(db, question):
    docs = db.similarity_search(question, k=3)
    context = ""
    retrieved_chunks = []
    for doc in docs:
        context += doc.page_content + "\n"
        retrieved_chunks.append(doc.page_content)

    return context, retrieved_chunks


def answer_question(db, question):
    context, retrieved_chunks = retrieve_context( db, question)
    prompt = f"""
            You are an AI assistant answering questions about an uploaded document.
            Only use the retrieved context.
            If the answer is not found in the context, say:
            "I couldn't find the answer in the uploaded document."

            Do not make up information.
            Answer ONLY using the provided context.

            If the answer is not present,
            say:
            "I could not find the answer in the uploaded document."
            Context:
            {context}
            Question:
            {question}
            Answer:
            """

    result = generator(prompt,max_new_tokens=150,do_sample=False,temperature=0.2)
    answer = result[0]["generated_text"].replace(prompt, "").strip()

    return answer, retrieved_chunks