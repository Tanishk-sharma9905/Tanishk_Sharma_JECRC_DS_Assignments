import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self, corpus_dir="data/corpus", persist_dir="storage/chroma_db_v2"):
        self.corpus_dir = os.path.abspath(corpus_dir)
        self.persist_dir = os.path.abspath(persist_dir)
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.vectorstore = None
        self._initialize_store()

    def _initialize_store(self):
        # Task 4: Improve RAG Initialization (Don't rebuild if exists)
        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            print(f"Loading existing Chroma database from {self.persist_dir}")
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            return

        print("Creating new Chroma index...")
        if not os.path.exists(self.corpus_dir) or not os.listdir(self.corpus_dir):
            print(f"Warning: Corpus directory '{self.corpus_dir}' is empty or does not exist.")
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)
            return

        loader = DirectoryLoader(self.corpus_dir, glob="**/*.md", loader_cls=TextLoader)
        documents = loader.load()
        
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(documents)
            
            self.vectorstore = Chroma.from_documents(
                documents=splits, 
                embedding=self.embeddings,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore = Chroma(persist_directory=self.persist_dir, embedding_function=self.embeddings)

    def retrieve(self, query: str, k: int = 3) -> str:
        """
        Retrieves the top k most relevant context chunks for the query.
        Returns them as a single concatenated string.
        """
        if not self.vectorstore:
            return ""
        
        # Task 6: Improve Retrieval using MMR
        retriever = self.vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k})
        results = retriever.invoke(query)
        
        formatted_results = []
        for doc in results:
            # Task 5: Add Metadata to RAG
            source = doc.metadata.get("source", "Unknown Source")
            formatted_results.append(f"--- SOURCE: {source} ---\n{doc.page_content}")
            
        return "\n\n".join(formatted_results)
