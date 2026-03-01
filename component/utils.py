from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pathlib import Path
import faiss

class all_chatbot_utils:
    def __init__(self):
        pass

    def get_chunks(self):
        current_file = Path(__file__)  # components/utils.py
        root_dir = current_file.parent.parent  # root_project/
        pdf_path = root_dir/'pdf_doc.pdf'
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        content_from_pdf = " ".join(doc.page_content for doc in docs)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 350, chunk_overlap=0)
        d = text_splitter.create_documents([content_from_pdf])
        # Assuming 'docs' is your list of Document objects
        all_chunks = [doc.page_content for doc in d]
        return all_chunks
    
    def store_chunks_in_faiss(self,chunks):
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast, high-quality
        # Get embeddings
        embeddings = embedding_model.encode(chunks, convert_to_numpy=True)
        # Build FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)  # or IndexFlatIP for cosine similarity (with normalized vectors)
        # Add vectors
        index.add(embeddings)
        return index
    
    def fetch_similar_chunks(self,query,chunks,index_of_faiss):
        #Retrieve the most relevant chunks from the FAISS index.
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([query], convert_to_numpy=True)  # Embed the query
        _, indices = index_of_faiss.search(query_embedding, k=2)  # Search top-k
        all_similar_chunks = [chunks[i] for i in indices[0]]
        retrieved_context = " ".join(all_similar_chunks)
        return retrieved_context


