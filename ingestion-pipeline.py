import os
from langchain_community.document_loaders import TextLoader, CSVLoader, DirectoryLoader, JSONLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


def load_documents(data_path):
    loader = DirectoryLoader(path=data_path, glob="*.csv", loader_cls=CSVLoader)
    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError("No documents found in the specified directory.")
    else:
        print(f"Loaded {len(documents)} documents.")
    
    return documents

'''
def load_documents(data_path):

    loader_kwargs = {
        "jq_schema": ".",          # prendi tutto l'oggetto della riga
        "content_key": "text",     # chiave per page_content
        "json_lines": True,        # per .jsonl
        "text_content": True       # converte in stringa
    }

    loader = DirectoryLoader(
        path=data_path,
        glob="**/*.jsonl",         # cerca ricorsivamente tutti i .jsonl (meglio di "*/*.jsonl")
        loader_cls=JSONLoader,
        loader_kwargs=loader_kwargs,
        show_progress=True         # opzionale, utile per debug
    )

    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError("No documents found in the specified directory.")
    else:
        print(f"Loaded {len(documents)} documents.")
    
    return documents
'''

def split_documents(documents, chunk_size, chunk_overlap):
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def create_vector_store(chunks, persist_directory):
    print("Creating vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )
    
    vector_store.persist()
    print("Vector store created and persisted.")
    return vector_store

if __name__ == "__main__":
    documents = load_documents("datasets")
    chunks =split_documents(documents, chunk_size = 800, chunk_overlap = 0)
    vector_store = create_vector_store(chunks, persist_directory="chroma_db")