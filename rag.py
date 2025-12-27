

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq


load_dotenv()

# Constants
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store
    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)
    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls):
    """
    This function scrapes data from a url and stores it in a vector db
    """
    yield "Initializing Components...✅"
    initialize_components()
    
    yield "Resetting vector store...✅"
    try:
        vector_store.delete_collection()
        vector_store._collection = vector_store._client.create_collection(
            name=vector_store._collection.name
        )
    except Exception as e:
        print(f"Error resetting collection: {e}")
    
    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    
    yield "Splitting text into chunks...✅"
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)
    
    yield "Adding chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)
    
    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    """
    Generate answer using simple RAG approach with sources
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    
    # Retrieve relevant documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
   
    docs = retriever.invoke(query)

    
    if not docs:
        return "No relevant information found.", ""
    
    # Format context and collect sources
    context_parts = []
    sources_set = set()
    
    for doc in docs:
        context_parts.append(doc.page_content)
        source = doc.metadata.get("source", "")
        if source:
            sources_set.add(source)
    
    context = "\n\n".join(context_parts)
    
    # Create prompt
    prompt_template = """Answer the question based on the following context. 
If you don't know the answer, say so clearly.

Context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Create simple chain
    chain = prompt | llm | StrOutputParser()
    
    try:
        answer = chain.invoke({"context": context, "question": query})
        sources = ", ".join(sources_set)
        
        return answer, sources
    
    except Exception as e:
        return f"Error generating answer: {str(e)}", ""
