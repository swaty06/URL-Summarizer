

from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

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
    :param urls: input urls
    :return:
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
    Generate answer using modern RAG chain with sources
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    
    # Create retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    # Create prompt that includes source tracking
    system_prompt = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, say that you don't know. 
Keep the answer concise and informative.

Context: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # Create chains
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    try:
        # Invoke the chain
        response = rag_chain.invoke({"input": query})
        
        # Extract answer
        answer = response.get("answer", "No answer generated.")
        
        # Extract and format sources
        source_docs = response.get("context", [])
        sources_list = []
        
        for doc in source_docs:
            source = doc.metadata.get("source", "")
            if source and source not in sources_list:
                sources_list.append(source)
        
        # Join sources as comma-separated string (matching original format)
        sources = ", ".join(sources_list) if sources_list else ""
        
        return answer, sources
    
    except Exception as e:
        return f"Error generating answer: {str(e)}", ""

# Alternative: If you want the exact same interface as RetrievalQAWithSourcesChain
def generate_answer_legacy_format(query):
    """
    Generate answer with exact same output format as the old RetrievalQAWithSourcesChain
    """
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")
    
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    
    system_prompt = """Answer the question based only on the following context.
Also provide the sources used to answer the question.

Context: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    try:
        response = rag_chain.invoke({"input": query})
        
        answer = response.get("answer", "")
        source_docs = response.get("context", [])
        
        # Format sources
        sources_set = set()
        for doc in source_docs:
            source = doc.metadata.get("source", "")
            if source:
                sources_set.add(source)
        
        sources = ", ".join(sources_set)
        
        return answer, sources
    
    except Exception as e:
        return f"Error: {str(e)}", ""

'''
if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    
    # Process URLs
    for status in process_urls(urls):
        print(status)
    
    # Generate answer
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Sources: {sources}")
'''
