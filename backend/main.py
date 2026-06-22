from pathlib import Path  # Used to handle file paths easily
import chromadb 
from fastapi import FastAPI  # FastAPI framework to build APIs
from pydantic import BaseModel  # Used to define request body structure
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Define the directory where your data files are stored
DATA_DIR = Path("../data")

# Only these file types will be processed
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py"}

# Define chunk size should be
CHUNK_SIZE = 100


CHROMA_DIR = Path("../chroma_db")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_or_create_collection(
    name="devrag_docs"
)

# Load the free local embedding model once when the backend starts.
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Define the structure of incoming POST request for /ask endpoint
class QuestionRequest(BaseModel):
    question: str  # User will send a question as string

# Function to split text into smaller chunks
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE):
    chunks = [] 
    # Loop through text in steps of chunk_size
    for start in range(0, len(text), chunk_size):
        end = start + chunk_size  # Define end index
        chunk = text[start:end]  # Slice the text
        chunks.append(chunk)  # Add chunk to list

    return chunks  # Return all chunks


# Root endpoint (just for testing)
@app.get("/")
def hello():
    return {"message": "Hello from DevRAG"}

# Health check endpoint (used to check if server is running)
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Endpoint to receive a question (currently just echoes it back)
@app.post("/ask")
def ask_question(request: QuestionRequest):
    return {
    "question": request.question,
    "message": "Question received"
    }


# Endpoint to read a single file (notes.txt)
@app.get("/read-file")
def read_file():
    file_path = DATA_DIR / "notes.txt"  # Build file path
    content = file_path.read_text(encoding="utf-8")

    return {
        "filename": "notes.txt",
        "content": content
    }
    

# Endpoint to read all supported files from the data folder
@app.get("/files")
def read_files():
    files = []  # List to store file data


    for file_path in DATA_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            
            # Read file content
            content = file_path.read_text(encoding="utf-8")

            # Add file data to list
            files.append({
                "filename": file_path.name,
                "content": content
            })

    return {
        "count": len(files),  # Total number of files
        "files": files        # List of file data
    }
    

# Endpoint to split all files into chunks 
@app.get("/chunks")
def get_chunks():
    all_chunks = []  # List to store all chunks

    
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            
            # Read file content
            content = file_path.read_text(encoding="utf-8")

            # Split content into chunks
            chunks = chunk_text(content)

            # Loop through each chunk and store metadata
            for index, chunk in enumerate(chunks):
                all_chunks.append({
                    "text": chunk,              # Actual chunk text
                    "source": file_path.name,   # File name (source)
                    "chunk_index": index        # Position of chunk
                })

    return {
        "count": len(all_chunks),  # Total number of chunks
        "chunks": all_chunks      # List of all chunks
    }
    

@app.get("/database")
def database_status():
    return {
        "collection": collection.name,
        "stored_chunks": collection.count()
    }

#embedding test
@app.get("/embedding-test")
def embedding_test():
    # This is the sample text we want to convert into numbers.
    text = "DevRAG answers questions about project files."

    # Generate the local embedding as a NumPy array.
    embedding = embedding_model.encode(text)

    return {
        "text": text,

        # all-MiniLM-L6-v2 normally creates 384 numbers.
        "dimensions": len(embedding),

        # Return only 8  numbers so the response stays readable.
        "preview": embedding[:8].tolist()
    }   


@app.post("/ingest")
def ingest_files():
    # Find records stored during an earlier ingestion.
    existing_records = collection.get()
    existing_ids = existing_records["ids"]

    # Clear old records so ChromaDB matches the current data folder.
    if existing_ids:
        collection.delete(ids=existing_ids)

    # Prepare the four lists needed by ChromaDB.
    ids = []
    documents = []
    metadatas = []

    # Read every supported file from the data folder.
    for file_path in DATA_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            content = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(content)

            # Give every chunk an ID and source information.
            for index, chunk in enumerate(chunks):
                ids.append(f"{file_path.name}:{index}")
                documents.append(chunk)
                metadatas.append({
                    "source": file_path.name,
                    "chunk_index": index
                })

    # Return early if the data folder has no supported files.
    if not documents:
        return {
            "message": "No supported files found",
            "stored_chunks": 0
        }

    # Generate embeddings locally for all chunks together.
    embeddings = embedding_model.encode(documents).tolist()

    # Store text, metadata, IDs, and embeddings in ChromaDB.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "message": "Ingestion completed",
        "files_processed": len({
            metadata["source"] for metadata in metadatas
        }),
        "stored_chunks": len(documents),
        "collection_total": collection.count()
    }


@app.post("/search")
def search_chunks(request: QuestionRequest):
    # Convert the user's question into the same 384-number format.
    question_embedding = embedding_model.encode(
    request.question
    ).tolist()

    # Find the three stored chunks closest to the question.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    # Create a beginner-friendly response instead of returning raw results.
    matches = []

    # ChromaDB returns one result list for each submitted question.
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Combine each retrieved document with its metadata and distance.
    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        matches.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"],
            "distance": distance
        })

    return {
        "question": request.question,
        "matches": matches
    }


@app.post("/query")
def answer_question(request: QuestionRequest):
    # Convert the question into a local embedding.
    question_embedding = embedding_model.encode(
        request.question
    ).tolist()

    # Retrieve the three most relevant chunks from ChromaDB.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "metadatas"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Add source names to the context given to the local LLM.
    context_parts = []

    for document, metadata in zip(documents, metadatas):
        context_parts.append(
            f"Source: {metadata['source']}\nContent: {document}"
        )

    context = "\n\n".join(context_parts)

    # Tell the model to use only retrieved project information.
    prompt = f"""
You are DevRAG, a developer assistant.

Answer the question using only the context below.
If the answer is not in the context, say:
"I could not find that information in the project files."

Mention the source filename used in your answer.

Context:
{context}

Question:
{request.question}

Answer:
"""

    # Send the prompt to Ollama running locally on port 11434.
    ollama_response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5-coder:1.5b",
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )

    # Raise an error if Ollama returned an unsuccessful response.
    ollama_response.raise_for_status()

    # Extract the generated text from Ollama's JSON response.
    answer = ollama_response.json()["response"].strip()

    # Return the answer and the source files used as context.
    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            metadata["source"] for metadata in metadatas
        ]
    }