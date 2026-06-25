from pathlib import Path  # Used to handle file paths easily
import chromadb 
from fastapi import FastAPI, HTTPException, UploadFile,File # HTTPException the API return clear error message, UploadFile allow the API to receive uploaded files.
from pydantic import BaseModel  # Used to define request body structure
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Define the directory where your data files are stored
DATA_DIR = Path("../data")

# Only these file types will be processed
SUPPORTED_EXTENSIONS = {".txt", ".md", ".py"}

# Define chunk size should be
CHUNK_SIZE = 500

MAX_FILE_SIZE = 1_000_000


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
    # A query cannot work until files have been ingested.
    stored_count = collection.count()

    if stored_count == 0:
        raise HTTPException(
            status_code=400,
            detail="No chunks found. Run POST /ingest first."
        )

    # Convert the question into the same embedding format as our chunks.
    question_embedding = embedding_model.encode(
        request.question
    ).tolist()

    # Never request more results than the collection contains.
    result_count = min(3, stored_count)

    # Retrieve the closest chunks from ChromaDB.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=result_count,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # Keep only chunks that are close enough to the question.
    # Smaller distance means better match.
    MAX_DISTANCE = 1.2

    filtered_documents = []
    filtered_metadatas = []

    for document, metadata, distance in zip(documents, metadatas, distances):
        if distance <= MAX_DISTANCE:
            filtered_documents.append(document)
            filtered_metadatas.append(metadata)

    if not filtered_documents:
        return {
            "question": request.question,
            "answer": "I could not find relevant information in the uploaded project files.",
            "sources": []
        }


    # Build context containing both chunk text and its source.
    context_parts = []

    for document, metadata in zip(filtered_documents, filtered_metadatas):
        context_parts.append(
            f"Source: {metadata['source']}\nContent: {document}"
        )

    context = "\n\n".join(context_parts)

    # Restrict the local model to information found in the project.
    prompt = f"""
You are DevRAG, a developer assistant.

Answer using only the supplied context.
If the answer is missing from the context, say:
"I could not find that information in the project files."

Mention the source filename used in the answer.

Context:
{context}

Question:
{request.question}

Answer:
"""

    try:
        # Ask the Ollama model running on this computer.
        ollama_response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5-coder:1.5b",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        # Treat unsuccessful Ollama responses as errors.
        ollama_response.raise_for_status()

    except requests.RequestException as error:
        # Return a useful API error instead of a long Python traceback.
        raise HTTPException(
            status_code=503,
            detail="Could not reach Ollama. Make sure Ollama is running."
        ) from error

    answer = ollama_response.json()["response"].strip()

    
# Each source tells us which file and which chunk was used.
    sources = []

    for metadata in filtered_metadatas:
        source_info = {
            "filename": metadata["source"],
            "chunk_index": metadata["chunk_index"]
        }

        if source_info not in sources:
            sources.append(source_info)


    return {
        "question": request.question,
        "answer": answer,
        "sources": sources
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Ensure the uploaded file has a name.
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename."
        )

    # Remove folder information from unsafe names such as ../../file.txt.
    safe_filename = Path(file.filename).name

    # Convert the extension to lowercase before validation.
    extension = Path(safe_filename).suffix.lower()

    # Reject unsupported file formats.
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .txt, .md, and .py files are supported."
        )

    # Read the uploaded file into memory.
    file_bytes = await file.read()

        # Do not allow empty files.
    if len(file_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    # Prevent unexpectedly large uploads.
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File must be smaller than 1 MB."
        )

    try:
        # Confirm that the file contains readable UTF-8 text.
        content = file_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=400,
            detail="File must contain valid UTF-8 text."
        ) from error

    # Save the validated file inside the data folder.
    destination = DATA_DIR / safe_filename
    destination.write_text(content, encoding="utf-8")

    return {
        "message": "File uploaded successfully",
        "filename": safe_filename,
        "size_bytes": len(file_bytes)
    }


@app.get("/status")
def project_status():
    # Count supported files inside the data folder.
    files = []

    for file_path in DATA_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(file_path.name)

    # Check how many chunks are currently stored in ChromaDB.
    stored_chunks = collection.count()

    # Check whether Ollama is running on this computer.
    try:
        ollama_response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=5
        )

        ollama_running = ollama_response.status_code == 200

    except requests.RequestException:
        ollama_running = False

    return {
        "api": "running",
        "supported_files_count": len(files),
        "supported_files": files,
        "stored_chunks": stored_chunks,
        "ollama_running": ollama_running
    }