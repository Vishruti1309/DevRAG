from pathlib import Path  # Used to handle file paths easily
import chromadb 
from fastapi import FastAPI  # FastAPI framework to build APIs
from pydantic import BaseModel  # Used to define request body structure
from sentence_transformers import SentenceTransformer


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


@app.post("/store-chunks")
def store_chunks():
    # These lists will hold the data ChromaDB needs.
    ids = []
    documents = []
    metadatas = []

    # Visit every item inside the data folder.
    for file_path in DATA_DIR.iterdir():
        # Process only supported files.
        if file_path.is_file() and file_path.suffix in SUPPORTED_EXTENSIONS:
            # Read the complete file.
            content = file_path.read_text(encoding="utf-8")

            # Split the file into smaller pieces.
            chunks = chunk_text(content)

            # Prepare every chunk for storage.
            for index, chunk in enumerate(chunks):
                # Create a repeatable unique ID, such as notes.txt:0.
                chunk_id = f"{file_path.name}:{index}"

                ids.append(chunk_id)
                documents.append(chunk)

                # Metadata remembers where this chunk came from.
                metadatas.append({
                    "source": file_path.name,
                    "chunk_index": index
                })

    # Avoid calling the embedding model when no files were found.
    if not documents:
        return {
            "message": "No supported files found",
            "stored_chunks": 0
        }

    # Convert all chunks into 384-number vectors in one batch.
    embeddings = embedding_model.encode(documents).tolist()

    # Add new records or update existing records in ChromaDB.
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    return {
        "message": "Chunks stored successfully",
        "stored_chunks": len(documents),
        "collection_total": collection.count()
    }