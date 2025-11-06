"""
Chat service for RAG-based Q&A on extracted documents
Handles submission_id specific conversations and embeddings
"""
import os
import json
import glob
from typing import List, Dict, Optional
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from dotenv import load_dotenv
from config.settings import OUTPUT_DIR, CHROMA_STORE_PATH

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Create persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_PATH)

# Define OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Store collections and conversation memory per submission_id
submission_collections: Dict[str, chromadb.Collection] = {}
submission_memories: Dict[str, List[Dict]] = {}


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Chunk text with overlap for better context preservation."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def load_json_files_for_submission(submission_id: str, from_s3: bool = False) -> List[Dict]:
    """Load all JSON files from the submission-specific outputs folder or S3."""
    from services.extract import get_latest_json_files
    
    docs = []
    
    if from_s3:
        # Get files from S3 using existing helper function
        json_files = get_latest_json_files(submission_id=submission_id, from_s3=True)
        print(f"📥 Loaded {len(json_files)} JSON files from S3 for submission {submission_id}")
    else:
        # Get files from local filesystem (backward compatibility)
        submission_output_dir = os.path.join(OUTPUT_DIR, submission_id)
        
        if not os.path.exists(submission_output_dir):
            return docs
        
        json_files = glob.glob(os.path.join(submission_output_dir, "*.json"))
        print(f"📥 Loaded {len(json_files)} JSON files from local filesystem for submission {submission_id}")
    
    if not json_files:
        return docs
    
    # Load JSON content from files
    for file_path in json_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                content = json.dumps(data, indent=2)
                docs.append({"source": os.path.basename(file_path), "content": content})
        except Exception as e:
            print(f"⚠️ Skipped {file_path}: {e}")
    
    return docs


def get_or_create_collection(submission_id: str, force_recreate: bool = False) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a specific submission_id."""
    collection_name = f"submission_{submission_id}"
    
    # If force_recreate, delete existing collection first
    if force_recreate:
        try:
            chroma_client.delete_collection(name=collection_name)
            if submission_id in submission_collections:
                del submission_collections[submission_id]
        except:
            pass
    
    # Always try to get or create with OpenAI embedding function (ensures OpenAI is used)
    try:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        submission_collections[submission_id] = collection
        return collection
    except Exception as e:
        # If get_or_create fails, try to get existing collection
        try:
            collection = chroma_client.get_collection(name=collection_name)
            # If collection exists but doesn't have embedding function, we need to recreate it
            # But we can't modify embedding function on existing collection, so we'll use it as is
            # However, when querying, we should ensure embedding function is used
            submission_collections[submission_id] = collection
            return collection
        except Exception as e2:
            # If get also fails, try to create new collection
            try:
                collection = chroma_client.create_collection(
                    name=collection_name,
                    embedding_function=openai_ef
                )
                submission_collections[submission_id] = collection
                return collection
            except Exception as e3:
                raise Exception(f"Failed to create or get collection '{collection_name}': {str(e)} (get: {str(e2)}, create: {str(e3)})")


def initialize_submission_embeddings(submission_id: str, from_s3: bool = False):
    """Initialize embeddings for a submission by loading and chunking JSON files."""
    documents = load_json_files_for_submission(submission_id, from_s3=from_s3)
    if not documents:
        return
    
    collection_name = f"submission_{submission_id}"
    
    # Clear existing collection if it exists
    try:
        chroma_client.delete_collection(name=collection_name)
        # Remove from cache if it was there
        if submission_id in submission_collections:
            del submission_collections[submission_id]
    except:
        pass
    
    # Create new collection
    try:
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        submission_collections[submission_id] = collection
    except Exception as e:
        # If collection creation fails, try to get existing one
        try:
            collection = chroma_client.get_collection(name=collection_name)
            submission_collections[submission_id] = collection
            # Clear existing data by deleting and recreating
            chroma_client.delete_collection(name=collection_name)
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=openai_ef
            )
            submission_collections[submission_id] = collection
        except:
            raise Exception(f"Failed to initialize collection: {str(e)}")
    
    texts = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["content"])):
            texts.append({
                "id": f"{doc['source']}_{i}",
                "text": chunk,
                "source": doc["source"]
            })
    
    if texts:
        # Add in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            collection.add(
                ids=[t["id"] for t in batch],
                documents=[t["text"] for t in batch],
                metadatas=[{"source": t["source"]} for t in batch]
            )
    
    print(f"✅ Initialized embeddings for submission {submission_id}: {len(documents)} files, {len(texts)} chunks")


def get_conversation_memory(submission_id: str) -> List[Dict]:
    """Get conversation memory for a submission_id."""
    if submission_id not in submission_memories:
        submission_memories[submission_id] = []
    return submission_memories[submission_id]


def add_to_memory(submission_id: str, role: str, content: str):
    """Add message to conversation memory for a submission_id."""
    memory = get_conversation_memory(submission_id)
    memory.append({"role": role, "content": content})
    if len(memory) > 8:  # Keep only recent 8 exchanges
        memory.pop(0)


def clear_memory(submission_id: str):
    """Clear conversation memory for a submission_id."""
    if submission_id in submission_memories:
        submission_memories[submission_id] = []


def retrieve_context(submission_id: str, query: str, top_k: int = 6) -> str:
    """Retrieve top-k relevant chunks from Chroma for a submission."""
    try:
        # Always get collection with OpenAI embedding function to ensure fast API-based embeddings
        collection = get_or_create_collection(submission_id)
        
        # Verify collection has data
        if collection.count() == 0:
            return "No documents indexed yet. Please ensure JSON files are available."
        
        # Expand query for insurance-related questions
        insurance_keywords = ["insurance", "worthy", "property", "risk", "coverage", "policy", "assessment", "underwriting"]
        if any(keyword in query.lower() for keyword in insurance_keywords):
            expanded_query = f"{query} insurance property risk assessment coverage deductible building construction hazards"
            query_to_use = expanded_query
        else:
            query_to_use = query
        
        # Query using OpenAI embeddings (same as chatbot.py)
        results = collection.query(query_texts=[query_to_use], n_results=top_k)
        if not results["documents"] or not results["documents"][0]:
            if query_to_use != query:
                results = collection.query(query_texts=[query], n_results=top_k)
            if not results["documents"] or not results["documents"][0]:
                return "No relevant context found in the JSON files."
        
        docs = results["documents"][0]
        scores = results["distances"][0] if results["distances"] else [0.0] * len(docs)
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else [{}] * len(docs)
        
        combined = [
            f"[Rank {i+1}, score={scores[i]:.3f}, Source: {metadatas[i].get('source', 'unknown')}]\n{docs[i]}" 
            for i in range(len(docs))
        ]
        return "\n\n".join(combined)
    except Exception as e:
        return f"Error retrieving context: {str(e)}"


async def answer_query(
    submission_id: str,
    query: str,
    insurance_type: str = "property_casualty",
    top_k: int = 6,
    temperature: float = 0.2,
    from_s3: bool = False
) -> str:
    """Generate an answer using retrieved context + chat memory for a submission."""
    if not query.strip():
        return "Please ask a question about the data."
    
    # Check if JSON files exist for this submission
    from services.extract import get_latest_json_files
    
    if from_s3:
        json_files = get_latest_json_files(submission_id=submission_id, from_s3=True)
        if not json_files:
            return f"No JSON files found in S3 for submission_id: {submission_id}. Please extract files first."
    else:
        # Backward compatibility: check local filesystem
        submission_output_dir = os.path.join(OUTPUT_DIR, submission_id)
        if not os.path.exists(submission_output_dir):
            return f"No documents found for submission_id: {submission_id}. Please extract files first."
        
        json_files = glob.glob(os.path.join(submission_output_dir, "*.json"))
        if not json_files:
            return f"No JSON files found for submission_id: {submission_id}. Please extract files first."
    
    # Ensure embeddings are initialized (automatically activates when JSON files are available)
    # Always use OpenAI embedding function for fast performance (same as chatbot.py)
    try:
        collection = get_or_create_collection(submission_id)
        doc_count = collection.count()
        if doc_count == 0:
            # Check if JSON files exist before initializing
            if json_files:
                print(f"📚 Initializing embeddings for submission {submission_id} with {len(json_files)} JSON files using OpenAI... (from_s3={from_s3})")
                initialize_submission_embeddings(submission_id, from_s3=from_s3)
                # Re-get collection after initialization (refresh cache)
                collection = get_or_create_collection(submission_id, force_recreate=False)
                if collection.count() == 0:
                    raise Exception(f"Failed to initialize embeddings: collection is empty after initialization")
            else:
                raise Exception(f"No JSON files found for submission {submission_id}")
        else:
            print(f"✅ Using existing collection with {doc_count} documents for submission {submission_id} (OpenAI embeddings)")
    except Exception as e:
        # If collection operations fail, try to initialize from scratch
        error_msg = str(e)
        if "does not exist" in error_msg or "Collection" in error_msg:
            # Collection doesn't exist, initialize it
            if json_files:
                print(f"📚 Collection not found, initializing embeddings for submission {submission_id}... (from_s3={from_s3})")
                initialize_submission_embeddings(submission_id, from_s3=from_s3)
                collection = get_or_create_collection(submission_id, force_recreate=False)
            else:
                raise Exception(f"No JSON files found for submission {submission_id} and collection doesn't exist")
        else:
            raise Exception(f"Error with collection for submission {submission_id}: {error_msg}")
    
    context = retrieve_context(submission_id, query, top_k)
    
    # Insurance-type specific context and guidance
    if insurance_type == "life":
        specialized_context = """
        You specialize in life insurance underwriting. When relevant, consider:
        - Medical information: pre-existing conditions, family history, medications, lifestyle factors
        - Risk factors: age, BMI, medical conditions, occupation
        - Policy considerations: standard vs. rated policies, term vs. permanent, coverage amounts, riders
        """
    else:  # property_casualty
        specialized_context = """
        You specialize in property & casualty insurance underwriting. When relevant, consider:
        - Property details: construction type, age, condition, protection features, location
        - Risk factors: natural hazards, occupancy types, neighboring properties, business operations
        - Policy considerations: coverage limits, deductibles, endorsements, risk mitigation
        """
    
    # Common instructions for all insurance types
    common_instructions = """
    Answer questions naturally and conversationally. Be professional, accurate, and helpful.
    
    When uncertain about specific details, acknowledge the limitations of the information provided.
    Avoid making definitive underwriting decisions, but provide guidance based on industry standards.
    
    Use all available information from the context naturally. 
    Make reasonable inferences based on the data provided. 
    Respond directly to the question asked - don't use formulaic phrases or templates.
    """
    
    # Build system prompt - more natural and conversational
    system_prompt = (
        f"You are a helpful insurance underwriter assistant specializing in {insurance_type.replace('_', ' ').title()} insurance. "
        f"You have access to insurance document data and help answer questions about it.\n\n"
        f"{specialized_context}\n\n"
        f"{common_instructions}"
    )
    
    # Build conversation history
    memory = get_conversation_memory(submission_id)
    messages = [{"role": "system", "content": system_prompt}]
    messages += memory
    messages.append({
        "role": "user",
        "content": f"Context from documents:\n\n{context}\n\nQuestion: {query}"
    })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
        )
        
        answer = response.choices[0].message.content
        add_to_memory(submission_id, "user", query)
        add_to_memory(submission_id, "assistant", answer)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"


def reload_submission_embeddings(submission_id: str) -> str:
    """Reload embeddings for a submission."""
    try:
        initialize_submission_embeddings(submission_id)
        collection = get_or_create_collection(submission_id)
        return f"✅ Reloaded embeddings for submission {submission_id}. Collection has {collection.count()} chunks."
    except Exception as e:
        return f"❌ Error reloading embeddings: {str(e)}"

