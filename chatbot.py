# ====================================================
# 🧠 Advanced RAG Chatbot on JSON files using ChromaDB + OpenAI
# ====================================================
import os
import json
import glob
import uuid
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from typing import List, Dict
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --------------------------
# 1. Environment setup
# --------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required. Please set it in your .env file.")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)

# Configuration
OUTPUT_DIR = "outputs"
CHROMA_STORE_PATH = "./chroma_store"

# Create persistent Chroma client
chroma_client = chromadb.PersistentClient(path=CHROMA_STORE_PATH)

# Unique collection name with UUID
collection_id = str(uuid.uuid4())[:8]
collection_name = f"json_docs_{collection_id}"

# Define OpenAI embedding function
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

# Create collection (persistent)
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=openai_ef
)

print(f"✅ Persistent ChromaDB collection created/loaded: {collection_name}")

# --------------------------
# 2. Load and chunk JSON files
# --------------------------
def load_json_files(folder_path: str = OUTPUT_DIR) -> List[Dict]:
    """Load all JSON files from the outputs folder."""
    docs = []
    json_files = glob.glob(f"{folder_path}/*.json")
    if not json_files:
        print(f"⚠️ No JSON files found in {folder_path}")
        return docs
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Handle both list and dict JSON files
                if isinstance(data, list):
                    content = json.dumps(data, indent=2)
                else:
                    content = json.dumps(data, indent=2)
                docs.append({"source": os.path.basename(file_path), "content": content})
            except Exception as e:
                print(f"⚠️ Skipped {file_path}: {e}")
    return docs

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """Chunk text with overlap for better context preservation."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap to preserve context
    return chunks

# --------------------------
# 3. Initialize database
# --------------------------
def initialize_database():
    """Load JSON files, chunk them, and store in ChromaDB."""
    documents = load_json_files()
    if not documents:
        return 0, 0
    
    texts = []
    for doc in documents:
        for i, chunk in enumerate(chunk_text(doc["content"])):
            texts.append({
                "id": f"{doc['source']}_{i}",
                "text": chunk,
                "source": doc["source"]
            })
    
    # Clear existing data and add new chunks
    global collection
    try:
        # Delete collection if exists and recreate
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
    except:
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
    
    if texts:
        # Add in batches to avoid memory issues
        batch_size = 100
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
            batch = texts[i:i+batch_size]
            collection.add(
                ids=[t["id"] for t in batch],
                documents=[t["text"] for t in batch],
                metadatas=[{"source": t["source"]} for t in batch]
            )
    
    return len(documents), len(texts)

# Initialize on startup
print("📚 Loading JSON files from outputs folder...")
num_files, num_chunks = initialize_database()
print(f"✅ Loaded {num_files} JSON files, created {num_chunks} chunks")

# --------------------------
# 4. Memory management
# --------------------------
conversation_memory = []

def add_to_memory(role: str, content: str):
    """Add message to conversation memory."""
    conversation_memory.append({"role": role, "content": content})
    if len(conversation_memory) > 8:  # keep only recent 8 exchanges
        conversation_memory.pop(0)

def clear_memory():
    """Clear conversation memory."""
    global conversation_memory
    conversation_memory = []

# --------------------------
# 5. Advanced RAG function
# --------------------------
def retrieve_context(query: str, top_k: int = 6):
    """Retrieve top-k relevant chunks from Chroma."""
    try:
        # Expand query for insurance-related questions
        insurance_keywords = ["insurance", "worthy", "property", "risk", "coverage", "policy", "assessment", "underwriting"]
        if any(keyword in query.lower() for keyword in insurance_keywords):
            # Use expanded query for better retrieval
            expanded_query = f"{query} insurance property risk assessment coverage deductible building construction hazards"
            query_to_use = expanded_query
        else:
            query_to_use = query
        
        results = collection.query(query_texts=[query_to_use], n_results=top_k)
        if not results["documents"] or not results["documents"][0]:
            # Try with original query if expanded query fails
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

def answer_query(query: str, insurance_type: str = "property_casualty", top_k: int = 6, temperature: float = 0.2):
    """Generate an answer using retrieved context + chat memory."""
    if not query.strip():
        return "Please ask a question about the data."
    
    context = retrieve_context(query, top_k)
    
    # Insurance-type specific context and guidance
    if insurance_type == "life":
        specialized_context = """
        Your primary focus is life insurance underwriting. When responding:
        
        1. For medical information, pay special attention to:
           - Pre-existing conditions and their severity/duration
           - Family history of hereditary diseases
           - Current medications and treatments
           - Lifestyle factors (smoking, alcohol, exercise)
        
        2. When discussing risk factors, consider:
           - Age and life expectancy implications
           - BMI and its impact on mortality risk
           - Medical conditions that may affect longevity
           - Occupation and hazardous activities
        
        3. For policy recommendations, focus on:
           - Standard vs. rated policies based on medical history
           - Term vs. permanent insurance considerations
           - Appropriate coverage amounts based on financial information
           - Riders that may be appropriate (waiver of premium, accelerated benefits)
        """
    else:  # property_casualty
        specialized_context = """
        Your primary focus is property & casualty insurance underwriting. When responding:
        
        1. For property information, pay special attention to:
           - Construction type and building materials
           - Age and condition of the structure
           - Protection features (sprinklers, alarms, etc.)
           - Proximity to fire stations and hydrants
        
        2. When discussing risk factors, consider:
           - Natural hazard exposure (flood zones, wildfire risk, etc.)
           - Occupancy types and their associated hazards
           - Neighboring properties and exposure risks
           - Business operations and liability concerns
        
        3. For policy recommendations, focus on:
           - Appropriate coverage limits based on property values
           - Deductible options for various perils
           - Specialized endorsements for specific risks
           - Risk mitigation measures to reduce premiums
        """
    
    # Common instructions for all insurance types
    common_instructions = """
    Please answer any questions about this document or analysis. Be professional, accurate, and helpful.
    
    When uncertain about specific details, acknowledge the limitations of the information provided.
    Avoid making definitive underwriting decisions, but provide guidance based on industry standards.
    
    Use all available information from the context, even if it seems incomplete. 
    Make reasonable inferences based on the data provided. 
    Format your response clearly with specific details from the data.
    """
    
    # Build system prompt
    system_prompt = (
        f"You are an expert insurance underwriter assistant analyzing {insurance_type.replace('_', ' ').title()} insurance applications. "
        f"Based on the data from insurance documents, provide a comprehensive assessment.\n\n"
        f"{specialized_context}\n\n"
        f"{common_instructions}"
    )

    # Build conversation history
    messages = [{"role": "system", "content": system_prompt}]
    messages += conversation_memory
    messages.append({
        "role": "user",
        "content": f"Context from JSON files:\n\n{context}\n\nUser Question: {query}\n\n"
                   f"Provide a comprehensive answer based on all available information in the context."
    })
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature
        )

        answer = response.choices[0].message.content
        add_to_memory("user", query)
        add_to_memory("assistant", answer)
        return answer
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --------------------------
# 6. Gradio Interface
# --------------------------
def chat_with_bot(message, history):
    """Gradio chat function."""
    response = answer_query(message)
    return response

def reload_database():
    """Reload JSON files from outputs folder."""
    global collection
    try:
        num_files, num_chunks = initialize_database()
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        return f"✅ Reloaded! Found {num_files} JSON files with {num_chunks} chunks."
    except Exception as e:
        return f"❌ Error reloading: {str(e)}"

def create_embeddings():
    """Create embeddings from JSON files in outputs folder."""
    global collection
    try:
        # Clear existing collection if it exists
        try:
            chroma_client.delete_collection(name=collection_name)
        except:
            pass
        
        # Initialize database (load, chunk, and embed)
        num_files, num_chunks = initialize_database()
        
        # Recreate collection reference
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=openai_ef
        )
        
        return f"✅ Embeddings created! Processed {num_files} JSON files, generated {num_chunks} chunks with embeddings."
    except Exception as e:
        return f"❌ Error creating embeddings: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="data Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🧠 data Chatbot
        
        Ask questions about the JSON files in the `outputs` folder. 
        The chatbot uses RAG (Retrieval-Augmented Generation) to answer based on the extracted document data.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_copy_button=True
            )
            msg = gr.Textbox(
                label="Your Question",
                placeholder="Type your question about the data here...",
                lines=2
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary", size="lg")
                clear_btn = gr.Button("Clear", variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Controls")
            insurance_type_dropdown = gr.Dropdown(
                choices=["property_casualty", "life"],
                value="property_casualty",
                label="Insurance Type",
                info="Select the type of insurance for specialized analysis"
            )
            create_embeddings_btn = gr.Button("🔢 Create Embeddings", variant="primary")
            reload_btn = gr.Button("🔄 Reload Database", variant="secondary")
            status = gr.Textbox(
                label="Status",
                value=f"✅ {num_files} JSON files loaded, {num_chunks} chunks indexed",
                interactive=False
            )
            clear_memory_btn = gr.Button("🗑️ Clear Memory", variant="secondary")
    
    # Event handlers
    def user_input(message, history, insurance_type):
        return "", history + [[message, None]], insurance_type
    
    def bot_response(history, insurance_type):
        if history and history[-1][1] is None:
            message = history[-1][0]
            response = answer_query(message, insurance_type=insurance_type)
            history[-1][1] = response
        return history
    
    def clear_chat():
        clear_memory()
        return []
    
    def update_status():
        result = reload_database()
        return result
    
    def update_embeddings_status():
        result = create_embeddings()
        return result
    
    msg.submit(user_input, [msg, chatbot, insurance_type_dropdown], [msg, chatbot, insurance_type_dropdown], queue=False).then(
        bot_response, [chatbot, insurance_type_dropdown], chatbot
    )
    submit_btn.click(user_input, [msg, chatbot, insurance_type_dropdown], [msg, chatbot, insurance_type_dropdown], queue=False).then(
        bot_response, [chatbot, insurance_type_dropdown], chatbot
    )
    clear_btn.click(clear_chat, None, chatbot, queue=False)
    clear_memory_btn.click(clear_chat, None, chatbot, queue=False)
    reload_btn.click(update_status, None, status, queue=False)
    create_embeddings_btn.click(update_embeddings_status, None, status, queue=False)

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 Starting Gradio Chatbot Interface...")
    print("="*50)
    print(f"📂 Reading JSON files from: {OUTPUT_DIR}")
    print(f"💾 ChromaDB store: {CHROMA_STORE_PATH}")
    print(f"🤖 OpenAI Model: gpt-4o-mini")
    print("\n💬 Chatbot ready! The Gradio interface will open in your browser.\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )

