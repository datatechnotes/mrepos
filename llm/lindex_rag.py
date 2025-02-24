from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Custom data
custom_data = """
Equipment Specification: Industrial PMC Milling Machine
Model: XZ-Mill Pro 5000
Manufacturer: XYZ Industrial Solutions
Application: High-precision milling, drilling, and cutting of metal and composite materials
Description
The XZ-Mill Pro 5000 is a high-performance CNC milling machine designed for precision 
machining in industrial applications. It features a robust cast-iron frame, high-speed 
spindle, and advanced control system for accurate and efficient material processing. 
The machine is equipped with an automated tool changer and real-time monitoring system, 
ensuring consistent performance and minimal downtime.
Technical Specifications:
    Spindle Power: 15 kW (20 HP)
    Spindle Speed: 100  12,000 RPM
    Worktable Size: 1200mm x 600mm
    Max Load Capacity: 1000 kg
    Tool Changer: 24-tool automatic carousel
    Precision: 0.005mm
    Control System: Siemens SINUMERIK 840D / Fanuc 31i
    Cooling System: Integrated liquid cooling
    Power Requirements: 400V, 50Hz, 3-phase
    Safety Features: Emergency stop, interlock system, overload protection

Caution & Alerts
- Operational Safety: Ensure that only trained personnel operate the 
  machine. Improper use can lead to serious injuries.
- Material Compatibility: The machine is designed for metal and 
  composite materials. Using incompatible materials may cause damage 
  to the spindle or cutting tools.
- Regular Maintenance: Perform routine maintenance, including lubrication 
  and spindle inspection, to prevent malfunctions.
- Emergency Stop Usage: The E-stop button should be used only in critical 
  situations, as frequent use may cause system calibration issues.
- Electrical Safety: Always disconnect the machine from the power supply 
  before performing maintenance to prevent electrical hazards.
"""

# --- Initialize LlamaIndex ---
# Create document
documents = [Document(text=custom_data)]

# Configure embeddings
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create vector store index
index = VectorStoreIndex.from_documents(documents)

print("LlamaIndex vector store created.")

# --- Load FLAN-T5 Model ---
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- Improved Retrieval with Relevance Check ---
def retrieve_relevant_chunk(query):
    """Retrieve context with relevance checking."""
    retriever = index.as_retriever(similarity_top_k=1)
    results = retriever.retrieve(query)
    
    if not results:
        return None, 0.0
    
    most_relevant = results[0]
    return most_relevant.text, most_relevant.score

# --- Enhanced QA System ---
def ask_rag(query, confidence_threshold=0.1):
    """Improved RAG system with relevance detection."""
    context, score = retrieve_relevant_chunk(query)
    
    # Handle low similarity scores
    if score < confidence_threshold:
        return "I don't know. This question seems unrelated to equipment specifications."
    
    # Enhanced prompt engineering
    prompt = f"""You are a technical assistant. Answer the question based only on the following context. 
    If the answer isn't in the context, say "I don't know."

    Context: {context}
    Question: {query}
    Answer:"""
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        num_beams=5,
        early_stopping=True
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process response
    response = response.strip()
    if "don't know" in response.lower():
        return "I don't have enough information to answer that question."
        
    return response

# --- Test Cases ---
questions = [
    "What is the maximum load capacity?",
    "Explain the cooling system requirements",
    "What's the capital of France?",
    "How do I bake a chocolate cake?",
    "What safety features does it have?"
]

for question in questions:
    answer = ask_rag(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
    print("-" * 50)


