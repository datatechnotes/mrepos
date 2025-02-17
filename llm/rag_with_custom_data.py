# Install Required Libraries
# pip install langchain transformers faiss-cpu sentence-transformers

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# Custom Equipment Specifications (Knowledge Base)
custom_documents = [
    "The engine starts at 6 AM and shuts down at 10 PM.",
    "High-pressure alert triggers when the pressure exceeds 250 PSI.",
    "The cooling fan activates when the temperature reaches 80 degree C.",
    "Battery recharge initiates if the voltage drops below 11.5V.",
    "The emergency alarm sounds if vibration levels exceed 5.0 mm/s.",
    "Oil replacement is required every 1,000 operating hours.",
    "System enters standby mode after 15 minutes of inactivity.",
    "Fuel consumption rate should not exceed 3.5L per hour.",
    "Air filter must be cleaned when airflow drops below 70%.",
    "The backup generator activates if the main power is lost for more than 5 seconds."
]

# Convert text into vector embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store document embeddings in FAISS (a fast similarity search library)
vector_store = FAISS.from_texts(custom_documents, embeddings)

# Define a retriever to find relevant documents (top-1 match)
retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# Load Flan-T5 Model and Tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Define a text generation pipeline
text_generation_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)

# Integrate LLM into LangChain
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Create a Retrieval-Augmented Generation (RAG) pipeline
rag_pipeline = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)


# Function to ask a question and retrieve answers
def ask(query):
    result = rag_pipeline({"query": query})
    print(f"Question: {query}")
    print(f"Answer: {result['result']}")
    print("Sources:")
    for doc in result['source_documents']:
        print(f"- {doc.page_content}")
    print("\n")



# Example usage
ask("When does the engine start?")
ask("What triggers the high-pressure alert?")
ask("When should we clean the air filter?")
ask("What is the weather today?")
ask("What is the capital of France?")
