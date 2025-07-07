# pip install langchain-community
# pip install faiss-cpu
# pip install pypdf
# pip install groq
# curl -fsSL https://ollama.com/install.sh | sh
# pip install camelot-py[cv]
# pip install -U langchain-ollama

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import camelot
from groq import Groq
import os
import json

# Step 1: Load PDF & extract tables as text
def extract_text_and_tables(pdf_path):
    print(f"Step 1: Extracting text and tables from '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])

    tables = camelot.read_pdf(pdf_path, pages='all')
    table_text = "\n\n".join([t.df.to_string(index=False) for t in tables])

    print(f"✅ Step 1 completed: Loaded {len(pages)} pages and {len(tables)} tables.")
    print(f"Full text length: {len(full_text)} characters")
    print(f"Table text length: {len(table_text)} characters")
    return full_text, table_text

# Step 2: Split text into chunks
def split_text(full_text):
    print("Step 2: Splitting text into smaller chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    print(f"✅ Step 2 completed: Split text into {len(docs)} chunks.")
    return docs

# Step 3: Embed and upload to vector DB, skip duplicate PDF
def upload_to_vector_db(docs, pdf_filename, vector_db_path='faiss_index'):
    print("Step 3: Starting upload to vector DB.")
    print(f"PDF filename: '{pdf_filename}'")
    print(f"Target vector DB path: '{vector_db_path}'")

    print("Initializing embeddings model: 'nomic-embed-text'...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    print("Embeddings model initialized successfully.")

    uploaded_pdfs = []

    # Metadata file to track uploaded PDFs
    metadata_file = os.path.join(vector_db_path, "uploaded_pdfs.json")

    if os.path.exists(vector_db_path):
        print(f"Vector DB already exists at '{vector_db_path}'. Loading existing vector DB...")
        vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        print("Existing vector DB loaded successfully.")

        # Check metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                uploaded_pdfs = json.load(f)
            print(f"Loaded uploaded PDF list: {uploaded_pdfs}")
        else:
            print("No metadata file found. Creating new metadata.")

        if pdf_filename in uploaded_pdfs:
            print(f"⚠️ PDF '{pdf_filename}' is already uploaded. Skipping upload to prevent duplicates.")
        else:
            print(f"Adding {len(docs)} new documents for '{pdf_filename}'...")
            vectordb.add_documents(docs)
            vectordb.save_local(vector_db_path)
            uploaded_pdfs.append(pdf_filename)
            os.makedirs(vector_db_path, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(uploaded_pdfs, f)
            print("✅ New documents added and metadata updated.")
    else:
        print("No existing vector DB found. Creating a new vector DB from documents...")
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(vector_db_path)
        os.makedirs(vector_db_path, exist_ok=True)
        uploaded_pdfs.append(pdf_filename)
        with open(metadata_file, "w") as f:
            json.dump(uploaded_pdfs, f)
        print("✅ New vector DB created, saved, and metadata initialized.")

    print(f"✅ Step 3 completed: Upload process finished for '{pdf_filename}'.")
    return vectordb

# Step 4: Ask question & get answer
def ask_question(vectordb, question, tablestext):
    print("Step 4: Asking question using vector DB and LLM...")
    retriever = vectordb.similarity_search(question)
    print(f"Retrieved {len(retriever)} relevant docs.")

    combined_text = "\n\n".join([doc.page_content for doc in retriever]) + "\n\n" + tablestext

    # Replace with your real API key
    client = Groq(api_key="YOUR_GROQ_API_KEY")
    messages = [
        {"role": "system", "content": "From the content below, answer the user's question."},
        {"role": "user", "content": question + "\n\n" + combined_text}
    ]
    chat = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
    print("✅ Step 4 completed: Got answer from LLM.")
    return chat.choices[0].message.content

# --- MAIN ---
if __name__ == "__main__":
    pdf_file = "researchpaper.pdf"  # Ensure this file exists
    question = "What is this PDF about?"

    print("🚀 Starting PDF processing pipeline...")

    # Step 1
    full_text, table_text = extract_text_and_tables(pdf_file)

    # Step 2
    docs = split_text(full_text)

    # Step 3
    vectordb = upload_to_vector_db(docs, pdf_filename=os.path.basename(pdf_file))

    # Step 4
    answer = ask_question(vectordb, question, table_text)
    print("\n--- ✅ FINAL ANSWER ---")
    print(answer)
