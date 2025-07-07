# pip install langchain-community
# pip install faiss-cpu
# pip install pypdf
# pip install groq
# pip install camelot-py[cv]
# pip install -U langchain-ollama
# pip install python-docx
# pip install unstructured
# pip install unstructured[docx]

from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import camelot
from groq import Groq
import os
import json

# Step 1: Load file based on extension
def extract_text_and_tables(file_path):
    print(f"Step 1: Extracting content from '{file_path}'...")
    ext = os.path.splitext(file_path)[-1].lower()

    table_text = ""  # default

    if ext == ".pdf":
        print("Detected: PDF")
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([p.page_content for p in pages])
        print(f"Loaded {len(pages)} pages.")

        # Extract tables
        tables = camelot.read_pdf(file_path, pages='all')
        table_text = "\n\n".join([t.df.to_string(index=False) for t in tables])
        print(f"Extracted {len(tables)} tables from PDF.")

    elif ext == ".docx":
        print("Detected: DOCX")
        loader = UnstructuredWordDocumentLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

    elif ext == ".txt":
        print("Detected: TXT")
        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

    elif ext == ".md":
        print("Detected: Markdown")
        loader = UnstructuredMarkdownLoader(file_path)
        docs = loader.load()
        full_text = "\n".join([d.page_content for d in docs])

    else:
        raise ValueError(f"❌ Unsupported file extension: {ext}")

    print(f"✅ Step 1 completed. Text length: {len(full_text)} chars | Tables length: {len(table_text)} chars")
    return full_text, table_text

# Step 2: Split text into chunks
def split_text(full_text):
    print("Step 2: Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    print(f"✅ Step 2 completed: {len(docs)} chunks created.")
    return docs

# Step 3: Embed & upload to vector DB, skip duplicates
def upload_to_vector_db(docs, filename, vector_db_path='faiss_index'):
    print(f"Step 3: Uploading '{filename}' to vector DB...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    uploaded_files = []
    metadata_file = os.path.join(vector_db_path, "uploaded_files.json")

    if os.path.exists(vector_db_path):
        vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        print("Loaded existing vector DB.")

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                uploaded_files = json.load(f)
            print(f"Uploaded files so far: {uploaded_files}")

        if filename in uploaded_files:
            print(f"⚠️ '{filename}' already uploaded. Skipping to avoid duplication.")
        else:
            vectordb.add_documents(docs)
            vectordb.save_local(vector_db_path)
            uploaded_files.append(filename)
            os.makedirs(vector_db_path, exist_ok=True)
            with open(metadata_file, "w") as f:
                json.dump(uploaded_files, f)
            print("✅ New docs added & metadata updated.")

    else:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(vector_db_path)
        os.makedirs(vector_db_path, exist_ok=True)
        uploaded_files.append(filename)
        with open(metadata_file, "w") as f:
            json.dump(uploaded_files, f)
        print("✅ Created new vector DB & saved metadata.")

    print(f"✅ Step 3 finished for '{filename}'.")
    return vectordb

# Step 4: Ask question & get answer
def ask_question(vectordb, question, tablestext):
    print("Step 4: Asking question...")
    retriever = vectordb.similarity_search(question)
    print(f"Retrieved {len(retriever)} docs from vector DB.")

    combined_text = "\n\n".join([doc.page_content for doc in retriever]) + "\n\n" + tablestext

    # Replace with your real key
    client = Groq(api_key="YOUR_GROQ_API_KEY")
    messages = [
        {"role": "system", "content": "Use this content to answer."},
        {"role": "user", "content": question + "\n\n" + combined_text}
    ]
    chat = client.chat.completions.create(messages=messages, model="llama-3.3-70b-versatile")
    print("✅ Got answer from LLM.")
    return chat.choices[0].message.content

# --- MAIN ---
if __name__ == "__main__":
    file_path = "researchpaper.pdf"   # try with "document.docx", "notes.txt", etc.
    question = "What is this document about?"

    print("🚀 Starting multi-format document processing...")

    # Step 1
    full_text, table_text = extract_text_and_tables(file_path)

    # Step 2
    docs = split_text(full_text)

    # Step 3
    vectordb = upload_to_vector_db(docs, filename=os.path.basename(file_path))

    # Step 4
    answer = ask_question(vectordb, question, table_text)
    print("\n--- ✅ FINAL ANSWER ---")
    print(answer)
