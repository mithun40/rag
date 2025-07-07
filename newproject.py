# pip install langchain-community
# pip install faiss-cpu
#pip install pypdf
# pip install groq
#curl -fsSL https://ollama.com/install.sh | sh
# pip install camelot-py[cv]
#pip install -U langchain-ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import camelot
from groq import Groq
import os

# Step 1: Load PDF & extract tables as text
def extract_text_and_tables(pdf_path):
    # Load text
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])

    # Load tables
    tables = camelot.read_pdf(pdf_path, pages='all')
    table_text = "\n\n".join([t.df.to_string(index=False) for t in tables])

    print(f"Loaded {len(pages)} pages and {len(tables)} tables.")
    print("Step 1 completed: Extracted text and tables from PDF.")
    print(f"Full text length: {len(full_text)} characters")
    print(f"Table text length: {len(table_text)} characters")
    return full_text, table_text

# Step 2: Split text into chunks
def split_text(full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    print(f"Step 2 completed: Split text into {len(docs)} chunks.")
    return docs

# Step 3: Embed and upload to vector DB
def upload_to_vector_db(docs, vector_db_path='faiss_index'):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(vector_db_path):
        vectordb = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        vectordb.add_documents(docs)
        vectordb.save_local(vector_db_path)
        print("Added new docs to existing vector DB.")
    else:
        vectordb = FAISS.from_documents(docs, embeddings)
        vectordb.save_local(vector_db_path)
        print("Created new vector DB and saved.")
    print(f"Step 3 completed: Uploaded {len(docs)} docs to vector DB at '{vector_db_path}'.")
    return vectordb

# Step 4: Ask question & get answer
def ask_question(vectordb, question, tablestext):
    retriever = vectordb.similarity_search(question)
    print(f"Retrieved {len(retriever)} relevant docs.")

    # Combine retrieved text + tables
    combined_text = "\n\n".join([doc.page_content for doc in retriever]) + "\n\n" + tablestext

    # Call Groq LLM to get answer
    client = Groq(api_key="GROQAPIKEY")  # replace with your real key
    messages = [
        {"role": "system", "content": "From the content below, answer the user's question."},
        {"role": "user", "content": question + "\n\n" + combined_text}
    ]
    chat = client.chat.completions.create(messages=messages, model="mixtral-8x7b-32768")
    print("Step 4 completed: Got answer from LLM.")
    return chat.choices[0].message.content

# --- MAIN ---
if __name__ == "__main__":
    pdf_file = "researchpaper.pdf"  # Make sure this file exists in your current directory
    question = "What is this PDF about?"

    print("Starting PDF processing pipeline...")

    # Step 1
    full_text, table_text = extract_text_and_tables(pdf_file)

    # Step 2
    docs = split_text(full_text)

    # Step 3
    vectordb = upload_to_vector_db(docs)

    # Step 4
    answer = ask_question(vectordb, question, table_text)
    print("\n--- ANSWER ---")
    print(answer)
