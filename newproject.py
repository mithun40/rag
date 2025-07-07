# pip install langchain-community
# pip install faiss-cpu
# pip install groq
#curl -fsSL https://ollama.com/install.sh | sh

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from groq import Groq

# Step 1: Load PDF & (try to) extract tables
def extract_text_and_tables(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join([p.page_content for p in pages])

    try:
        import camelot
        tables = camelot.read_pdf(pdf_path, pages='all')
        table_text = "\n\n".join([t.df.to_string(index=False) for t in tables])
        print(f"Loaded {len(pages)} pages and {len(tables)} tables.")
    except Exception as e:
        print(f"Could not extract tables: {e}")
        table_text = ""
        print(f"Loaded {len(pages)} pages, skipped tables.")

    return full_text, table_text

# Step 2: Split text
def split_text(full_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([full_text])
    return docs

# Step 3: Embed and upload to FAISS
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
    return vectordb

# Step 4: Ask question
def ask_question(vectordb, question, tablestext):
    retriever = vectordb.similarity_search(question)
    print(f"Retrieved {len(retriever)} relevant docs.")
    combined_text = "\n\n".join([doc.page_content for doc in retriever]) + "\n\n" + tablestext
    client = Groq(api_key="YOUR_GROQ_API_KEY")  # replace with your real key
    messages = [
        {"role": "system", "content": "From the content below, answer the user's question."},
        {"role": "user", "content": question + "\n\n" + combined_text}
    ]
    chat = client.chat.completions.create(messages=messages, model="mixtral-8x7b-32768")
    return chat.choices[0].message.content

# --- MAIN ---
if __name__ == "__main__":
    pdf_file = r"C:\Users\44754\Downloads\NewProject\researchpaper.pdf"
    question = "What is this PDF about?"

    full_text, table_text = extract_text_and_tables(pdf_file)
    docs = split_text(full_text)
    vectordb = upload_to_vector_db(docs)
    answer = ask_question(vectordb, question, table_text)
    print("\n--- ANSWER ---")
    print(answer)
