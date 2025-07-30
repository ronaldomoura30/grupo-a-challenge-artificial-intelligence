# indexador_huggingFace.py

import os
import whisper
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

DATA_PATH = "data"
DB_PATH = "vector_db"


def run_indexing():
    # --- 1. Transcrição dos Vídeos (permanece igual) ---
    print("Iniciando transcrição dos vídeos...")
    video_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".mp4")]

    if not video_files:
        print("Nenhum vídeo (.mp4) encontrado na pasta 'data'.")
    else:
        whisper_model = whisper.load_model("base")
        for video_file in video_files:
            video_path = os.path.join(DATA_PATH, video_file)
            transcript_path = os.path.join(DATA_PATH, f"{os.path.splitext(video_file)[0]}.txt")

            if not os.path.exists(transcript_path):
                print(f"Transcrevendo {video_file}...")
                result = whisper_model.transcribe(video_path, fp16=False)
                with open(transcript_path, "w", encoding="utf-8") as f:
                    f.write(result["text"])
                print(f"Transcrição de {video_file} salva.")
            else:
                print(f"Transcrição para {video_file} já existe.")

    # --- 2. Carregamento dos Documentos (LÓGICA CORRIGIDA) ---
    print("\nCarregando documentos de texto e PDF...")

    # Loader para PDFs
    pdf_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )

    # Loader para TXTs (incluindo as transcrições)
    txt_loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )

    print("Carregando PDFs...")
    pdf_documents = pdf_loader.load()
    print("Carregando arquivos de texto...")
    txt_documents = txt_loader.load()

    # Juntando as duas listas de documentos
    documents = pdf_documents + txt_documents

    if not documents:
        print("Nenhum documento .txt ou .pdf encontrado. A indexação será encerrada.")
        return

    print(f"Total de documentos carregados: {len(documents)}")

    # --- 3. Divisão dos Documentos em Chunks (permanece igual) ---
    print("\nDividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    # --- 4. Geração de Embeddings e Armazenamento (permanece igual) ---
    print("\nGerando embeddings e criando o Vector DB...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"\nIndexação concluída! Banco de dados salvo em '{DB_PATH}'.")


if __name__ == "__main__":
    run_indexing()