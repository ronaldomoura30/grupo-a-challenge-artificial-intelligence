import os
import whisper
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # <-- MUDANÇA PRINCIPAL

DATA_PATH = "data"
DB_PATH = "vector_db"


def run_indexing():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY não encontrada. Verifique seu arquivo .env")

    print("Iniciando transcrição dos vídeos...")
    # (O código de transcrição permanece o mesmo)
    video_files = [f for f in os.listdir(DATA_PATH) if f.endswith(".mp4")]
    if video_files:
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

    print("\nCarregando documentos...")
    pdf_loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True,
                                 use_multithreading=True)
    txt_loader = DirectoryLoader(DATA_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True,
                                 use_multithreading=True)
    documents = pdf_loader.load() + txt_loader.load()
    print(f"Total de documentos carregados: {len(documents)}")

    print("\nDividindo documentos em chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Documentos divididos em {len(chunks)} chunks.")

    print("\nGerando embeddings com a API da OpenAI e criando o Vector DB...")
    # <-- MUDANÇA PRINCIPAL: Usando o embedding da OpenAI. É estável e rápido.
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )
    print(f"\nIndexação concluída com o modelo da OpenAI! Banco de dados salvo em '{DB_PATH}'.")


if __name__ == "__main__":
    run_indexing()
