import os
import gradio as gr
from dotenv import load_dotenv
from typing import List, Any
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# --- VERSÃO FINAL E ROBUSTA DO ADAPTADOR ---
class FinalGoogleGenerativeAIEmbeddings(GoogleGenerativeAIEmbeddings):
    def embed_documents(self, texts: List[str], **kwargs: Any) -> List[List[float]]:
        # Repassa quaisquer argumentos extras para a função original
        return super().embed_documents(texts, **kwargs)

    def embed_query(self, text: str, **kwargs: Any) -> List[float]:
        # Repassa quaisquer argumentos extras e corrige o formato da saída
        embedding = super().embed_query(text, **kwargs)
        return embedding[0] if isinstance(embedding, list) and len(embedding) == 1 and isinstance(embedding[0], list) else embedding
# ----------------------------------------------------

# --- 1. Configuração Inicial ---
print("Iniciando a configuração do Tutor IA (versão final)...")
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY não encontrada. Verifique seu arquivo .env")

db_path = "vector_db"
if not os.path.exists(db_path):
    raise FileNotFoundError(f"Diretório do Vector DB '{db_path}' não encontrado.")

embedding_model = FinalGoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})
print("Retriever carregado com sucesso.")

# --- 2. Prompt de Sistema ---
system_template = """
Você é um Tutor de IA da +A Educação, especialista nos fundamentos de programação contidos no material fornecido. Sua personalidade é a de um professor paciente, objetivo e encorajador.

SIGA ESTAS REGRAS ESTRITAMENTE:
1. SEMPRE baseie sua resposta no CONTEXTO fornecido. Seu conhecimento é LIMITADO a este contexto.
2. SE O CONTEXTO for vazio ou não contiver a resposta para a pergunta do usuário, sua única resposta deve ser: "Peço desculpas, mas não encontrei informações sobre este tópico específico no meu material de estudo. Você poderia tentar perguntar de outra forma ou sobre outro assunto relacionado a fundamentos de programação?" NUNCA, JAMAIS, invente uma resposta ou use conhecimento externo.
3. Se a pergunta do usuário for vaga (ex: "me ensine a programar"), faça UMA pergunta para esclarecer (ex: "Claro! Qual tópico dos fundamentos de programação te interessa mais agora?").
4. Após o diagnóstico (ou se a pergunta já for específica), vá direto para a explicação, usando o formato que o usuário preferir, se ele indicar. Se não, use um texto claro com exemplos de código do contexto.
5. Seja Conciso: Forneça respostas diretas e curtas. Evite parágrafos longos.

Contexto do material de estudo recuperado:
---
{context}
---
"""
prompt = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("human", "{question}"),
])

# --- 3. LLM e Cadeia RAG ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.2)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("Cadeia RAG construída. O tutor está pronto.")

# --- 4. Função para o Gradio ---
def chat_response(message, history):
    return rag_chain.invoke(message)

# --- 5. Interface Gradio ---
print("Iniciando a interface Gradio...")
iface = gr.ChatInterface(
    fn=chat_response,
    title="Tutor de IA - Fundamentos de Programação",
    description="Faça perguntas sobre o conteúdo de programação e receba ajuda personalizada.",
    theme="soft",
    examples=[
        ["O que são laços de repetição?"],
        ["Qual a diferença entre 'while' e 'for'?"],
        ["Me explique o que é uma variável."]
    ],
    chatbot=gr.Chatbot(height=500, show_label=False, type='messages')
).launch(server_name="0.0.0.0", server_port=7860)