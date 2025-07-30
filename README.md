# Desafio Grupo A - Artificial Intelligence 🚀

## <b> Pontos importantes sobre o projeto: </b> 

### 1 - Essa solução foi inicialmente desenvolvida com uma IDE onde seja possível realizar os testes e visualizar as respostas. Esse ambiente é somente para desenvolvimento utilizando o framework Gradio. 

### 2 - Esta tudo em container para que seja mais facil de se manter e se comportar com o ambiente onde se desenvolve e seja possível ser aptado para qualquer nuvem ou ambiente on-premise. 

### 3 - No projeto temos três llms e embaddings diferentes. 
#### 3.1 - Primeira solução foi utilizado llm e embeddings do Hugging Face por questão de custo, mas o desempenho não foi satisfatório.
#### 3.2 - Depois foi utilizado gemini do Google que esta apresentando erros de embeddings. Por questão de tempo não dei continuidade
#### 3.3 - Por ultimo foi utilizado o ChatGPT do OpenAI e as repostas foram feitas com embeddings do OpenAI. Dando tudo certo o desempenho aceitável.

### <b> Muitas melhorias e diversas soluções podem ser definidas. </b> </br>
#### 1 - Desenvolvimento de uma API com FastAPI para vetorizar todos os arquivos de texto (PDF e TXT) e indexar no banco de dados.</br>
#### 2 - Integrar interação com ElevenLabs para respostas em audio</br>
#### 3 - Utilizar o Crewai para reliazar pesquisas externas e apoio nas respostas. </br>
#### 4 - Desenvolvimento em Next.js para o ambiente de AI, funcional e multiplataforma. </br>
#### 5 - Melhorar o prompt e criação de contextos para a inteligência artificial, separação dos contextos por pastas</br>
---

## 🧭 Sumario

- [✅ Pre-requisitos](#pre-requisitos)
- [⚙️ Instalacao](#instalacao)
- [🐳 Comandos Docker](#comandos-docker)
- [🛠️ Dicas e Troubleshooting](#dicas-e-troubleshooting)

---

## ✅ Pre-requisitos
- Docker e Docker Compose instalados
- Python 3.11+
- Acesso às variáveis de ambiente (.env) configuradas
- Pip instalado

---

## ⚙️ Instalacao

Clone o repositório:

```bash
   git clone https://github.com/ronaldomoura30/grupo-a-challenge-artificial-intelligence
```

## 🐳 Comandos Docker
<p style="color:royalblue"> <b>
    Toda aplicação esta rodando em cima do docker. Que é um sistema de gerenciamento de conteineres.
</b> </p>

<b> Subir e buildar as imagens do docker compose e com o parametro -d para rodar em background </b>

```bash
   docker compose run --build --rm ia-tutor-app python indexador.py
```

<b> Subir o docker compose com o parametro -d para rodar em background </b>

```bash
   docker-compose up -d
```

<b> Forçar um rebuild completo sem cache das imagens do docker compose </b>

```bash
   docker compose down
   docker compose build --no-cache
   docker compose up -d
```

<b> Descer e limpar os volumes do docker compose, só use se precisar vai deletar o banco. </b>

```bash
   docker-compose restart
```

<b> Descer e limpar os volumes do docker compose, só use se precisar vai deletar o banco. </b>

```bash
   docker-compose down -v --remove-orphans
```

## 🛠️ Dicas e Troubleshooting
🐘 Precisa acessar o banco diretamente?
Use ferramentas como PgAdmin ou psql com as credenciais definidas no .env.

🔐 Variáveis de ambiente não reconhecidas?
Confirme se o .env foi copiado corretamente e está na raiz do projeto.
