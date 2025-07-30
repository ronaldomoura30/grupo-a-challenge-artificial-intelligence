# Usamos uma imagem base oficial do Python. A versão slim é mais leve.
FROM python:3.10-slim

# Instala o ffmpeg, que é uma dependência do sistema para o openai-whisper processar áudio/vídeo.
RUN apt-get update && apt-get install -y ffmpeg

# Define o diretório de trabalho dentro do container.
WORKDIR /app

# Copia o arquivo de requisitos primeiro para aproveitar o cache do Docker.
COPY requirements.txt .

# Instala as dependências Python.
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o resto do código do projeto para o diretório de trabalho.
COPY . .

# Expõe a porta que o Gradio usará.
EXPOSE 7860

# O comando padrão para executar quando o container iniciar.
# Será sobrescrito pelo docker-compose para a tarefa de indexação.
CMD ["python", "app_gradio.py"]