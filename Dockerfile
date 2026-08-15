FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
        pip install --no-cache-dir --default-timeout=1000 --retries 5 -r requirements.txt

COPY . .

ENV OLLAMA_HOST="http://host.docker.internal:11434"

CMD ["python", "main.py"]