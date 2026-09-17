FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/ ./data/

# Ollama itself runs outside this container (on the host, or a separate service) --
# set OLLAMA_HOST to reach it, e.g. http://host.docker.internal:11434
ENV OLLAMA_HOST=http://host.docker.internal:11434

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
