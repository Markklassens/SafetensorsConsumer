FROM python:3.11-slim

WORKDIR /app

# Install minimal deps
COPY requirements.txt .
RUN pip install --no-cache-dir numpy scipy

# Optional: better embeddings (uncomment to include)
# RUN pip install --no-cache-dir sentence-transformers

COPY . .

# Persistent store volume
RUN mkdir -p /data
VOLUME ["/data"]

ENV SRHN_STORE=/data
ENV SRHN_LLM=mock
ENV PORT=8765

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

CMD ["python3", "main.py", "server", "--host", "0.0.0.0"]
