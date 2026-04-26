FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODE_PROVIDER_MODE=offline

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Slim image: Gradio + API + OpenEnv (no PyTorch). Builds in minutes.
# For GPU + local HF inside Docker: docker build -f Dockerfile.train -t forge:train .
COPY requirements-core.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements-core.txt

COPY . .

RUN mkdir -p data logs models outputs

EXPOSE 7860 8000

CMD ["python", "app.py"]
