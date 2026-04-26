FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODE_PROVIDER_MODE=mock

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-train.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-train.txt

COPY . .

RUN mkdir -p data logs models outputs

EXPOSE 7860 8000

CMD ["python", "app.py"]
