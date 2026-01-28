FROM python:3.11-slim
LABEL maintainer="AdaoJOAQUIM"
LABEL description="GENESIS V14 - Ultra-Meta Cognitive OS"
LABEL version="14.0.0"
ENV PYTHONUNBUFFERED=1
ENV GENESIS_COMPUTE=sparse
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python -c "import httpx; r=httpx.get('http://localhost:8000/health'); assert r.status_code==200"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
