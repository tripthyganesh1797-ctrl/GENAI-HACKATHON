FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; rank_bm25/rapidfuzz/scikit-learn are pure
# Python or ship wheels, so no compiler toolchain is required.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Container runs fully offline-capable out of the box (no LLM_API_KEY
# required -- see offline_fallback.py). Set LLM_PROVIDER/LLM_API_KEY at
# `docker run` time (-e) to enable the higher-quality LLM path instead.
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
