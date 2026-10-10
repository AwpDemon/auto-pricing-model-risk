# Reproducible environment for the full pipeline and batch scoring.
#   docker build -f deploy/Dockerfile -t auto-pricing .
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" auto-pricing run
FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml README.md ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY config ./config
COPY models ./models
COPY reports/metrics.json ./reports/metrics.json
COPY data/sample ./data/sample
RUN pip install --no-cache-dir -e . --no-deps
ENTRYPOINT ["python", "-m", "autopricing"]
CMD ["run", "--sample"]
