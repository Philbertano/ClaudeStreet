FROM python:3.12-slim

WORKDIR /app

# Install system deps for numpy/pandas
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt anthropic

COPY src/ ./

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import claudestreet; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "claudestreet.evolve_engine.engine"]
