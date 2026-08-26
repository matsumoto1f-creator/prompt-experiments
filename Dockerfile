FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[api]"

ENV PROMPT_EXPERIMENTS_DB=/data/experiments.db \
    PROMPT_EXPERIMENTS_PROVIDER=mock
VOLUME ["/data"]
EXPOSE 8000

# Default to the CLI; override with `docker run ... uvicorn prompt_experiments.api:app --host 0.0.0.0`
ENTRYPOINT ["prompt-exp"]
CMD ["stats", "peeking"]
