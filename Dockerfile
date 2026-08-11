# pmbot has no third-party dependencies, so the image is just Python + the app.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PMBOT_DATA_DIR=/data

WORKDIR /app

COPY pyproject.toml README.md ./
COPY pmbot ./pmbot
COPY data ./data
COPY scripts ./scripts

RUN pip install --no-cache-dir . && mkdir -p /data

# Runs as root on purpose. Platform volumes (Railway included) mount owned by
# root, and a non-root container then cannot write its own journal -- which
# fails at the first bet rather than at deploy time, where you'd notice.
# If you mount a volume you control the permissions of, add a USER line.

EXPOSE 8080

HEALTHCHECK --interval=60s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8080)}/healthz',timeout=4)"

# $PORT is injected by the platform; the shell form expands it.
CMD ["sh", "-c", "pmbot serve --host 0.0.0.0 --port ${PORT:-8080}"]
