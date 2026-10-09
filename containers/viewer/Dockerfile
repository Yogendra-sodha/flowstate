ARG PYTHON_BASE=python:3.12-slim
FROM ${PYTHON_BASE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    FLOWSTATE_VIEWER_HTML=/snapshot/index.html \
    PORT=8080

WORKDIR /app
COPY server.py /app/server.py
USER 65532:65532
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/healthz', timeout=3).read()"
ENTRYPOINT ["python", "/app/server.py"]
