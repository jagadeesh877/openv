# ─────────────────────────────────────────────────────────────────
# OpenEnv AI Environment — Dockerfile
# Hugging Face Spaces (Docker SDK) compatible
# Exposes FastAPI on port 7860
# ─────────────────────────────────────────────────────────────────

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies first (layer cache optimisation)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create __init__ files (ensure directories exist if needed, 
# but they are already in the source)

# Expose the Hugging Face Spaces port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Start FastAPI server on port 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--loop", "uvloop", "--http", "httptools", "--log-level", "info", "--timeout-keep-alive", "60"]
