FROM python:3.11-slim

WORKDIR /app

# System deps for python-whois and SSL
RUN apt-get update && apt-get install -y --no-install-recommends \
    whois \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Expose port (HF Spaces uses 7860 by default)
EXPOSE 7860

# Start server on port 7860 for HF Spaces compatibility
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
