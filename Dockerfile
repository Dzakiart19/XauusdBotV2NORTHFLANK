# Trading Bot Docker Image - Dioptimalkan untuk Koyeb Free Tier
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies dengan cleanup untuk hemat memory
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv \
    && apt-get clean

# Copy requirements first untuk caching layer
COPY requirements.txt .

# Install Python dependencies dengan optimasi memory
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip/*

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs charts backups

# Set environment variables untuk Koyeb deployment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV FONTCONFIG_PATH=/etc/fonts

# Koyeb-specific environment variables
ENV FREE_TIER_MODE=true
ENV SELF_PING_ENABLED=true
ENV SELF_PING_INTERVAL=240

# Default port (Koyeb akan set PORT env var)
ENV PORT=8080
EXPOSE 8080

# Healthcheck untuk Koyeb - cek setiap 30 detik
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the bot dengan unbuffered output
CMD ["python", "-u", "main.py"]
