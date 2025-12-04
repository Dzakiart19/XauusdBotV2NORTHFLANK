# Trading Bot Docker Image - Optimized for Koyeb Free Tier (512MB RAM)
# Multi-stage build for minimal image size

# ============================================
# Stage 1: Builder - Install dependencies
# ============================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/build/wheels -r requirements.txt

# ============================================
# Stage 2: Runtime - Minimal production image
# ============================================
FROM python:3.11-slim AS runtime

LABEL maintainer="XAUUSD Trading Bot"
LABEL description="Telegram Trading Bot optimized for Koyeb Free Tier"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && fc-cache -fv

COPY --from=builder /build/wheels /wheels

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels /root/.cache/pip/*

COPY bot/ ./bot/
COPY webapp/ ./webapp/
COPY config.py .
COPY main.py .

RUN mkdir -p data logs charts backups && \
    chmod -R 755 data logs charts backups

# ============================================
# Environment Variables for Koyeb Free Tier
# ============================================

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV FONTCONFIG_PATH=/etc/fonts

ENV FREE_TIER_MODE=true
ENV SELF_PING_ENABLED=true
ENV SELF_PING_INTERVAL=240

ENV MEMORY_WARNING_THRESHOLD_MB=380
ENV MEMORY_CRITICAL_THRESHOLD_MB=450
ENV MEMORY_MONITOR_INTERVAL_SECONDS=60

ENV PORT=8000
EXPOSE 8000

# ============================================
# Health Check for Koyeb - uses PORT env var
# ============================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# ============================================
# Use tini for proper signal handling
# ============================================
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-u", "main.py"]
