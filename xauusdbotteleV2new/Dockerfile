# Trading Bot Docker Image for Koyeb Deployment
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including fonts for chart rendering
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    fonts-dejavu-core \
    fonts-liberation \
    fontconfig \
    && rm -rf /var/lib/apt/lists/* \
    && fc-cache -fv

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs charts backups

# Set environment variables for Koyeb deployment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV MPLBACKEND=Agg
ENV FONTCONFIG_PATH=/etc/fonts

# Expose health check port (Koyeb will set PORT env var)
EXPOSE 8080

# Run the bot
CMD ["python", "main.py"]
