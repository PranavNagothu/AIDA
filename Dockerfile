# Fixed Dockerfile - works with your existing web_app.py
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy ALL files (this ensures both .py files are copied)
COPY . .

# Ensure files have correct permissions
RUN chmod +r *.py

# Debug: Show what files we have
RUN echo "Files in /app:" && ls -la /app/

# Create necessary directories
RUN mkdir -p /app/temp /app/analysis_output

# Expose port
EXPOSE 7860

# Run Streamlit with explicit file path
CMD ["python", "-m", "streamlit", "run", "/app/web_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]