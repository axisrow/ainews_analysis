FROM python:3.11-slim as builder

# Install system dependencies for NLP libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet')"

# Production stage
FROM python:3.11-slim

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app
WORKDIR /home/app

# Create necessary directories
RUN mkdir -p data logs reports

# Copy application code
COPY --chown=app:app . .

# Copy database file if it exists
COPY --chown=app:app data/ai_news.db data/ai_news.db

# Set Python path
ENV PYTHONPATH="/home/app:/home/app/src"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose port for Streamlit dashboard
EXPOSE 8501

# Default command - run dashboard service
CMD ["python", "main.py", "--dashboard"]