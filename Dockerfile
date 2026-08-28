# Use official lightweight Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (required by Hugging Face Spaces, user id 1000)
RUN useradd -m -u 1000 user
RUN mkdir -p /home/user/.cache/huggingface /home/user/app && chown -R user:user /home/user

# Switch to non-root user
USER user
WORKDIR /home/user/app

# Copy requirements and install
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY --chown=user . /home/user/app

# Expose port 7860 (Hugging Face Spaces default)
EXPOSE 7860

# Start FastAPI server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
