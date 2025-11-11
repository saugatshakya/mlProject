FROM python:3.11-slim

# Keep Python output unbuffered and avoid .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install system dependencies needed to build some Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list and install first (cacheable layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY . /app

# Ensure uploads and models directories exist (used by the app)
RUN mkdir -p /app/uploads /app/models \
    && chown -R root:root /app

EXPOSE 5001

# Default command - the project's run.sh executes `python app.py`.
# Use the same entry point here for simplicity. For production consider
# using gunicorn (add it to requirements and replace CMD accordingly).
CMD ["python", "app.py"]
