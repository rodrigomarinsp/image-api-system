# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install PostgreSQL client tools for initialization script
RUN apt-get update && apt-get install -y postgresql-client curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory for initialization scripts
RUN mkdir -p /app/scripts

# Create startup script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Copy application files
COPY . .

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Use entrypoint script to initialize and run the application
ENTRYPOINT ["/app/entrypoint.sh"]