FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements (if any) and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
