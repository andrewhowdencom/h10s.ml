# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We also install build-essential for some potential compilation needs
RUN apt-get update && apt-get install -y build-essential && \
    pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set python path so src modules can be imported
ENV PYTHONPATH=/app

# Default command (can be overridden)
CMD ["python", "src/train.py"]
