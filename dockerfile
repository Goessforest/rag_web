
# Start from a minimal Python image
FROM python:3.9-slim

# Set a working directory in the container
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY . .

# Expose the Django dev server port
EXPOSE 8000

# By default, run the Django dev server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
