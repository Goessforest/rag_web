# Start from official Python image
FROM python:3.9-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create directory for the Django app
WORKDIR /code

# Install any system dependencies you need, e.g. libpq-dev for Postgres
RUN apt-get update \
    && apt-get install -y gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt /code/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . /code/

# Expose port 8000 for Django (if you plan to run manage.py runserver or gunicorn on port 8000)
EXPOSE 8000

# By default, run Django server (can be overridden by docker-compose or explicitly in the Dockerfile)
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
