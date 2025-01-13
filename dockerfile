FROM python:3.9.6
WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
