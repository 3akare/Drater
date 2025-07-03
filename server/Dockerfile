FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --timeout=100 --retries=10 -r requirements.txt

COPY . .

EXPOSE 8080 

CMD ["python", "-u", "src/server.py"]
