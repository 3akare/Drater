FROM tensorflow/tensorflow:2.16.2

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

EXPOSE 50051

CMD ["python", "-u", "src/lstm.py"]
