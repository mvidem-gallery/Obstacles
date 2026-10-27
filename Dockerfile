FROM ubuntu:latest
LABEL authors="Дембіцький Микола"

FROM python:3.11-slim

WORKDIR /app

# системні залежності (приклад для OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# коду тут немає, він буде змонтований
CMD ["python", "main.py"]

ENTRYPOINT ["top", "-b"]