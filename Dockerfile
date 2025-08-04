FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/logs

EXPOSE 8000

CMD ["uvicorn", "fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
