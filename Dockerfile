FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY api.py .
COPY predict.py .

CMD ["python", "api.py"]