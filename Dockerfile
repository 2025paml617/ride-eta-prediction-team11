FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/best_model.pkl

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY deploy ./deploy
COPY best_model.pkl ./best_model.pkl

EXPOSE 8000
CMD ["uvicorn", "deploy.api:app", "--host", "0.0.0.0", "--port", "8000"]
