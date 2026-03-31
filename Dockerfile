FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp.py .
COPY templates/ templates/
COPY static/ static/

# Copy only the essential prediction data
COPY predictions/proba_matrix.npz predictions/
COPY predictions/papers.json predictions/
COPY predictions/journals.json predictions/
COPY predictions/meta.json predictions/
COPY predictions/community_reviews.json predictions/

# Training dataset for search and ground-truth display
COPY labeled_dataset_slim.json labeled_dataset.json

EXPOSE 8080

CMD ["gunicorn", "webapp:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120", "--preload"]
