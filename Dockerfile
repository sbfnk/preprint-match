FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp.py .
COPY templates/ templates/
COPY static/ static/

# Copy only the essential prediction data. papers_slim.json (abstracts
# stripped) + abstracts.db keep the working set off the Python heap.
COPY predictions/proba_matrix.npz predictions/
COPY predictions/papers_slim.json predictions/papers_slim.json
COPY predictions/abstracts.db predictions/
COPY predictions/journals.json predictions/
COPY predictions/meta.json predictions/
COPY predictions/community_reviews.json predictions/

# Training dataset for search and ground-truth display
COPY labeled_dataset_slim.json labeled_dataset.json

EXPOSE 8080

# No --preload: with a single worker, preloading makes the master hold the
# data and CPython refcount churn copies it into the worker (copy-on-write
# defeat), roughly doubling RSS. Letting the lone worker load it once keeps
# memory ~1x.
CMD ["gunicorn", "webapp:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "120"]
