FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY webapp.py .
COPY templates/ templates/

# Copy only the essential prediction data
COPY predictions/proba_matrix.npz predictions/
COPY predictions/papers_slim.json predictions/papers.json
COPY predictions/journals.json predictions/
COPY predictions/meta.json predictions/

EXPOSE 8080

CMD ["gunicorn", "webapp:app", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "2", "--timeout", "120"]
