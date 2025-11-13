# CityRent Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

# Por defecto, producción con Gunicorn (Render).
CMD sh -c 'if [ "$DEV" = "1" ]; then \
              python app/app.py; \
            else \
              gunicorn app.app:server --workers=1 --threads=4 --timeout=180 --bind 0.0.0.0:${PORT:-8050}; \
            fi'
