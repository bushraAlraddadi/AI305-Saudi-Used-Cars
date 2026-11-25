FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what the API needs at runtime
COPY src/common.py src/common.py
COPY app/ app/
COPY models/best_* models/

EXPOSE 5000
CMD ["python", "app/app.py"]
