FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt -e .

COPY . .

EXPOSE 8000 8501

CMD ["uvicorn", "riskops.api:app", "--host", "0.0.0.0", "--port", "8000"]
