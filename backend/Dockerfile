FROM python:3.12
WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY data .
COPY model .
COPY models .
COPY src .
COPY ufc_fast_api.py .

CMD ["python", "-m", "uvicorn", "ufc_fast_api:app", "--host","0.0.0.0", "--port", "8000"]