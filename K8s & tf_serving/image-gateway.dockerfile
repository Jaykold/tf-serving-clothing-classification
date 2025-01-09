FROM python:3.10.16-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY ["gateway.py", "proto.py", "./"]

EXPOSE 8001

ENTRYPOINT [ "uvicorn", "gateway:app", "--host", "0.0.0.0", "--port", "8001" ]