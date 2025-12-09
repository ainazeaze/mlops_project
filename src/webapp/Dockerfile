FROM python:3.11-slim

RUN mkdir /webapp
RUN mkdir /model
WORKDIR /webapp
RUN pip install mlflow fastapi uvicorn loguru pymongo
COPY ./* /webapp
COPY ./model/* /model
RUN pip install -r /model/requirements.txt
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
