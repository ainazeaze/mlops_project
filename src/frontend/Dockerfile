FROM python:3.11-slim

RUN mkdir /frontend
WORKDIR /frontend
RUN pip install streamlit requests pandas
COPY ./* /frontend
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
