FROM python:3.9.12-slim

RUN mkdir /app
COPY ./app

WORKDIR /app

EXPOSE 8501

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run"]
CMD streamlit run app.py --server.port 8501
