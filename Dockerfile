FROM python:3.10

WORKDIR /app
COPY . /app
EXPOSE 8501
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD streamlit run server.py
