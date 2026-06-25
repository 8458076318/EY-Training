FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit httpx
COPY streamlit_app/ ./streamlit_app/
ENV PYTHONPATH=/app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
