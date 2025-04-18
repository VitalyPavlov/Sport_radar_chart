FROM python:3.11-slim

WORKDIR .

COPY setup.py .
RUN pip install --no-cache-dir -e .

COPY .env .
COPY ./utils ./utils
COPY ./app ./app
RUN chmod +x ./app/entrypoint.sh

ENTRYPOINT ["./app/entrypoint.sh"]
CMD ["streamlit", "run", "./app/app.py"]