FROM python:3.11-slim

ENV SOCCERDATA_LOGLEVEL="ERROR"
ENV SOCCERDATA_DIR="/soccerdata"

WORKDIR .

COPY setup.py .
RUN pip install --no-cache-dir -e .

COPY .env .
COPY ./utils ./utils
COPY ./db ./db
RUN chmod +x ./db/entrypoint.sh


ENTRYPOINT ["./db/entrypoint.sh"]
CMD ["python", "./db/create_db.py"]
