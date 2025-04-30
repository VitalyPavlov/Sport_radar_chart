import json
import pandas as pd
from datetime import datetime
from typing import Any
from sqlalchemy import create_engine, URL
from dotenv import dotenv_values


def get_connection(test_mode=False):
    """ Return sqlalchemy engine for DB connection """
    
    config = dotenv_values(".env")

    sql_url = URL.create(
            "postgresql",
            host=config["DATABASE_HOST"],
            port=config["DATABASE_PORT"],
            database=config["DATABASE_NAME"] if not test_mode else config["DATABASE_NAME_TEST"],
            username=config["DATABASE_USER"],
            password=config["DATABASE_PASSWORD"],
        )
    
    try:
        create_engine(sql_url).connect()
    except:
        sql_url = URL.create(
            "postgresql",
            host="localhost",
            port=config["DATABASE_PORT"],
            database=config["DATABASE_NAME"],
            username=config["DATABASE_USER"],
            password=config["DATABASE_PASSWORD"],
        )

    engine = create_engine(sql_url)

    return engine