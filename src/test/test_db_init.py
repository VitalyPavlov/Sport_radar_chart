import pytest
import pandas as pd
from pathlib import Path

from db.database import DataBase
from db.create_db import save_data
from utils.db_utils import get_connection

@pytest.mark.code
def test_db_init():
    engine = get_connection(test_mode=True)
    db = DataBase(engine)
    db.init_db()
    
    save_data(db, test_mode=True)

    query = "SELECT * FROM spadl"
    stats_info = pd.read_sql(query, engine)
    assert len(stats_info) > 0
    
    query = "SELECT * FROM dashboard"
    dashboard = pd.read_sql(query, engine)
    assert len(dashboard) > 0