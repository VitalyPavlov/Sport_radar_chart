
import pytest
import time
import numpy as np
import pandas as pd
from pathlib import Path

from db.database import DataBase
from utils.db_utils import get_connection

RESPONSE_TIME_THRESHOLD = 1.0
STATS = ['pass', 'cross', 'throw_in', 'freekick_crossed',
        'freekick_short', 'corner_crossed', 'corner_short',
        'take_on', 'foul', 'tackle', 'interception', 'shot',
        'shot_penalty', 'shot_freekick', 'keeper_save', 'keeper_claim',
        'keeper_punch', 'keeper_pick_up', 'clearance', 'bad_touch',
        'non_action', 'dribble','goalkick']

@pytest.mark.app
def test_db_connection():
    try:
        start_time = time.time()

        engine = get_connection()
        conn = engine.connect()
        
        response_time = time.time() - start_time
        assert response_time <= RESPONSE_TIME_THRESHOLD, (
            f"Database response time {response_time:.2f}s exceeded threshold of {RESPONSE_TIME_THRESHOLD}s"
        )

        conn.close()

    except Exception as e:
        pytest.fail(f"Database connection failed: {e}")

@pytest.mark.app
def test_db_tables():
    engine = get_connection(test_mode=True)

    query = "SELECT * FROM spadl LIMIT 1"
    stats_info = pd.read_sql(query, engine)
    assert len(stats_info) == 1
    
    query = "SELECT * FROM dashboard LIMIT 1"
    dashboard = pd.read_sql(query, engine)
    assert len(dashboard) == 1

# @pytest.mark.app
# def test_db_consistency():
#     file = Path("./test/data/dashboard_history.csv")

#     engine = get_connection(test_mode=True)

#     query = query = f"""
#         SELECT * FROM dashboard 
#         WHERE game_id = 1284741
#     """
#     data = pd.read_sql(query, engine)

#     if not file.exists():
#         data.to_csv(file, index=False)
    
#     reference = pd.read_csv(file)

#     data.sort_values(["game_id", "player_id"], inplace=True)
#     reference.sort_values(["game_id", "player_id"], inplace=True)

#     data = data[STATS].astype(float).reset_index(drop=True)
#     reference = reference[STATS].astype(float).reset_index(drop=True)

#     assert np.array_equal(data.values, reference.values)
