import json
import pandas as pd
from pathlib import Path
from utils.soccerdata_utils import WhoScored
from tqdm import tqdm
import time
from socceraction.spadl import config as spadlconfig

from utils.logger import create_logger
from utils.db_utils import get_connection
from db.database import DataBase


logger = create_logger()

LEAGUES = [
    'ENG-Premier League',
    # 'ESP-La Liga',
    # 'FRA-Ligue 1',
    # 'GER-Bundesliga',
    # 'ITA-Serie A',
    # 'INT-European Championship',
    # 'INT-World Cup'
 ]

SEASONS = [
    '18-19',
    # '19-20',
    # '20-21',
    # '21-22',
    # '22-23',
    # '23-24',
]

def save_data(db:DataBase, test_mode=False) -> None:
    """ Saving data from files in database db"""

    leagues = LEAGUES[:1] if test_mode else LEAGUES
    seasons = SEASONS[:1] if test_mode else SEASONS

    for league in leagues:
        for season in seasons:
            logger.info(f"Saving {league} {season}")
            try:
                ws = WhoScored(leagues=league, seasons=season)
                schedule = ws.read_schedule(force_cache=True)
            except:
                continue

            for match_id in schedule.game_id.values:
                spadl_df = ws.read_events(match_id=int(match_id), output_fmt="spadl", force_cache=True)

                if len(spadl_df) == 0 or 'type_id' not in spadl_df.columns:
                    continue

                spadl_df['type'] = spadl_df['type_id'].apply(lambda x: spadlconfig.actiontypes[x])
                
                db.save_spadl(spadl_df)

                if test_mode:
                    break
    
    logger.info(f"Spadl data was saved in database")
    
    db.save_dashboard()
    logger.info(f"Dashboard was saved in database")  

def main():
    # Database initialization
    engine = get_connection()
    db = DataBase(engine)
    db.init_db()

    logger.info("Tables in database were initializated")

    # # Saving files into DB
    save_data(db)
    logger.info("Create DB was finished")

    # while True:
    #     time.sleep(5)

if __name__ == "__main__":
    main()
