from datetime import datetime
import pandas as pd
from sqlalchemy import text
from sqlalchemy import MetaData, Table, Column, Integer, String, Float, DateTime, BigInteger

from utils.logger import create_logger


logger = create_logger()

STATS = ['pass', 'cross', 'throw_in', 'freekick_crossed',
                    'freekick_short', 'corner_crossed', 'corner_short',
                    'take_on', 'foul', 'tackle', 'interception', 'shot',
                    'shot_penalty', 'shot_freekick', 'keeper_save', 'keeper_claim',
                    'keeper_punch', 'keeper_pick_up', 'clearance', 'bad_touch',
                    'non_action', 'dribble','goalkick']


class DataBase:  
    def __init__(self, engine):  
        self.engine = engine

    def init_db(self) -> None:
        """ initialization of tables in PostgresDB """

        metadata = MetaData()
        spadl_table = Table(
            "spadl", metadata,
            Column("game_id", Integer, primary_key=True),
            Column("original_event_id", BigInteger),
            Column("period_id", Integer),
            Column("time_seconds", Float),
            Column("team_id", Integer),
            Column("player_id", Integer),
            Column("start_x", Float),
            Column("end_x", Float),
            Column("start_y", Float),
            Column("end_y", Float),
            Column("type_id", Integer),
            Column("type", String),
            Column("result_id", Integer),
            Column("bodypart_id", Integer),
            Column("action_id", Integer, primary_key=True),
            Column("player", String),
            Column("team", String),
            Column("position", String),
        )

        dashboard_table = Table(
            "dashboard", metadata,
            Column("game_id", Integer, primary_key=True),
            Column("team_id", Integer),
            Column("player_id", Integer, primary_key=True),
            Column("player", String),
            Column("team", String),
            Column("pass", Integer),
            Column("cross", Integer),
            Column("throw_in", Integer),
            Column("freekick_crossed", Integer),
            Column("freekick_short", Integer),
            Column("corner_crossed", Integer),
            Column("corner_short", Integer),
            Column("take_on", Integer),
            Column("foul", Integer),
            Column("tackle", Integer),
            Column("interception", Integer),
            Column("shot", Integer),
            Column("shot_penalty", Integer),
            Column("shot_freekick", Integer),
            Column("keeper_save", Integer),
            Column("keeper_claim", Integer),
            Column("keeper_punch", Integer),
            Column("keeper_pick_up", Integer),
            Column("clearance", Integer),
            Column("bad_touch", Integer),
            Column("non_action", Integer),
            Column("dribble", Integer),
            Column("goalkick", Integer),
            Column("timestamp", DateTime),
        )
        
        spadl_table.drop(self.engine, checkfirst=True)
        dashboard_table.drop(self.engine, checkfirst=True)

        metadata.create_all(self.engine)

    def save_spadl(self, data: pd.DataFrame) -> None:    
        columns = ['game_id', 'original_event_id', 'period_id', 'time_seconds', 'team_id',
                    'player_id', 'start_x', 'end_x', 'start_y', 'end_y', 'type_id', 'type',
                    'result_id', 'bodypart_id', 'action_id', 'player', 'team', 'position']
        
        try:
            data[columns].to_sql("spadl", self.engine, if_exists="append", index=False)
        except Exception as e:
            logger.error("Error with writing in spadl table")
            logger.error(e)

    def save_dashboard(self) -> None:
        """ Calculating statistics and saving it in dashboard table """

        query = f"SELECT * FROM spadl"
        df = pd.read_sql(query, self.engine)

        

        by_col = ["game_id", "team_id", "player_id", "player", "team"]
        stats = []
        for col in STATS:
            stats.append(df[df["type"] == col].groupby(by_col)["original_event_id"].count().rename(col))

        stats = pd.concat(stats, axis=1)
        
        stats["timestamp"] = datetime.now()
        stats.fillna(0, inplace=True)
        stats.reset_index(inplace=True)
        
        columns = by_col + STATS + ['timestamp']
        try:
            stats[columns].to_sql("dashboard", self.engine, if_exists="append", index=False)
        except Exception as e: 
            logger.error("Error with writing in dashboard table")
            logger.error(e)
