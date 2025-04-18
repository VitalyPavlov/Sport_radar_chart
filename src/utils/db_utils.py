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


def search_qualifier_id(x:str, ids:Any) -> bool:
    """ Check if events have special qualifierId """
    return any([y["qualifierId"] in ids for y in json.loads(x)])

def search_qualifier_value_by_ids(x:str, ids:Any) -> float:
    """ Take qualifier value if events have special qualifierId """
    return float([y["value"] for y in json.loads(x) if y["qualifierId"] in ids][0])

def get_lineup(engine) -> pd.DataFrame:
    """ Take line up for each fixture """

    query = f"SELECT fixture_id, line_up FROM stats_info"
    lineup = pd.read_sql(query, engine)

    lineup.set_index("fixture_id", inplace=True)
    temp = lineup["line_up"].apply(lambda x: {y["contestantId"]:[z["playerId"] for z in y["player"]] for y in json.loads(x)})
    lineup = pd.concat([pd.concat([pd.Series(y).rename("line_up").rename_axis("contestant_id").reset_index(), 
                                   pd.Series([x, x]).rename("fixture_id")], axis=1) for x,y in zip(temp.index, temp.values)], axis=0)
    
    lineup.set_index(["fixture_id", "contestant_id"], inplace=True)
    return lineup

def calculate_statistics(engine) -> pd.DataFrame:
    """ Statistics for dashboard """

    query = f"SELECT * FROM player_events"
    df = pd.read_sql(query, engine)
    
    by_col = ["fixture_id", "date", "contestant_id", "player_id", "player_name"]

    stats = []
    stats.append(df[df["type_id"] == 1].groupby(by_col)["id"].count().rename("total_passes"))
    stats.append(df[(df["type_id"] == 1) & (df["outcome"] == 1)].groupby(by_col)["id"].count().rename("success_passes"))
    stats.append(df[(df["type_id"].isin([13,14,15,60]))].groupby(by_col)["qualifier"].apply(lambda x: sum([search_qualifier_id(y, ids=[29,55]) for y in x])).rename("key_passes"))
    stats.append(df[df["type_id"] == 1].groupby(by_col)["qualifier"].apply(lambda x: sum([search_qualifier_id(y, ids=[1]) for y in x])).rename("long_passes"))
    stats.append(df[df["type_id"] == 1].groupby(by_col).apply(lambda x: sum([y > search_qualifier_value_by_ids(z, ids=[140]) for y, z in zip(x["x"], x["qualifier"])]), include_groups=False).rename("back_passes"))
    stats.append(df[(df["type_id"].isin([15,16]))].groupby(by_col)["id"].count().rename("shots_on_target"))
    stats.append(df[df["type_id"] == 44].groupby(by_col)["id"].count().rename("aerial_duels"))
    stats.append(df[df["type_id"] == 4].groupby(by_col)["qualifier"].apply(lambda x: sum([search_qualifier_id(y, ids=[264]) for y in x])).rename("aerial_duels_foul"))
    stats.append(df[(df["type_id"] == 44) & (df["outcome"] == 1)].groupby(by_col)["id"].count().rename("success_aerial_duels"))
    stats = pd.concat(stats, axis=1)

    stats["pass_success_rate"] = stats.success_passes / stats.total_passes
    stats["total_aerial_duels"] = stats["aerial_duels"] + stats["aerial_duels_foul"]
    stats["aerial_duel_success_rate"] = stats.success_aerial_duels / stats.total_aerial_duels

    stats["timestamp"] = datetime.now()
    stats.fillna(0, inplace=True)
    stats.reset_index(inplace=True)

    return stats
