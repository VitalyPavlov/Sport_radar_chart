import os
from tqdm import tqdm
import soccerdata as sd

os.environ['SOCCERDATA_LOGLEVEL'] = 'WARNING'

LEAGUES = [
    'ENG-Premier League',
    'ESP-La Liga',
    'FRA-Ligue 1',
    'GER-Bundesliga',
    'ITA-Serie A',
    'INT-European Championship',
    'INT-World Cup'
 ]

SEASONS = [
    '18-19',
    '19-20',
    '20-21',
    '21-22',
    '22-23',
    '23-24',
]


def main():
    for league in LEAGUES:
        for season in SEASONS:
            try:
                ws = sd.WhoScored(leagues=league, seasons=season, no_cache=False, no_store=False)
                schedule = ws.read_schedule(force_cache=True)
                for match_id in tqdm(schedule.game_id.values):
                    events_df = ws.read_events(match_id=int(match_id), output_fmt="events", force_cache=True)
                    del events_df
            except:
                continue


if __name__ == "__main__":
    main()
