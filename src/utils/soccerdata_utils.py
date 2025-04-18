

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
from lxml import html
from typing import Callable, Literal, Optional, Union, IO
from collections.abc import Iterable
from enum import Enum
import warnings
import pprint
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from dateutil.relativedelta import relativedelta

from utils.soccerdata_config import DATA_DIR, LEAGUE_DICT, MAXAGE, TEAMNAME_REPLACEMENTS, logger

WHOSCORED_DATADIR = DATA_DIR / "WhoScored"

COLS_EVENTS = {
    # The ID of the game
    "game_id": np.nan,
    # 'PreMatch', 'FirstHalf', 'SecondHalf', 'PostGame'
    "period": np.nan,
    # Integer indicating the minute of the event, ignoring stoppage time
    "minute": -1,
    # Integer indicating the second of the event, ignoring stoppage time
    "second": -1,
    # Integer indicating the minute of the event, taking into account stoppage time
    "expanded_minute": -1,
    # String describing the event type (e.g. 'Goal', 'Yellow Card', etc.)
    "type": np.nan,
    # String describing the event outcome ('Succesful' or 'Unsuccessful')
    "outcome_type": np.nan,
    # The ID of the team that the event is associated with
    "team_id": np.nan,
    # The name of the team that the event is associated with
    "team": np.nan,
    # The ID of the player that the event is associated with
    "player_id": np.nan,
    # The name of the player that the event is associated with
    "player": np.nan,
    # Coordinates of the event's location
    "x": np.nan,
    "y": np.nan,
    "end_x": np.nan,
    "end_y": np.nan,
    # Coordinates of a shot's location
    "goal_mouth_y": np.nan,
    "goal_mouth_z": np.nan,
    # The coordinates where the ball was blocked
    "blocked_x": np.nan,
    "blocked_y": np.nan,
    # List of dicts with event qualifiers
    "qualifiers": [],
    # Some boolean flags
    "is_touch": False,
    "is_shot": False,
    "is_goal": False,
    # 'Yellow', 'Red', 'SecondYellow'
    "card_type": np.nan,
    # The ID of an associated event
    "related_event_id": np.nan,
    # The ID of a secondary player that the event is associated with
    "related_player_id": np.nan,
}

def _parse_url(url: str) -> dict:
    """Parse a URL from WhoScored.

    Parameters
    ----------
    url : str
        URL to parse.

    Raises
    ------
    ValueError
        If the URL could not be parsed.

    Returns
    -------
    dict
    """
    patt = (
        r"^(?:https:\/\/www.whoscored.com)?\/"
        + r"(?:regions\/(\d+)\/)?"
        + r"(?:tournaments\/(\d+)\/)?"
        + r"(?:seasons\/(\d+)\/)?"
        + r"(?:stages\/(\d+)\/)?"
        + r"(?:matches\/(\d+)\/)?"
    )
    matches = re.search(patt, url, re.IGNORECASE)
    if matches:
        return {
            "region_id": matches.group(1),
            "league_id": matches.group(2),
            "season_id": matches.group(3),
            "stage_id": matches.group(4),
            "match_id": matches.group(5),
        }
    raise ValueError(f"Could not parse URL: {url}")


def make_game_id(row: pd.Series) -> str:
    """Return a game id based on date, home and away team."""
    if pd.isnull(row["date"]):
        game_id = "{}-{}".format(
            row["home_team"],
            row["away_team"],
        )
    else:
        game_id = "{} {}-{}".format(
            row["date"].strftime("%Y-%m-%d"),
            row["home_team"],
            row["away_team"],
        )
    return game_id

def standardize_colnames(df: pd.DataFrame, cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Convert DataFrame column names to snake case."""

    def to_snake(name: str) -> str:
        name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        name = re.sub("__([A-Z])", r"_\1", name)
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name)
        return name.lower().replace("-", "_").replace(" ", "")

    if df.columns.nlevels > 1 and cols is None:
        # only standardize the first level
        new_df = df.copy()
        new_cols = [to_snake(c) for c in df.columns.levels[0]]
        new_df.columns = new_df.columns.set_levels(new_cols, level=0)
        return new_df

    if cols is None:
        cols = list(df.columns)

    return df.rename(columns={c: to_snake(c) for c in cols})


class SeasonCode(Enum):
    """How to interpret season codes.

    Attributes
    ----------
    SINGLE_YEAR: The season code is a single year, e.g. '2021'.
    MULTI_YEAR: The season code is a range of years, e.g. '2122'.
    """

    SINGLE_YEAR = "single-year"
    MULTI_YEAR = "multi-year"

    @staticmethod
    def from_league(league: str) -> "SeasonCode":
        """Return the default season code for a league.

        Parameters
        ----------
        league : str
            The league to consider.

        Raises
        ------
        ValueError
            If the league is not recognized.

        Returns
        -------
        SeasonCode
            The season code format to use.
        """
        if league not in LEAGUE_DICT:
            raise ValueError(f"Invalid league '{league}'")
        select_league_dict = LEAGUE_DICT[league]
        if "season_code" in select_league_dict:
            return SeasonCode(select_league_dict["season_code"])
        start_month = datetime.strptime(  # noqa: DTZ007
            select_league_dict.get("season_start", "Aug"),
            "%b",
        ).month
        end_month = datetime.strptime(  # noqa: DTZ007
            select_league_dict.get("season_end", "May"),
            "%b",
        ).month
        return SeasonCode.MULTI_YEAR if (end_month - start_month) < 0 else SeasonCode.SINGLE_YEAR

    @staticmethod
    def from_leagues(leagues: list[str]) -> "SeasonCode":
        """Determine the season code to use for a set of leagues.

        If the given leagues have different default season codes,
        the multi-year format is usded.

        Parameters
        ----------
        leagues : list of str
            The leagues to consider.

        Returns
        -------
        SeasonCode
            The season code format to use.
        """
        season_codes = {SeasonCode.from_league(league) for league in leagues}
        if len(season_codes) == 1:
            return season_codes.pop()
        warnings.warn(
            "The leagues have different default season codes. Using multi-year season codes.",
            stacklevel=2,
        )
        return SeasonCode.MULTI_YEAR

    def parse(self, season: Union[str, int]) -> str:  # noqa: C901
        """Convert a string or int to a standard season format."""
        season = str(season)
        patterns = [
            (
                re.compile(r"^[0-9]{4}$"),  # 1994 | 9495
                lambda s: process_four_digit_year(s),
            ),
            (
                re.compile(r"^[0-9]{2}$"),  # 94
                lambda s: process_two_digit_year(s),
            ),
            (
                re.compile(r"^[0-9]{4}-[0-9]{4}$"),  # 1994-1995
                lambda s: process_full_year_range(s),
            ),
            (
                re.compile(r"^[0-9]{4}/[0-9]{4}$"),  # 1994/1995
                lambda s: process_full_year_range(s.replace("/", "-")),
            ),
            (
                re.compile(r"^[0-9]{4}-[0-9]{2}$"),  # 1994-95
                lambda s: process_partial_year_range(s),
            ),
            (
                re.compile(r"^[0-9]{2}-[0-9]{2}$"),  # 94-95
                lambda s: process_short_year_range(s),
            ),
            (
                re.compile(r"^[0-9]{2}/[0-9]{2}$"),  # 94/95
                lambda s: process_short_year_range(s.replace("/", "-")),
            ),
        ]

        current_year = datetime.now(tz=timezone.utc).year

        def process_four_digit_year(season: str) -> str:
            """Process a 4-digit string like '1994' or '9495'."""
            if self == SeasonCode.MULTI_YEAR:
                if int(season[2:]) == int(season[:2]) + 1:
                    if season == "2021":
                        msg = (
                            f'Season id "{season}" is ambiguous: '
                            f'interpreting as "{season[:2]}-{season[-2:]}"'
                        )
                        warnings.warn(msg, stacklevel=1)
                    return season
                if season[2:] == "99":
                    return "9900"
                return season[-2:] + f"{int(season[-2:]) + 1:02d}"
            if season == "1920":
                return "1919"
            if season == "2021":
                return "2020"
            if season[:2] == "19" or season[:2] == "20":
                return season
            if int(season) <= current_year:
                return "20" + season[:2]
            return "19" + season[:2]

        def process_two_digit_year(season: str) -> str:
            """Process a 2-digit string like '94'."""
            if self == SeasonCode.MULTI_YEAR:
                if season == "99":
                    return "9900"
                return season + f"{int(season) + 1:02d}"
            if int("20" + season) <= current_year:
                return "20" + season
            return "19" + season

        def process_full_year_range(season: str) -> str:
            """Process a range of 4-digit strings like '1994-1995'."""
            if self == SeasonCode.MULTI_YEAR:
                return season[2:4] + season[-2:]
            return season[:4]

        def process_partial_year_range(season: str) -> str:
            """Process a range of 4-digit and 2-digit string like '1994-95'."""
            if self == SeasonCode.MULTI_YEAR:
                return season[2:4] + season[-2:]
            return season[:4]

        def process_short_year_range(season: str) -> str:
            """Process a range of 2-digit strings like '94-95'."""
            if self == SeasonCode.MULTI_YEAR:
                return season[:2] + season[-2:]
            if int("20" + season[:2]) <= current_year:
                return "20" + season[:2]
            return "19" + season[:2]

        for pattern, action in patterns:
            if pattern.match(season):
                return action(season)

        raise ValueError(f"Unrecognized season code: '{season}'")


class BaseReader(ABC):
    """Base class for data readers.

    Parameters
    ----------
    leagues : str or list of str, optional
        The leagues to read. If None, all available leagues are read.
    proxy : 'tor' or or dict or list(dict) or callable, optional
        Use a proxy to hide your IP address. Valid options are:
            - "tor": Uses the Tor network. Tor should be running in
              the background on port 9050.
            - dict: A dictionary with the proxy to use. The dict should be
              a mapping of supported protocols to proxy addresses. For example::

                  {
                      'http': 'http://10.10.1.10:3128',
                      'https': 'http://10.10.1.10:1080',
                  }

            - list(dict): A list of proxies to choose from. A different proxy will
              be selected from this list after failed requests, allowing rotating
              proxies.
            - callable: A function that returns a valid proxy. This function will
              be called after failed requests, allowing rotating proxies.
    no_cache : bool
        If True, will not use cached data.
    no_store : bool
        If True, will not store downloaded data.
    data_dir : Path
        Path to directory where data will be cached.
    """

    def __init__(
        self,
        leagues: Optional[Union[str, list[str]]] = None,
        proxy: Optional[
            Union[str, dict[str, str], list[dict[str, str]], Callable[[], dict[str, str]]]
        ] = None,
        no_cache: bool = False,
        no_store: bool = False,
        data_dir: Path = DATA_DIR,
    ):
        """Create a new data reader."""
        if isinstance(proxy, str) and proxy.lower() == "tor":
            self.proxy = lambda: {
                "http": "socks5://127.0.0.1:9050",
                "https": "socks5://127.0.0.1:9050",
            }
        elif isinstance(proxy, dict):
            self.proxy = lambda: proxy
        elif isinstance(proxy, list):
            self.proxy = lambda: random.choice(proxy)
        elif callable(proxy):
            self.proxy = proxy
        else:
            self.proxy = dict

        self._selected_leagues = leagues  # type: ignore
        self.no_cache = no_cache
        self.no_store = no_store
        self.data_dir = data_dir
        self.rate_limit = 0
        self.max_delay = 0
        if self.no_store:
            logger.info("Caching is disabled")
        else:
            logger.info("Saving cached data to %s", self.data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

    def get(
        self,
        url: str,
        filepath: Optional[Path] = None,
        max_age: Optional[Union[int, timedelta]] = MAXAGE,
        no_cache: bool = False,
        var: Optional[Union[str, Iterable[str]]] = None,
    ) -> IO[bytes]:
        """Load data from `url`.

        By default, the source of `url` is downloaded and saved to `filepath`.
        If `filepath` exists, the `url` is not visited and the cached data is
        returned.

        Parameters
        ----------
        url : str
            URL to download.
        filepath : Path, optional
            Path to save downloaded file. If None, downloaded data is not cached.
        max_age : int for age in days, or timedelta object
            The max. age of locally cached file before re-download.
        no_cache : bool
            If True, will not use cached data. Overrides the class property.
        var : str or list of str, optional
            Return a JavaScript variable instead of the page source.

        Raises
        ------
        TypeError
            If max_age is not an integer or timedelta object.

        Returns
        -------
        io.BufferedIOBase
            File-like object of downloaded data.
        """
        is_cached = self._is_cached(filepath, max_age)
        if no_cache or self.no_cache or not is_cached:
            logger.debug("Scraping %s", url)
            return self._download_and_save(url, filepath, var)
        logger.debug("Retrieving %s from cache", url)
        if filepath is None:
            raise ValueError("No filepath provided for cached data.")
        return filepath.open(mode="rb")

    def _is_cached(
        self,
        filepath: Optional[Path] = None,
        max_age: Optional[Union[int, timedelta]] = None,
    ) -> bool:
        """Check if `filepath` contains valid cached data.

        Parameters
        ----------
        filepath : Path, optional
            Path where file should be cached. If None, return False.
        max_age : int for age in days, or timedelta object
            The max. age of locally cached file.

        Raises
        ------
        TypeError
            If max_age is not an integer or timedelta object.

        Returns
        -------
        bool
            True in case of a cache hit, otherwise False.
        """
        # Validate inputs
        if max_age is not None:
            if isinstance(max_age, int):
                _max_age = timedelta(days=max_age)
            elif isinstance(max_age, timedelta):
                _max_age = max_age
            else:
                raise TypeError("'max_age' must be of type int or datetime.timedelta")
        else:
            _max_age = None

        cache_invalid = False
        # Check if cached file is too old
        if _max_age is not None and filepath is not None and filepath.exists():
            last_modified = datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc)
            now = datetime.now(timezone.utc)
            if (now - last_modified) > _max_age:
                cache_invalid = True

        return not cache_invalid and filepath is not None and filepath.exists()

    @abstractmethod
    def _download_and_save(
        self,
        url: str,
        filepath: Optional[Path] = None,
        var: Optional[Union[str, Iterable[str]]] = None,
    ) -> IO[bytes]:
        """Download data at `url` to `filepath`.

        Parameters
        ----------
        url : str
            URL to download.
        filepath : Path, optional
            Path to save downloaded file. If None, downloaded data is not cached.
        var : str or list of str, optional
            Return a JavaScript variable instead of the page source.

        Returns
        -------
        io.BufferedIOBase
            File-like object of downloaded data.
        """

    @classmethod
    def available_leagues(cls) -> list[str]:
        """Return a list of league IDs available for this source."""
        return sorted(cls._all_leagues().keys())

    @classmethod
    def _all_leagues(cls) -> dict[str, str]:
        """Return a dict mapping all canonical league IDs to source league IDs."""
        if not hasattr(cls, "_all_leagues_dict"):
            cls._all_leagues_dict = {  # type: ignore
                k: v[cls.__name__] for k, v in LEAGUE_DICT.items() if cls.__name__ in v
            }
        return cls._all_leagues_dict  # type: ignore

    @classmethod
    def _translate_league(cls, df: pd.DataFrame, col: str = "league") -> pd.DataFrame:
        """Map source league ID to canonical ID."""
        flip = {v: k for k, v in cls._all_leagues().items()}
        mask = ~df[col].isin(flip)
        df.loc[mask, col] = np.nan
        df[col] = df[col].replace(flip)
        return df

    @property
    def _selected_leagues(self) -> dict[str, str]:
        """Return a dict mapping selected canonical league IDs to source league IDs."""
        return self._leagues_dict

    @_selected_leagues.setter
    def _selected_leagues(self, ids: Optional[Union[str, list[str]]] = None) -> None:
        if ids is None:
            self._leagues_dict = self._all_leagues()
        else:
            if len(ids) == 0:
                raise ValueError("Empty iterable not allowed for 'leagues'")
            if isinstance(ids, str):
                ids = [ids]
            tmp_league_dict = {}
            for i in ids:
                if i not in self._all_leagues():
                    raise ValueError(
                        f"""
                        Invalid league '{i}'. Valid leagues are:
                        {pprint.pformat(self.available_leagues())}
                        """
                    )
                tmp_league_dict[i] = self._all_leagues()[i]
            self._leagues_dict = tmp_league_dict

    @property
    def _season_code(self) -> SeasonCode:
        return SeasonCode.from_leagues(self.leagues)

    def _is_complete(self, league: str, season: str) -> bool:
        """Check if a season is complete."""
        if league in LEAGUE_DICT:
            league_dict = LEAGUE_DICT[league]
        else:
            flip = {v: k for k, v in self._all_leagues().items()}
            if league in flip:
                league_dict = LEAGUE_DICT[flip[league]]
            else:
                raise ValueError(f"Invalid league '{league}'")
        if "season_end" not in league_dict:
            season_ends = datetime(
                datetime.strptime(season[-2:], "%y").year,  # noqa: DTZ007
                7,
                1,
                tzinfo=timezone.utc,
            )
        else:
            season_ends = datetime(
                datetime.strptime(season[-2:], "%y").year,  # noqa: DTZ007
                datetime.strptime(  # noqa: DTZ007
                    league_dict["season_end"], "%b"
                ).month,
                1,
                tzinfo=timezone.utc,
            ) + relativedelta(months=1)
        return datetime.now(tz=timezone.utc) >= season_ends

    @property
    def leagues(self) -> list[str]:
        """Return a list of selected leagues."""
        return list(self._leagues_dict.keys())

    @property
    def seasons(self) -> list[str]:
        """Return a list of selected seasons."""
        return self._season_ids

    @seasons.setter
    def seasons(self, seasons: Optional[Union[str, int, Iterable[Union[str, int]]]]) -> None:
        if seasons is None:
            logger.info("No seasons provided. Will retrieve data for the last 5 seasons.")
            year = datetime.now(tz=timezone.utc).year
            seasons = [f"{y - 1}-{y}" for y in range(year, year - 6, -1)]
        if isinstance(seasons, (str, int)):
            seasons = [seasons]
        self._season_ids = [self._season_code.parse(s) for s in seasons]


class WhoScored(BaseReader):
    def __init__(
        self, 
        leagues: Optional[Union[str, list[str]]] = None,
        seasons: Optional[Union[str, int, Iterable[Union[str, int]]]] = None,
    ):
        
        data_dir: Path = WHOSCORED_DATADIR
        """Initialize the WhoScored reader."""
        super().__init__(
            leagues=leagues,
            data_dir=data_dir,
        )

        self.seasons = seasons
    

    def read_leagues(self) -> pd.DataFrame:
        """Retrieve the selected leagues from the datasource.

        Returns
        -------
        pd.DataFrame
        """
        filepath = self.data_dir / "tiers.json"
        reader = filepath.open(mode="rb")
        data = json.load(reader)

        leagues = []
        for region in data:
            for league in region["tournaments"]:
                leagues.append(
                    {
                        "region_id": region["id"],
                        "region": region["name"],
                        "league_id": league["id"],
                        "league": league["name"],
                    }
                )

        return (
            pd.DataFrame(leagues)
            .assign(league=lambda x: x.region + " - " + x.league)
            .pipe(self._translate_league)
            .set_index("league")
            .loc[self._selected_leagues.keys()]
            .sort_index()
        )
    
    def read_seasons(self) -> pd.DataFrame:
        """Retrieve the selected seasons for the selected leagues.

        Returns
        -------
        pd.DataFrame
        """
        df_leagues = self.read_leagues()

        seasons = []
        for lkey, league in df_leagues.iterrows():
            filemask = "seasons/{}.html"
            filepath = self.data_dir / filemask.format(lkey)
            
            if not filepath.is_file():
                continue

            reader = filepath.open(mode="rb")

            # extract team links
            tree = html.parse(reader)
            for node in tree.xpath("//select[contains(@id,'seasons')]/option"):
                # extract team IDs from links
                season_url = node.get("value")
                season_id = _parse_url(season_url)["season_id"]
                seasons.append(
                    {
                        "league": lkey,
                        "season": self._season_code.parse(node.text),
                        "region_id": league.region_id,
                        "league_id": league.league_id,
                        "season_id": season_id,
                    }
                )
        
        idxs = [xx for xx in itertools.product(self.leagues, self.seasons) if xx in [(x['league'], x['season']) for x in seasons]]
        return (
            pd.DataFrame(seasons)
            .set_index(["league", "season"])
            .sort_index()
            .loc[idxs]
        )
    
    def read_season_stages(self, force_cache: bool = False) -> pd.DataFrame:
        """Retrieve the season stages for the selected leagues.

        Parameters
        ----------
        force_cache : bool
             By default no cached data is used for the current season.
             If True, will force the use of cached data anyway.

        Returns
        -------
        pd.DataFrame
        """
        df_seasons = self.read_seasons()
        filemask = "seasons/{}_{}.html"

        season_stages = []
        for (lkey, skey), season in df_seasons.iterrows():
            current_season = not self._is_complete(lkey, skey)

            # get season page
            filepath = self.data_dir / filemask.format(lkey, skey)

            if not filepath.is_file():
                continue

            reader = filepath.open(mode="rb")
            tree = html.parse(reader)

            # get default season stage
            fixtures_url = tree.xpath("//a[text()='Fixtures']/@href")[0]
            stage_id = _parse_url(fixtures_url)["stage_id"]
            season_stages.append(
                {
                    "league": lkey,
                    "season": skey,
                    "region_id": season.region_id,
                    "league_id": season.league_id,
                    "season_id": season.season_id,
                    "stage_id": stage_id,
                    "stage": None,
                }
            )

            # extract additional stages
            for node in tree.xpath("//select[contains(@id,'stages')]/option"):
                stage_url = node.get("value")
                stage_id = _parse_url(stage_url)["stage_id"]
                season_stages.append(
                    {
                        "league": lkey,
                        "season": skey,
                        "region_id": season.region_id,
                        "league_id": season.league_id,
                        "season_id": season.season_id,
                        "stage_id": stage_id,
                        "stage": node.text,
                    }
                )

        idxs = [xx for xx in itertools.product(self.leagues, self.seasons) if xx in [(x['league'], x['season']) for x in season_stages]]
        return (
            pd.DataFrame(season_stages)
            .drop_duplicates(subset=["league", "season", "stage_id"], keep="last")
            .set_index(["league", "season"])
            .sort_index()
            .loc[idxs]
        )
    
    def read_schedule(self, force_cache: bool = False) -> pd.DataFrame:
        """Retrieve the game schedule for the selected leagues and seasons.

        Parameters
        ----------
        force_cache : bool
             By default no cached data is used for the current season.
             If True, will force the use of cached data anyway.

        Returns
        -------
        pd.DataFrame
        """
        df_season_stages = self.read_season_stages(force_cache=force_cache)
        filemask_schedule = "matches/{}_{}_{}_{}.json"

        all_schedules = []
        for (lkey, skey), stage in df_season_stages.iterrows():
            current_season = not self._is_complete(lkey, skey)
            stage_id = stage["stage_id"]
            stage_name = stage["stage"]

            # get the calendar of the season stage
            if stage_name is not None:
                calendar_filepath = self.data_dir / f"matches/{lkey}_{skey}_{stage_id}.html"
                logger.info(
                    "Retrieving calendar for %s %s (%s)",
                    lkey,
                    skey,
                    stage_name,
                )
            else:
                calendar_filepath = self.data_dir / f"matches/{lkey}_{skey}.html"
                logger.info(
                    "Retrieving calendar for %s %s",
                    lkey,
                    skey,
                )

            if not calendar_filepath.is_file():
                continue

            calendar = calendar_filepath.open(mode="rb")
            mask = json.load(calendar)["mask"]

            # get the fixtures for each month
            it = [(year, month) for year in mask for month in mask[year]]
            for i, (year, month) in enumerate(it):
                filepath = self.data_dir / filemask_schedule.format(lkey, skey, stage_id, month)

                if stage_name is not None:
                    logger.info(
                        "[%s/%s] Retrieving fixtures for %s %s (%s)",
                        i + 1,
                        len(it),
                        lkey,
                        skey,
                        stage_name,
                    )
                else:
                    logger.info(
                        "[%s/%s] Retrieving fixtures for %s %s",
                        i + 1,
                        len(it),
                        lkey,
                        skey,
                    )

                if not filepath.is_file():
                    continue

                reader = filepath.open(mode="rb")
                data = json.load(reader)
                for tournament in data["tournaments"]:
                    df_schedule = pd.DataFrame(tournament["matches"])
                    df_schedule["league"] = lkey
                    df_schedule["season"] = skey
                    df_schedule["stage"] = stage_name
                    all_schedules.append(df_schedule)

        if len(all_schedules) == 0:
            return pd.DataFrame(index=["league", "season", "game"])

        # Construct the output dataframe
        return (
            pd.concat(all_schedules)
            .drop_duplicates(subset=["id"])
            .replace(
                {
                    "homeTeamName": TEAMNAME_REPLACEMENTS,
                    "awayTeamName": TEAMNAME_REPLACEMENTS,
                }
            )
            .rename(
                columns={
                    "homeTeamName": "home_team",
                    "awayTeamName": "away_team",
                    "id": "game_id",
                    "startTimeUtc": "date",
                }
            )
            .assign(date=lambda x: pd.to_datetime(x["date"]))
            .assign(game=lambda df: df.apply(make_game_id, axis=1))
            .pipe(standardize_colnames)
            .set_index(["league", "season", "game"])
            .sort_index()
        )
    
    def read_missing_players(
        self,
        match_id: Optional[Union[int, list[int]]] = None,
        force_cache: bool = False,
    ) -> pd.DataFrame:
        """Retrieve a list of injured and suspended players ahead of each game.

        Parameters
        ----------
        match_id : int or list of int, optional
            Retrieve the missing players for a specific game.
        force_cache : bool
            By default no cached data is used to scrapre the list of available
            games for the current season. If True, will force the use of
            cached data anyway.

        Raises
        ------
        ValueError
            If the given match_id could not be found in the selected seasons.

        Returns
        -------
        pd.DataFrame
        """
        filemask = "WhoScored/previews/{}_{}/{}.html"

        df_schedule = self.read_schedule(force_cache).reset_index()
        if match_id is not None:
            iterator = df_schedule[
                df_schedule.game_id.isin([match_id] if isinstance(match_id, int) else match_id)
            ]
            if len(iterator) == 0:
                raise ValueError("No games found with the given IDs in the selected seasons.")
        else:
            iterator = df_schedule.sample(frac=1)

        match_sheets = []
        for i, (_, game) in enumerate(iterator.iterrows()):
            filepath = DATA_DIR / filemask.format(game["league"], game["season"], game["game_id"])

            logger.info(
                "[%s/%s] Retrieving game with id=%s",
                i + 1,
                len(iterator),
                game["game_id"],
            )

            if not filepath.is_file():
                continue

            reader = filepath.open(mode="rb")

            # extract missing players
            tree = html.parse(reader)
            for node in tree.xpath("//div[@id='missing-players']/div[2]/table/tbody/tr"):
                # extract team IDs from links
                match_sheets.append(
                    {
                        "league": game["league"],
                        "season": game["season"],
                        "game": game["game"],
                        "game_id": game["game_id"],
                        "team": game["home_team"],
                        "player": node.xpath("./td[contains(@class,'pn')]/a")[0].text,
                        "player_id": int(
                            node.xpath("./td[contains(@class,'pn')]/a")[0]
                            .get("href")
                            .split("/")[2]
                        ),
                        "reason": node.xpath("./td[contains(@class,'reason')]/span")[0].get(
                            "title"
                        ),
                        "status": node.xpath("./td[contains(@class,'confirmed')]")[0].text,
                    }
                )
            for node in tree.xpath("//div[@id='missing-players']/div[3]/table/tbody/tr"):
                # extract team IDs from links
                match_sheets.append(
                    {
                        "league": game["league"],
                        "season": game["season"],
                        "game": game["game"],
                        "game_id": game["game_id"],
                        "team": game["away_team"],
                        "player": node.xpath("./td[contains(@class,'pn')]/a")[0].text,
                        "player_id": int(
                            node.xpath("./td[contains(@class,'pn')]/a")[0]
                            .get("href")
                            .split("/")[2]
                        ),
                        "reason": node.xpath("./td[contains(@class,'reason')]/span")[0].get(
                            "title"
                        ),
                        "status": node.xpath("./td[contains(@class,'confirmed')]")[0].text,
                    }
                )

        if len(match_sheets) == 0:
            return pd.DataFrame(index=["league", "season", "game", "team", "player"])

        return (
            pd.DataFrame(match_sheets)
            .set_index(["league", "season", "game", "team", "player"])
            .sort_index()
        )

    def read_events(  # noqa: C901
        self,
        match_id: Optional[Union[int, list[int]]] = None,
        force_cache: bool = False,
        live: bool = False,
        output_fmt: Optional[str] = "events",
        retry_missing: bool = True,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> Optional[Union[pd.DataFrame, dict[int, list], "OptaLoader"]]:  # type: ignore  # noqa: F821
        """Retrieve the the event data for each game in the selected leagues and seasons.

        Parameters
        ----------
        match_id : int or list of int, optional
            Retrieve the event stream for a specific game.
        force_cache : bool
            By default no cached data is used to scrape the list of available
            games for the current season. If True, will force the use of
            cached data anyway.
        live : bool
            If True, will not return a cached copy of the event data. This is
            usefull to scrape live data.
        output_fmt : str, default: 'events'
            The output format of the returned data. Possible values are:
                - 'events' (default): Returns a dataframe with all events.
                - 'raw': Returns the original unformatted WhoScored JSON.
                - 'spadl': Returns a dataframe with the SPADL representation
                  of the original events.
                  See https://socceraction.readthedocs.io/en/latest/documentation/SPADL.html#spadl
                - 'atomic-spadl': Returns a dataframe with the Atomic-SPADL representation
                  of the original events.
                  See https://socceraction.readthedocs.io/en/latest/documentation/SPADL.html#atomic-spadl
                - 'loader': Returns a socceraction.data.opta.OptaLoader
                  instance, which can be used to retrieve the actual data.
                  See https://socceraction.readthedocs.io/en/latest/modules/generated/socceraction.data.opta.OptaLoader.html#socceraction.data.opta.OptaLoader
                - None: Doesn't return any data. This is useful to just cache
                  the data without storing the events in memory.
        retry_missing : bool
            If no events were found for a game in a previous attempt, will
            retry to scrape the events
        on_error : "raise" or "skip", default: "raise"
            Wheter to raise an exception or to skip the game if an error occurs.

        Raises
        ------
        ValueError
            If the given match_id could not be found in the selected seasons.
        ConnectionError
            If the match page could not be retrieved.
        ImportError
            If the requested output format is 'spadl', 'atomic-spadl' or
            'loader' but the socceraction package is not installed.

        Returns
        -------
        See the description of the ``output_fmt`` parameter.
        """
        output_fmt = output_fmt.lower() if output_fmt is not None else None
        if output_fmt in ["loader", "spadl", "atomic-spadl"]:
            if self.no_store:
                raise ValueError(
                    f"The '{output_fmt}' output format is not supported "
                    "when using the 'no_store' option."
                )
            try:
                from socceraction.atomic.spadl import convert_to_atomic
                from socceraction.data.opta import OptaLoader
                from socceraction.data.opta.loader import _eventtypesdf
                from socceraction.data.opta.parsers import WhoScoredParser
                from socceraction.spadl.opta import convert_to_actions

                if output_fmt == "loader":
                    import socceraction
                    from packaging import version

                    if version.parse(socceraction.__version__) < version.parse("1.2.3"):
                        raise ImportError(
                            "The 'loader' output format requires socceraction >= 1.2.3"
                        )
            except ImportError:
                raise ImportError(
                    "The socceraction package is required to use the 'spadl' "
                    "or 'atomic-spadl' output format. "
                    "Please install it with `pip install socceraction`."
                )
        
        filemask = "events/{}_{}/{}.json"

        df_schedule = self.read_schedule(force_cache).reset_index()
        if match_id is not None:
            iterator = df_schedule[
                df_schedule.game_id.isin([match_id] if isinstance(match_id, int) else match_id)
            ]
            if len(iterator) == 0:
                raise ValueError("No games found with the given IDs in the selected seasons.")
        else:
            iterator = df_schedule.sample(frac=1)

        events = {}
        player_names = {}
        team_names = {}
        player_position = {}
        for i, (_, game) in enumerate(iterator.iterrows()):
            # get league and season
            logger.info(
                "[%s/%s] Retrieving game with id=%s",
                i + 1,
                len(iterator),
                game["game_id"],
            )
            filepath = self.data_dir / filemask.format(
                game["league"], game["season"], game["game_id"]
            )

            try:
                if not filepath.is_file():
                    continue

                reader = filepath.open(mode="rb")
                reader_value = reader.read()
                if retry_missing and reader_value == b"null" or reader_value == b"":
                    reader = filepath.open(mode="rb")
            except ConnectionError as e:
                if on_error == "skip":
                    logger.warning("Error while scraping game %s: %s", game["game_id"], e)
                    continue
                raise
            reader.seek(0)
            json_data = json.load(reader)
            if json_data is not None:
                player_names.update(
                    {int(k): v for k, v in json_data["playerIdNameDictionary"].items()}
                )
                team_names.update(
                    {
                        int(json_data[side]["teamId"]): json_data[side]["name"]
                        for side in ["home", "away"]
                    }
                )

                player_position.update(
                    {v['playerId']: v['position'] for v in json_data["home"]['players']}
                )
                player_position.update(
                    {v['playerId']: v['position'] for v in json_data["away"]['players']}
                )

                if "events" in json_data:
                    game_events = json_data["events"]
                    if output_fmt == "events":
                        df_events = pd.DataFrame(game_events)
                        df_events["game"] = game["game"]
                        df_events["league"] = game["league"]
                        df_events["season"] = game["season"]
                        df_events["game_id"] = game["game_id"]
                        events[game["game_id"]] = df_events
                    elif output_fmt == "raw":
                        events[game["game_id"]] = game_events
                    elif output_fmt in ["spadl", "atomic-spadl"]:
                        parser = WhoScoredParser(
                            str(filepath),
                            competition_id=game["league"],
                            season_id=game["season"],
                            game_id=game["game_id"],
                        )
                        df_events = (
                            pd.DataFrame.from_dict(parser.extract_events(), orient="index")
                            .merge(_eventtypesdf, on="type_id", how="left")
                            .reset_index(drop=True)
                        )
                        df_actions = convert_to_actions(
                            df_events, home_team_id=int(json_data["home"]["teamId"])
                        )
                        if output_fmt == "spadl":
                            events[game["game_id"]] = df_actions
                        else:
                            events[game["game_id"]] = convert_to_atomic(df_actions)

            else:
                logger.warning("No events found for game %s", game["game_id"])

        if output_fmt is None:
            return None

        if output_fmt == "raw":
            return events

        if output_fmt == "loader":
            return OptaLoader(
                root=self.data_dir,
                parser="whoscored",
                feeds={
                    "whoscored": str(Path("events/{competition_id}_{season_id}/{game_id}.json"))
                },
            )

        if len(events) == 0:
            return pd.DataFrame(index=["league", "season", "game"])

        df = (
            pd.concat(events.values())
            .pipe(standardize_colnames)
            .assign(
                player=lambda x: x.player_id.replace(player_names),
                team=lambda x: x.team_id.replace(team_names).replace(TEAMNAME_REPLACEMENTS),
                position=lambda x: x.player_id.replace(player_position),
            )
        )

        if output_fmt == "events":
            df = df.set_index(["league", "season", "game"]).sort_index()
            # add missing columns
            for col, default in COLS_EVENTS.items():
                if col not in df.columns:
                    df[col] = default
            df["outcome_type"] = df["outcome_type"].apply(
                lambda x: x.get("displayName") if pd.notnull(x) else x
            )
            df["card_type"] = df["card_type"].apply(
                lambda x: x.get("displayName") if pd.notnull(x) else x
            )
            df["type"] = df["type"].apply(lambda x: x.get("displayName") if pd.notnull(x) else x)
            df["period"] = df["period"].apply(
                lambda x: x.get("displayName") if pd.notnull(x) else x
            )
            df = df[list(COLS_EVENTS.keys())]

        return df
    
    def _download_and_save(
        self,
        url: str,
        filepath: Optional[Path] = None,
        var: Optional[Union[str, Iterable[str]]] = None,
    ) -> IO[bytes]:
        pass
