import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from utils.logger import create_logger
from utils.app_utils import ComplexRadar
from utils.db_utils import get_connection

logger = create_logger()

STATS = ['pass', 'cross', 'foul', 'tackle', 'interception', 'shot', 'clearance', 'bad_touch', 'dribble']

@st.cache_data
def load_data():
    engine = get_connection()

    query = f"SELECT * FROM dashboard"
    stats = pd.read_sql(query, engine)

    stats = stats[(~stats.team.isnull()) & (~stats.player.isnull())].reset_index(drop=True)

    stats = stats.groupby(["team", "player"])[STATS].mean()
    stats.fillna(0, inplace=True)
    stats.reset_index(inplace=True)
    return stats


def create_radar_chart(df1: pd.DataFrame, ranges:list[tuple], df2: pd.DataFrame=None):
    """
    Create a radar chart
    """      
    
    fig = plt.figure(figsize=(5, 5))
    radar = ComplexRadar(fig, STATS, ranges)
    radar.plot(df1[STATS].iloc[0].values, label=df1["player"].iloc[0])
    radar.fill(df1[STATS].iloc[0].values, alpha=0.2)
    
    if df2 is not None:
        radar.plot(df2[STATS].iloc[0].values, label=df2["player"].iloc[0])
        radar.fill(df2[STATS].iloc[0].values, alpha=0.2)

    return fig

def main():
    # Set page title
    st.set_page_config(page_title="Player Stats Visualization", layout="wide")
    st.sidebar.header("Player Stats Visualization")

    # Load data
    df = load_data()

    # set range of each statistic
    ranges = [(0, df[col].max()) for col in STATS]  

    # Player name selection
    unique_teams = df["team"].unique()

    # Checkbox for comparing two players
    compare_players = st.sidebar.checkbox("Compare Two Players")
    
    # Select first player
    selected_team1 = st.sidebar.selectbox(
            "Select First Team", 
            unique_teams,
            key="team1_select"
        )
    
    # Select second player
    selected_team2 = None
    if compare_players:
        selected_team2 = st.sidebar.selectbox(
            "Select Second Team", 
            unique_teams,
            key="team2_select"
        )

    logger.info(f'selected_team1 {selected_team1}, selected_team2 {selected_team2}')

    unique_players1 = df[df.team == selected_team1]["player"].unique()
    unique_players2 = df[df.team == selected_team2]["player"].unique()

    # Select first player
    selected_player1 = st.sidebar.selectbox(
            "Select First Player", 
            unique_players1,
            key="player1_select"
        )
    
    # Select second player
    selected_player2 = None
    if compare_players:
        selected_player2 = st.sidebar.selectbox(
            "Select Second Player", 
            [p for p in unique_players2 if p != selected_player1],
            key="player2_select"
        )

    logger.info(f'selected_player1 {selected_player1}, selected_player2 {selected_player2}')

    # Filter dataframe based on selections
    filtered_df1 = df[(df['team'] == selected_team1) & (df["player"] == selected_player1)]

    filtered_df2 = None
    if selected_player2:
        filtered_df2 = df[(df['team'] == selected_team2) & (df["player"] == selected_player2)]

    if len(filtered_df1) > 0:
        # Main content area
        if compare_players:
            st.title(f"Comparison: {selected_player1} vs {selected_player2}")
        else:
            st.title(f"Statistics for {selected_player1}")

        # Display the radar chart
        _, col2, _ = st.columns([1,3,1])
        
        fig = create_radar_chart(filtered_df1, ranges, filtered_df2)
        
        with col2:
            st.pyplot(fig)
            st.markdown("\* Average statistics per match")

if __name__ == "__main__":
    main()
