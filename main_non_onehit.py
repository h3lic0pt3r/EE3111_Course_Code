from io import StringIO
import pandas as pd
import os
import glob
import numpy as np

from weights import *
# --- Sentiment Mappings ---
BATTING_STRENGTH = {
    'Sunrisers Hyderabad': 3, 'Kolkata Knight Riders': 3, 'Royal Challengers Bengaluru': 3, 'Royal Challengers Bangalore': 3, 'Mumbai Indians': 3,
    'Chennai Super Kings': 2, 'Rajasthan Royals': 2, 'Lucknow Super Giants': 2,
    'Delhi Capitals': 1, 'Punjab Kings': 1, 'Kings XI Punjab': 1, 'Gujarat Titans': 1
}

BOWLING_STRENGTH = {
    'Rajasthan Royals': 3, 'Kolkata Knight Riders': 3, 'Chennai Super Kings': 3, 'Gujarat Titans': 3,
    'Mumbai Indians': 2, 'Lucknow Super Giants': 2, 'Sunrisers Hyderabad': 2,
    'Royal Challengers Bengaluru': 1, 'Royal Challengers Bangalore': 1, 'Delhi Capitals': 1, 'Punjab Kings': 1, 'Kings XI Punjab': 1
}

TEAM_WINS_BY_YEAR = None

def build_team_wins_by_year(raw_dir="data/raw"):
    wins = {}
    if not os.path.exists(raw_dir):
        return wins
    info_files = glob.glob(os.path.join(raw_dir, "*_info.csv"))
    for info_filepath in info_files:
        try:
            with open(info_filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception:
            continue
        match_year = None
        winner = None
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) >= 3 and parts[1] == 'date':
                try:
                    match_year = int(parts[2][:4])
                except ValueError:
                    match_year = None
            if len(parts) >= 3 and parts[1] == 'winner':
                winner = ",".join(parts[2:]).strip().replace('"', "").replace("'", "")
        if match_year is not None and winner:
            wins[(match_year, winner)] = wins.get((match_year, winner), 0) + 1
    return wins

def predict(match_data, ball_by_ball_data):
    # --- 2. Safely Parse Ball-by-Ball Data ---
    try:
        df = pd.read_csv(StringIO(ball_by_ball_data))
        
        df["runs_off_bat"] = pd.to_numeric(df["runs_off_bat"], errors="coerce").fillna(0)
        df["extras"] = pd.to_numeric(df["extras"], errors="coerce").fillna(0)
        df["total_runs"] = df["runs_off_bat"] + df["extras"]
        
        # Over-by-over runs
        df['over_index'] = df['ball'].astype(int)
        over_0_runs = df[df['over_index'] == 0]["total_runs"].sum()
        over_1_runs = df[df['over_index'] == 1]["total_runs"].sum()
        over_2_runs = df[df['over_index'] == 2]["total_runs"].sum()
        total_after_3 = over_0_runs + over_1_runs + over_2_runs
        boundaries_4s6s = ((df["runs_off_bat"] == 4) | (df["runs_off_bat"] == 6)).sum()
        over_0_runs_sq = over_0_runs * over_0_runs
        over_1_runs_sq = over_1_runs * over_1_runs
        over_2_runs_sq = over_2_runs * over_2_runs
        boundaries_4s6s_sq = boundaries_4s6s * boundaries_4s6s
        
        dot_balls = (df["total_runs"] == 0).sum()
        wickets_after_3 = df["player_dismissed"].notna().sum() if "player_dismissed" in df.columns else 0
        wickets_squared = wickets_after_3 * wickets_after_3
        dot_balls_sq = dot_balls * dot_balls
        wickets_after_3_sq = wickets_after_3 * wickets_after_3
        wickets_squared_sq = wickets_squared * wickets_squared

        dismissed_sr_sum = 0.0
        if "player_dismissed" in df.columns and "striker" in df.columns:
            dismissed_players = df["player_dismissed"].dropna().unique()
            if len(dismissed_players) > 0:
                if "wides" in df.columns:
                    wides_col = pd.to_numeric(df["wides"], errors="coerce").fillna(0)
                else:
                    wides_col = pd.Series(0, index=df.index)
                for player in dismissed_players:
                    player_mask = df["striker"] == player
                    if not player_mask.any():
                        continue
                    runs_scored = df.loc[player_mask, "runs_off_bat"].sum()
                    balls_faced = (player_mask & (wides_col == 0)).sum()
                    if balls_faced > 0:
                        dismissed_sr_sum += (runs_scored / balls_faced) * 100
        dismissed_sr_sum_sq = dismissed_sr_sum * dismissed_sr_sum

        batter_runs = {}
        if "striker" in df.columns:
            for _, row_ball in df.iterrows():
                batter = row_ball.get("striker")
                if pd.isna(batter):
                    continue
                batter_runs[batter] = batter_runs.get(batter, 0) + float(row_ball.get("runs_off_bat", 0) or 0)
        batters_above_20 = sum(1 for runs in batter_runs.values() if runs > 20)
        batters_above_20 = min(2, batters_above_20)
        batters_above_20_sq = batters_above_20 * batters_above_20

        total_runs_log = float(np.log1p(total_after_3))
        striker_sr_sum = 0.0
        for batter_col in ["striker", "non_striker"]:
            if batter_col not in df.columns:
                continue
            batter = df[batter_col].iloc[0]
            if pd.isna(batter):
                continue
            player_mask = df["striker"] == batter if "striker" in df.columns else None
            if player_mask is None or not player_mask.any():
                continue
            runs_scored = df.loc[player_mask, "runs_off_bat"].sum()
            if "wides" in df.columns:
                wides_col = pd.to_numeric(df["wides"], errors="coerce").fillna(0)
            else:
                wides_col = pd.Series(0, index=df.index)
            balls_faced = (player_mask & (wides_col == 0)).sum()
            if balls_faced > 0:
                striker_sr_sum += (runs_scored / balls_faced) * 100
        striker_sr_sum_log = float(np.log1p(striker_sr_sum))
            
        batting_team = df["batting_team"].iloc[0] if "batting_team" in df.columns else "Unknown"
        bowling_team = df["bowling_team"].iloc[0] if "bowling_team" in df.columns else "Unknown"

        match_year = None
        for line in str(match_data).split('\n'):
            parts = line.split(',')
            if len(parts) >= 3 and parts[0] == 'info' and parts[1] == 'date':
                try:
                    match_year = int(parts[2][:4])
                except ValueError:
                    match_year = None
                break

        global TEAM_WINS_BY_YEAR
        if TEAM_WINS_BY_YEAR is None:
            TEAM_WINS_BY_YEAR = build_team_wins_by_year()
        team_wins_year = TEAM_WINS_BY_YEAR.get((match_year, batting_team), 0) if match_year is not None else 0
        team_wins_year_sq = team_wins_year * team_wins_year
            
    except Exception:
        # Fallback values
        print("FALLBACK")
        over_0_runs, over_1_runs, over_2_runs = 8, 8, 8
        total_after_3 = over_0_runs + over_1_runs + over_2_runs
        over_0_runs_sq = over_0_runs * over_0_runs
        over_1_runs_sq = over_1_runs * over_1_runs
        over_2_runs_sq = over_2_runs * over_2_runs
        boundaries_4s6s = 0
        boundaries_4s6s_sq = 0
        dot_balls = 6
        dot_balls_sq = dot_balls * dot_balls
        wickets_after_3 = 1
        wickets_squared = 1
        wickets_after_3_sq = wickets_after_3 * wickets_after_3
        wickets_squared_sq = wickets_squared * wickets_squared
        dismissed_sr_sum = 0.0
        dismissed_sr_sum_sq = 0.0
        batters_above_20 = 0
        batters_above_20_sq = 0
        team_wins_year = 0
        team_wins_year_sq = 0
        total_runs_log = float(np.log1p(total_after_3))
        striker_sr_sum_log = 0.0
        batting_team, bowling_team = "Unknown", "Unknown"

    # Map teams to strengths (Default to 2)
    bat_score = BATTING_STRENGTH.get(batting_team, 2)
    bowl_score = BOWLING_STRENGTH.get(bowling_team, 2)

    top100_after_3 = 1 if total_after_3 >= TOP100_AFTER_3_THRESHOLD else 0
    top50_after_3 = 1 if total_after_3 >= TOP50_AFTER_3_THRESHOLD else 0
    top25_after_3 = 1 if total_after_3 >= TOP25_AFTER_3_THRESHOLD else 0
    top100_after_3_sq = top100_after_3 * top100_after_3
    top50_after_3_sq = top50_after_3 * top50_after_3
    top25_after_3_sq = top25_after_3 * top25_after_3

    # --- 3. Compute Prediction ---
    predicted_score = (
        INTERCEPT + 
        (over_0_runs * WEIGHT_O0_RUNS) + 
        (over_1_runs * WEIGHT_O1_RUNS) + 
        (over_2_runs * WEIGHT_O2_RUNS) + 
        (over_0_runs_sq * WEIGHT_O0_RUNS_SQ) +
        (over_1_runs_sq * WEIGHT_O1_RUNS_SQ) +
        (over_2_runs_sq * WEIGHT_O2_RUNS_SQ) +
        (dot_balls * WEIGHT_DOT_BALLS) +
        (dot_balls_sq * WEIGHT_DOT_BALLS_SQ) +
        (wickets_after_3 * WEIGHT_WICKETS) + 
        (wickets_after_3_sq * WEIGHT_WICKETS_AFTER_3_SQ) +
        (wickets_squared * WEIGHT_WICKETS_SQUARED) +
        (wickets_squared_sq * WEIGHT_WICKETS_SQUARED_SQ) +
        (dismissed_sr_sum * WEIGHT_DISMISSED_SR) +
        (dismissed_sr_sum_sq * WEIGHT_DISMISSED_SR_SQ) +
        (boundaries_4s6s * WEIGHT_BOUNDARIES_4S6S) +
        (boundaries_4s6s_sq * WEIGHT_BOUNDARIES_4S6S_SQ) +
        (batters_above_20 * WEIGHT_BATTERS_ABOVE_20) +
        (batters_above_20_sq * WEIGHT_BATTERS_ABOVE_20_SQ) +
        (team_wins_year * WEIGHT_TEAM_WINS_YEAR) +
        (team_wins_year_sq * WEIGHT_TEAM_WINS_YEAR_SQ) +
        (total_runs_log * WEIGHT_TOTAL_RUNS_LOG) +
        (striker_sr_sum_log * WEIGHT_STRIKER_SR_SUM_LOG) +
        (top100_after_3 * WEIGHT_TOP100_AFTER_3) +
        (top50_after_3 * WEIGHT_TOP50_AFTER_3) +
        (top25_after_3 * WEIGHT_TOP25_AFTER_3) +
        (top100_after_3_sq * WEIGHT_TOP100_AFTER_3_SQ) +
        (top50_after_3_sq * WEIGHT_TOP50_AFTER_3_SQ) +
        (top25_after_3_sq * WEIGHT_TOP25_AFTER_3_SQ) +
        (bat_score * WEIGHT_BAT_STRENGTH) + 
        (bowl_score * WEIGHT_BOWL_STRENGTH)
    )
    
    return float(predicted_score)