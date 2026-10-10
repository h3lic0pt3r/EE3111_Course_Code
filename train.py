import pandas as pd
from sklearn.linear_model import LinearRegression
from io import StringIO
from matchups import MATCHUP_SCORING, VENUE_SCORING
import os
import glob
import numpy as np

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

def prepare_training_data():
    print("Loading data and mapping data-driven categorical features...")
    df_raw = pd.read_pickle("data/processed/all_data.pkl")
    team_wins_by_year = build_team_wins_by_year()
    
    feature_list = []
    
    for index, row in df_raw.iterrows():
        try:
            venue_name = "Unknown"
            for line in str(row['match_data']).split('\n'):
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == 'info' and parts[1] == 'venue':
                    venue_name = ",".join(parts[2:]).strip().replace('"', "").replace("'", "")
                    break
            venue_cat = VENUE_SCORING.get(venue_name, 'Low')

            ball_df = pd.read_csv(StringIO(row['ball_by_ball_data']))
            
            ball_df["runs_off_bat"] = pd.to_numeric(ball_df["runs_off_bat"], errors="coerce").fillna(0)
            ball_df["extras"] = pd.to_numeric(ball_df["extras"], errors="coerce").fillna(0)
            ball_df["total_runs"] = ball_df["runs_off_bat"] + ball_df["extras"]
            
            ball_df['over_index'] = ball_df['ball'].astype(int)
            over_0_runs = ball_df[ball_df['over_index'] == 0]['total_runs'].sum()
            over_1_runs = ball_df[ball_df['over_index'] == 1]['total_runs'].sum()
            over_2_runs = ball_df[ball_df['over_index'] == 2]['total_runs'].sum()
            total_after_3 = over_0_runs + over_1_runs + over_2_runs
            boundaries_4s6s = ((ball_df["runs_off_bat"] == 4) | (ball_df["runs_off_bat"] == 6)).sum()
            
            wickets = ball_df["player_dismissed"].notna().sum() if "player_dismissed" in ball_df.columns else 0
            dot_balls = (ball_df["total_runs"] == 0).sum()
            wickets_squared = wickets * wickets
            over_0_runs_sq = over_0_runs * over_0_runs
            over_1_runs_sq = over_1_runs * over_1_runs
            over_2_runs_sq = over_2_runs * over_2_runs
            dot_balls_sq = dot_balls * dot_balls
            wickets_after_3_sq = wickets * wickets
            wickets_squared_sq = wickets_squared * wickets_squared

            dismissed_sr_sum = 0.0
            if "player_dismissed" in ball_df.columns and "striker" in ball_df.columns:
                dismissed_players = ball_df["player_dismissed"].dropna().unique()
                if len(dismissed_players) > 0:
                    if "wides" in ball_df.columns:
                        wides_col = pd.to_numeric(ball_df["wides"], errors="coerce").fillna(0)
                    else:
                        wides_col = pd.Series(0, index=ball_df.index)
                    for player in dismissed_players:
                        player_mask = ball_df["striker"] == player
                        if not player_mask.any():
                            continue
                        runs_scored = ball_df.loc[player_mask, "runs_off_bat"].sum()
                        balls_faced = (player_mask & (wides_col == 0)).sum()
                        if balls_faced > 0:
                            dismissed_sr_sum += (runs_scored / balls_faced) * 100
            dismissed_sr_sum_sq = dismissed_sr_sum * dismissed_sr_sum
            boundaries_4s6s_sq = boundaries_4s6s * boundaries_4s6s

            batter_runs = {}
            if "striker" in ball_df.columns:
                for _, row_ball in ball_df.iterrows():
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
                if batter_col not in ball_df.columns:
                    continue
                batter = ball_df[batter_col].iloc[0]
                if pd.isna(batter):
                    continue
                player_mask = ball_df["striker"] == batter if "striker" in ball_df.columns else None
                if player_mask is None or not player_mask.any():
                    continue
                runs_scored = ball_df.loc[player_mask, "runs_off_bat"].sum()
                if "wides" in ball_df.columns:
                    wides_col = pd.to_numeric(ball_df["wides"], errors="coerce").fillna(0)
                else:
                    wides_col = pd.Series(0, index=ball_df.index)
                balls_faced = (player_mask & (wides_col == 0)).sum()
                if balls_faced > 0:
                    striker_sr_sum += (runs_scored / balls_faced) * 100
            striker_sr_sum_log = float(np.log1p(striker_sr_sum))
            
            batting_team = ball_df["batting_team"].iloc[0] if "batting_team" in ball_df.columns else "Unknown"
            bowling_team = ball_df["bowling_team"].iloc[0] if "bowling_team" in ball_df.columns else "Unknown"

            match_year = None
            for line in str(row['match_data']).split('\n'):
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == 'info' and parts[1] == 'date':
                    try:
                        match_year = int(parts[2][:4])
                    except ValueError:
                        match_year = None
                    break
            team_wins_year = team_wins_by_year.get((match_year, batting_team), 0) if match_year is not None else 0
            team_wins_year_sq = team_wins_year * team_wins_year
            
            matchup_key = f"{batting_team}_vs_{bowling_team}"
            matchup_cat = MATCHUP_SCORING.get(matchup_key, 'Mid') # Matchups still have a 'Mid'
            
            feature_list.append({
                'over_0_runs': over_0_runs,
                'over_1_runs': over_1_runs,
                'over_2_runs': over_2_runs,
                'over_0_runs_sq': over_0_runs_sq,
                'over_1_runs_sq': over_1_runs_sq,
                'over_2_runs_sq': over_2_runs_sq,
                'total_after_3': total_after_3,
                'total_runs_log': total_runs_log,
                'boundaries_4s6s': boundaries_4s6s,
                'boundaries_4s6s_sq': boundaries_4s6s_sq,
                'dot_balls': dot_balls,
                'dot_balls_sq': dot_balls_sq,
                'wickets_after_3': wickets,
                'wickets_after_3_sq': wickets_after_3_sq,
                'wickets_squared': wickets_squared,
                'wickets_squared_sq': wickets_squared_sq,
                'dismissed_sr_sum': dismissed_sr_sum,
                'dismissed_sr_sum_sq': dismissed_sr_sum_sq,
                'batters_above_20': batters_above_20,
                'batters_above_20_sq': batters_above_20_sq,
                'team_wins_year': team_wins_year,
                'team_wins_year_sq': team_wins_year_sq,
                'striker_sr_sum_log': striker_sr_sum_log,
                'matchup_strength': matchup_cat,
                'venue_scoring': venue_cat,
                'target': row['target']
            })
        except Exception:
            continue
            
    return pd.DataFrame(feature_list)

def fit_and_export_model():
    df = prepare_training_data()

    totals = df['total_after_3'].sort_values(ascending=False).reset_index(drop=True)
    if len(totals) >= 100:
        top100_threshold = float(totals.iloc[99])
    elif len(totals) > 0:
        top100_threshold = float(totals.iloc[-1])
    else:
        top100_threshold = 0.0

    if len(totals) >= 50:
        top50_threshold = float(totals.iloc[49])
    elif len(totals) > 0:
        top50_threshold = float(totals.iloc[-1])
    else:
        top50_threshold = 0.0

    if len(totals) >= 25:
        top25_threshold = float(totals.iloc[24])
    elif len(totals) > 0:
        top25_threshold = float(totals.iloc[-1])
    else:
        top25_threshold = 0.0

    df['top100_after_3'] = (df['total_after_3'] >= top100_threshold).astype(int)
    df['top50_after_3'] = (df['total_after_3'] >= top50_threshold).astype(int)
    df['top25_after_3'] = (df['total_after_3'] >= top25_threshold).astype(int)
    df['top100_after_3_sq'] = df['top100_after_3'] * df['top100_after_3']
    df['top50_after_3_sq'] = df['top50_after_3'] * df['top50_after_3']
    df['top25_after_3_sq'] = df['top25_after_3'] * df['top25_after_3']

    X = pd.get_dummies(df[['over_0_runs', 'over_1_runs', 'over_2_runs', 'over_0_runs_sq', 'over_1_runs_sq',
                           'over_2_runs_sq', 'dot_balls', 'dot_balls_sq', 'wickets_after_3', 'wickets_after_3_sq',
                           'wickets_squared', 'wickets_squared_sq', 'dismissed_sr_sum', 'dismissed_sr_sum_sq',
                           'total_runs_log', 'striker_sr_sum_log', 'boundaries_4s6s', 'boundaries_4s6s_sq',
                           'top100_after_3', 'top50_after_3', 'top25_after_3', 'top100_after_3_sq',
                           'top50_after_3_sq', 'top25_after_3_sq', 'batters_above_20', 'batters_above_20_sq',
                           'team_wins_year', 'team_wins_year_sq', 'matchup_strength', 'venue_scoring']])
    y = df['target']
    
    print("Fitting Model...")
    model = LinearRegression()
    model.fit(X, y)
    
    coefs = dict(zip(X.columns, model.coef_))
    
    matchup_weights = {k.replace('matchup_strength_', ''): v for k, v in coefs.items() if k.startswith('matchup_strength_')}
    venue_weights = {k.replace('venue_scoring_', ''): v for k, v in coefs.items() if k.startswith('venue_scoring_')}
    with open("weights.py", "w") as f:
        f.write("# Automatically generated weights\n\n")
        f.write(f"INTERCEPT = {model.intercept_:.4f}\n")
        f.write(f"WEIGHT_O0_RUNS = {coefs.get('over_0_runs', 0):.4f}\n")
        f.write(f"WEIGHT_O1_RUNS = {coefs.get('over_1_runs', 0):.4f}\n")
        f.write(f"WEIGHT_O2_RUNS = {coefs.get('over_2_runs', 0):.4f}\n")
        f.write(f"WEIGHT_O0_RUNS_SQ = {coefs.get('over_0_runs_sq', 0):.4f}\n")
        f.write(f"WEIGHT_O1_RUNS_SQ = {coefs.get('over_1_runs_sq', 0):.4f}\n")
        f.write(f"WEIGHT_O2_RUNS_SQ = {coefs.get('over_2_runs_sq', 0):.4f}\n")
        f.write(f"WEIGHT_DOT_BALLS = {coefs.get('dot_balls', 0):.4f}\n")
        f.write(f"WEIGHT_DOT_BALLS_SQ = {coefs.get('dot_balls_sq', 0):.4f}\n")
        f.write(f"WEIGHT_WICKETS = {coefs.get('wickets_after_3', 0):.4f}\n\n")
        f.write(f"WEIGHT_WICKETS_AFTER_3_SQ = {coefs.get('wickets_after_3_sq', 0):.4f}\n\n")
        f.write(f"WEIGHT_WICKETS_SQUARED = {coefs.get('wickets_squared', 0):.4f}\n\n")
        f.write(f"WEIGHT_WICKETS_SQUARED_SQ = {coefs.get('wickets_squared_sq', 0):.4f}\n\n")
        f.write(f"WEIGHT_DISMISSED_SR = {coefs.get('dismissed_sr_sum', 0):.4f}\n\n")
        f.write(f"WEIGHT_DISMISSED_SR_SQ = {coefs.get('dismissed_sr_sum_sq', 0):.4f}\n\n")
        f.write(f"WEIGHT_TOTAL_RUNS_LOG = {coefs.get('total_runs_log', 0):.4f}\n")
        f.write(f"WEIGHT_STRIKER_SR_SUM_LOG = {coefs.get('striker_sr_sum_log', 0):.4f}\n")
        f.write(f"WEIGHT_BOUNDARIES_4S6S = {coefs.get('boundaries_4s6s', 0):.4f}\n")
        f.write(f"WEIGHT_BOUNDARIES_4S6S_SQ = {coefs.get('boundaries_4s6s_sq', 0):.4f}\n")
        f.write(f"WEIGHT_BATTERS_ABOVE_20 = {coefs.get('batters_above_20', 0):.4f}\n")
        f.write(f"WEIGHT_BATTERS_ABOVE_20_SQ = {coefs.get('batters_above_20_sq', 0):.4f}\n")
        f.write(f"WEIGHT_TEAM_WINS_YEAR = {coefs.get('team_wins_year', 0):.4f}\n")
        f.write(f"WEIGHT_TEAM_WINS_YEAR_SQ = {coefs.get('team_wins_year_sq', 0):.4f}\n")
        f.write(f"WEIGHT_TOP100_AFTER_3 = {coefs.get('top100_after_3', 0):.4f}\n")
        f.write(f"WEIGHT_TOP50_AFTER_3 = {coefs.get('top50_after_3', 0):.4f}\n")
        f.write(f"WEIGHT_TOP25_AFTER_3 = {coefs.get('top25_after_3', 0):.4f}\n")
        f.write(f"WEIGHT_TOP100_AFTER_3_SQ = {coefs.get('top100_after_3_sq', 0):.4f}\n")
        f.write(f"WEIGHT_TOP50_AFTER_3_SQ = {coefs.get('top50_after_3_sq', 0):.4f}\n")
        f.write(f"WEIGHT_TOP25_AFTER_3_SQ = {coefs.get('top25_after_3_sq', 0):.4f}\n")
        f.write(f"TOP100_AFTER_3_THRESHOLD = {top100_threshold:.4f}\n")
        f.write(f"TOP50_AFTER_3_THRESHOLD = {top50_threshold:.4f}\n")
        f.write(f"TOP25_AFTER_3_THRESHOLD = {top25_threshold:.4f}\n\n")
        f.write("MATCHUP_WEIGHTS = {\n")
        for cat, w in matchup_weights.items(): f.write(f"    '{cat}': {w:.4f},\n")
        f.write("}\n\n")

        f.write("VENUE_WEIGHTS = {\n")
        for cat, w in venue_weights.items(): f.write(f"    '{cat}': {w:.4f},\n")
        f.write("}\n")
        
        

    print("Successfully wrote updated coefficients to weights.py")

if __name__ == "__main__":
    fit_and_export_model()