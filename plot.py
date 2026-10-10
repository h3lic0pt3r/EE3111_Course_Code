import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
import os

try:
    from matchups import VENUE_SCORING, MATCHUP_SCORING
except ImportError:
    print("Error: matchups.py not found. Please run calc.py first.")
    exit()

def extract_features_for_plotting(pickle_path):
    print(f"Loading data from {pickle_path}...")
    if not os.path.exists(pickle_path):
        return None
        
    df_raw = pd.read_pickle(pickle_path)
    extracted_data = []
    
    for index, row in df_raw.iterrows():
        try:
            match_info_lines = str(row['match_data']).strip().split('\n')
            venue_name = "Unknown"
            for line in match_info_lines:
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == 'info' and parts[1] == 'venue':
                    venue_name = ",".join(parts[2:]).strip() 
            
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
            
            wickets = ball_df["player_dismissed"].notna().sum() if "player_dismissed" in ball_df.columns else 0
            dot_balls = (ball_df["total_runs"] == 0).sum()
            
            batting_team = ball_df["batting_team"].iloc[0] if "batting_team" in ball_df.columns else "Unknown"
            bowling_team = ball_df["bowling_team"].iloc[0] if "bowling_team" in ball_df.columns else "Unknown"
            
            matchup_key = f"{batting_team}_vs_{bowling_team}"
            matchup_cat = MATCHUP_SCORING.get(matchup_key, 'Mid')
            
            extracted_data.append({
                'Over 0 Runs': over_0_runs,
                'Over 1 Runs': over_1_runs,
                'Over 2 Runs': over_2_runs,
                'Total Runs (After 3)': total_after_3,
                'Dot Balls': dot_balls,
                'Wickets (After 3)': wickets,
                'Matchup Strength': matchup_cat,
                'Venue Scoring Type': venue_cat,
                'Target (Runs at 6 Overs)': row['target']
            })
        except Exception:
            continue
            
    return pd.DataFrame(extracted_data)

def generate_plots():
    df = extract_features_for_plotting("data/processed/all_data.pkl")
    if df is None or df.empty: return

    print("Generating feature vs target plots...")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    fig.suptitle("Data-Driven Feature Analysis (2-Tier Venue Scoring)", fontsize=18, fontweight='bold', y=0.98)

    numeric_features = ['Over 0 Runs', 'Over 1 Runs', 'Over 2 Runs', 'Total Runs (After 3)', 'Dot Balls']
    for i, feature in enumerate(numeric_features):
        ax = axes[i // 2, i % 2]
        sns.regplot(data=df, x=feature, y='Target (Runs at 6 Overs)', ax=ax, scatter_kws={'alpha': 0.3, 'color': '#2b7bba'}, line_kws={'color': '#e74c3c', 'linewidth': 2})
        ax.set_title(f"{feature} Correlation", fontsize=14)

    cat_features = [
        ('Wickets (After 3)', None), 
        ('Matchup Strength', ['Weak', 'Mid', 'Strong']), 
        ('Venue Scoring Type', ['Low', 'High']) # Updated explicitly to 2 categories
    ]
    
    for i, (feature, order) in enumerate(cat_features):
        idx = i + 5 
        ax = axes[idx // 2, idx % 2]
        sns.boxplot(data=df, x=feature, y='Target (Runs at 6 Overs)', ax=ax, order=order, palette="Set2")
        ax.set_title(f"Target Distribution by {feature}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig("datadriven_feature_correlations.png", dpi=300)
    print("Plot saved successfully!")
    plt.show()

if __name__ == "__main__":
    generate_plots()