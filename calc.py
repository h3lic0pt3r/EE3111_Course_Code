import pandas as pd
import numpy as np
from io import StringIO
import os

def calculate_and_export():
    print("Loading data to calculate data-driven matchups...")
    df_raw = pd.read_pickle("data/processed/all_data.pkl")
    
    records = []
    
    for index, row in df_raw.iterrows():
        try:
            # Extract Venue
            match_info_lines = str(row['match_data']).strip().split('\n')
            venue_name = "Unknown"
            for line in match_info_lines:
                parts = line.split(',')
                if len(parts) >= 3 and parts[0] == 'info' and parts[1] == 'venue':
                    venue_name = ",".join(parts[2:]).strip() 
                    
            # Extract Teams
            ball_df = pd.read_csv(StringIO(row['ball_by_ball_data']))
            batting_team = ball_df["batting_team"].iloc[0] if "batting_team" in ball_df.columns else "Unknown"
            bowling_team = ball_df["bowling_team"].iloc[0] if "bowling_team" in ball_df.columns else "Unknown"
            
            records.append({
                'venue': venue_name,
                'matchup': f"{batting_team}_vs_{bowling_team}",
                'target': row['target']
            })
        except Exception:
            continue
            
    df = pd.DataFrame(records)
    
    # --- Calculate Venue Categories (2 Categories: High/Low) ---
    venue_stats = df.groupby('venue')['target'].agg(['mean', 'count'])
    venue_stats = venue_stats[venue_stats['count'] >= 3]
    
    venue_stats['rank'] = venue_stats['mean'].rank(method='first')
    # q=2 splits perfectly at the median
    venue_stats['category'] = pd.qcut(venue_stats['rank'], q=2, labels=['Low', 'High'])
    venue_dict = venue_stats['category'].to_dict()
            
    # --- Calculate Matchup Categories (3 Categories: Weak/Mid/Strong) ---
    matchup_stats = df.groupby('matchup')['target'].agg(['mean', 'count'])
    matchup_stats = matchup_stats[matchup_stats['count'] >= 2]
    
    matchup_stats['rank'] = matchup_stats['mean'].rank(method='first')
    matchup_stats['category'] = pd.qcut(matchup_stats['rank'], q=3, labels=['Weak', 'Mid', 'Strong'])
    matchup_dict = matchup_stats['category'].to_dict()

    # --- Export to matchups.py ---
    with open("matchups.py", "w") as f:
        f.write("# Automatically generated data-driven categorizations\n\n")
        f.write("VENUE_SCORING = {\n")
        for k, v in venue_dict.items(): f.write(f"    '{k.replace('"', "").replace("'", "")}': '{v}',\n")
        f.write("}\n\n")
        
        f.write("MATCHUP_SCORING = {\n")
        for k, v in matchup_dict.items(): f.write(f"    '{k}': '{v}',\n")
        f.write("}\n")
        
    print("Success! 2-Tier Venue and 3-Tier Matchup categories written to matchups.py")

if __name__ == "__main__":
    calculate_and_export()