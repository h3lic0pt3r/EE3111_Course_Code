import os
import glob
import argparse
import pandas as pd

def clean_data(start_year, raw_dir="data/raw", out_file="data/processed/all_data.pkl"):
    if not os.path.exists(raw_dir):
        print(f"Error: Raw directory '{raw_dir}' not found. Please ensure your files are there.")
        return

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(f"Scanning matches from {start_year} onwards...")

    dataset = []
    
    # Find all the info files first
    info_files = glob.glob(os.path.join(raw_dir, "*_info.csv"))

    for info_filepath in info_files:
        # Extract the match ID to find the corresponding ball-by-ball file
        match_id = os.path.basename(info_filepath).replace('_info.csv', '')
        ball_filepath = os.path.join(raw_dir, f"{match_id}.csv")
        
        if not os.path.exists(ball_filepath):
            continue
            
        # --- 1. Process the Info File ---
        try:
            with open(info_filepath, 'r', encoding='utf-8') as f:
                info_lines = f.readlines()
        except Exception:
            continue
            
        is_male_match = False
        match_year = 0
        match_month = 0
        match_day = 0
        filtered_info_lines = []
        
        for line in info_lines:
            line_lower = line.lower()
            parts = line.strip().split(',')
            
            # Check gender
            if len(parts) >= 3 and parts[1] == 'gender' and parts[2].lower() == 'male':
                is_male_match = True
                
            # Check year
            if len(parts) >= 3 and parts[1] == 'date':
                try:
                    # Extract YYYY from dates like YYYY-MM-DD
                    year_str = parts[2][:4]
                    month_str = parts[2][5:7]
                    day_str = parts[2][8:10]
                    match_year = int(year_str)
                    match_month = int(month_str)
                    match_day = int(day_str)
                except ValueError:
                    pass
            
            # Remove winner/outcome info to comply with project rules
            if 'winner' not in line_lower and 'outcome' not in line_lower:
                # Strip single and double quotes from the info line to keep names clean
                clean_line = line.replace('"', '').replace("'", "")
                filtered_info_lines.append(clean_line.strip())
                
        # Apply Filters: Must be Male and >= the specified year
        # if not is_male_match or match_year < start_year:
        #     continue

        if match_year in {2009, 2014, 2020, 2021}:
            continue


            
        match_data_str = "\n".join(filtered_info_lines)
        
        # --- 2. Process the Ball-by-Ball File ---
        try:
            df = pd.read_csv(ball_filepath)
        except Exception:
            continue
            
        if 'innings' not in df.columns or 'ball' not in df.columns:
            continue
            
        # Clean quotes from player columns in the ball-by-ball DataFrame
        player_cols = ['batter', 'striker', 'non_striker', 'bowler', 'player_dismissed']
        for col in player_cols:
            if col in df.columns:
                # Use apply to strip quotes from strings while safely ignoring NaNs
                df[col] = df[col].apply(
                    lambda x: str(x).replace('"', '').replace("'", "") if pd.notna(x) else x
                )
                
        first_innings_df = df[df['innings'] == 1]
        if first_innings_df.empty:
            continue
            
        # Target (Y): Runs at 6.0 overs
        powerplay_df = first_innings_df[first_innings_df['ball'] < 6.0]
        
        runs = powerplay_df['runs_off_bat'].sum() if 'runs_off_bat' in powerplay_df.columns else 0
        extras = powerplay_df['extras'].sum() if 'extras' in powerplay_df.columns else 0
        target_y = float(runs + extras)
        
        # Inputs (X): Data available up to end of 3.0 overs
        first_3_overs_df = first_innings_df[first_innings_df['ball'] < 3.0]
        ball_by_ball_str = first_3_overs_df.to_csv(index=False)
        
        for _ in range(max(1, 0 * (match_year - 2020)**3 // 10)):
            dataset.append({
                'match_data': match_data_str,
                'ball_by_ball_data': ball_by_ball_str,
                'target': target_y
            })

    if not dataset:
        print(f"No valid matches found from {start_year} onwards. Check your data format or start year.")
        return

    # Save to pickle
    df_final = pd.DataFrame(dataset)
    df_final.to_pickle(out_file)
    print(f"Success! Saved {len(df_final)} matches from {start_year}-present to {out_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean and filter Cricsheet data by year.")
    parser.add_argument(
        "year", 
        type=int, 
        help="The starting year to include in the dataset (e.g., 2021)"
    )
    
    args = parser.parse_args()
    clean_data(args.year)