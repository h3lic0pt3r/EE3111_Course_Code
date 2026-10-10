import os
import glob
import pandas as pd
import numpy as np
from io import StringIO
from main import predict


def build_2026_dataset(raw_dir="data/raw"):
    if not os.path.exists(raw_dir):
        return []

    dataset = []
    info_files = glob.glob(os.path.join(raw_dir, "*_info.csv"))

    for info_filepath in info_files:
        match_id = os.path.basename(info_filepath).replace('_info.csv', '')
        ball_filepath = os.path.join(raw_dir, f"{match_id}.csv")
        if not os.path.exists(ball_filepath):
            continue

        try:
            with open(info_filepath, 'r', encoding='utf-8') as f:
                info_lines = f.readlines()
        except Exception:
            continue

        match_year = 0
        filtered_info_lines = []

        for line in info_lines:
            line_lower = line.lower()
            parts = line.strip().split(',')

            if len(parts) >= 3 and parts[1] == 'date':
                try:
                    year_str = parts[2][:4]
                    match_year = int(year_str)
                except ValueError:
                    pass

            if 'winner' not in line_lower and 'outcome' not in line_lower:
                clean_line = line.replace('"', '').replace("'", "")
                filtered_info_lines.append(clean_line.strip())

        if match_year != 2026:
            continue

        match_data_str = "\n".join(filtered_info_lines)

        try:
            df = pd.read_csv(ball_filepath)
        except Exception:
            continue

        if 'innings' not in df.columns or 'ball' not in df.columns:
            continue

        first_innings_df = df[df['innings'] == 1]
        if first_innings_df.empty:
            continue

        powerplay_df = first_innings_df[first_innings_df['ball'] < 6.0]
        runs = powerplay_df['runs_off_bat'].sum() if 'runs_off_bat' in powerplay_df.columns else 0
        extras = powerplay_df['extras'].sum() if 'extras' in powerplay_df.columns else 0
        target_y = float(runs + extras)

        first_3_overs_df = first_innings_df[first_innings_df['ball'] < 3.0]
        ball_by_ball_str = first_3_overs_df.to_csv(index=False)

        dataset.append({
            'match_id': match_id,
            'match_data': match_data_str,
            'ball_by_ball_data': ball_by_ball_str,
            'target': target_y
        })

    return dataset


def run_backtest_2026():
    data = build_2026_dataset()
    if not data:
        print("No 2026 matches found in data/raw.")
        return

    predictions = []
    actuals = []

    print("match_id,predicted,actual")
    for row in data:
        pred = predict(row['match_data'], row['ball_by_ball_data'])
        predictions.append(pred)
        actuals.append(row['target'])
        print(f"{row['match_id']},{pred:.2f},{row['target']:.2f}")

    preds = np.array(predictions)
    acts = np.array(actuals)
    rmse = np.sqrt(np.mean((preds - acts) ** 2))

    print(f"Backtest Complete over {len(data)} matches.")
    print(f"Model RMSE: {rmse:.4f}")


if __name__ == "__main__":
    run_backtest_2026()
