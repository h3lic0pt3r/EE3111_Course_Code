import pandas as pd
import numpy as np
import sys
from submission import predict

def run_backtest(file: str):
    print(f"Loading backtest data on {file}...")
    try:
        df = pd.read_pickle(file)
    except FileNotFoundError:
        print("Data not found. Run clean_data.py first.")
        return

    predictions = []
    actuals = []

    print("Running predictions...")
    for index, row in df.iterrows():
        # Pass data as Python string objects as required by the interface
        pred = predict(row['match_data'], row['ball_by_ball_data'])
        predictions.append(pred)
        actuals.append(row['target'])

    # Calculate RMSE
    preds = np.array(predictions)
    acts = np.array(actuals)
    
    squared_errors = (preds - acts) ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse = np.sqrt(mean_squared_error)
    
    print("-" * 30)
    print(f"Backtest Complete over {len(df)} matches.")
    print(f"Model RMSE: {rmse:.4f}")
    print("-" * 30)
    print("Remember: Lower RMSE is better for your leaderboard ranking!")

if __name__ == "__main__":
    run_backtest("data/processed/all_data.pkl")
    run_backtest("data/processed/training_data.pkl")
    run_backtest("data/processed/testing_data.pkl")