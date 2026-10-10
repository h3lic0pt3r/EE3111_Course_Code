import pandas as pd
import argparse
import os
from sklearn.model_selection import train_test_split

def split_dataset(train_pct):
    input_path = "data/processed/all_data.pkl"
    train_path = "data/processed/training_data.pkl"
    test_path = "data/processed/testing_data.pkl"

    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Please make sure your data is generated and named correctly.")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_pickle(input_path)
    total_matches = len(df)
    print(f"Total matches found: {total_matches}")

    # Calculate test size proportion (e.g., 80% train -> 0.20 test size)
    test_size = 1.0 - (train_pct / 100.0)

    # Perform the split
    print(f"Splitting data: {train_pct}% Train / {100 - train_pct:.1f}% Test...")
    train_df, test_df = train_test_split(df, test_size=test_size)

    # Save the splits
    train_df.to_pickle(train_path)
    test_df.to_pickle(test_path)

    print("\n--- Split Successful ---")
    print(f"Training set: {len(train_df)} matches -> Saved to {train_path}")
    print(f"Testing set:  {len(test_df)} matches -> Saved to {test_path}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Split all_data.pkl into train and test sets.")
    parser.add_argument(
        "train_percentage", 
        type=float, 
        help="Percentage of data to use for training (e.g., 80 for an 80/20 split)."
    )
    
    args = parser.parse_args()

    # Validate percentage
    if args.train_percentage <= 0 or args.train_percentage >= 100:
        print("Error: Train percentage must be strictly between 0 and 100.")
    else:
        split_dataset(args.train_percentage)