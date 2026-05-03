import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(input_path, train_file_path, test_file_path):
    df = pd.read_csv(input_path)

    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=42
    )

    train_df.to_csv(train_file_path, index=False)
    test_df.to_csv(test_file_path, index=False)

    print("Split done: train/test")