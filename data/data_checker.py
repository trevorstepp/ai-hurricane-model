import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv("parsed_hurdat2.csv")
    print(df["storm_id"].str[:2].value_counts())