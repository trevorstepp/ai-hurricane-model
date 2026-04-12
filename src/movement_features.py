import pandas as pd

def add_movement_features(original_df: pd.DataFrame, csv_name: str) -> pd.DataFrame:
    """
    Compute changes in latitude and longitude for each storm and add them as features.

    Rows with missing values in the computed features are removed.

    Parameters
    ----
    original_df : pd.DataFrame
        DataFrame containing storm track data, including `storm_id`, `lat`, and `lon`.

    Returns
    ----
    pd.DataFrame
        A new DataFrame with added `dlat` and `dlon` columns, where each row represents
        the change in position from the previous timestep within the same storm.
    """
    df = original_df.copy()
    # get change in lat/lon between successive entries for each storm
    df["dlat"] = df.groupby("storm_id")["lat"].diff()
    df["dlon"] = df.groupby("storm_id")["lon"].diff()

    # drop any row with missing dlat and dlon (first row)
    df = df.dropna(subset=["dlat", "dlon"]).reset_index(drop=True)
    #df = df.dropna().reset_index(drop=True)
    df.to_csv(csv_name, index=False)
    return df