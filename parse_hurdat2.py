import pandas as pd
from datetime import datetime

MISSING_PRESSURE = -999
MISSING_WIND = -99

def convert_lat_lon(value: str) -> float:
    """
    Convert latitude and longitude (number and direction) to only a number.
    Example: `28.0N` becomes `28.0` and `91.8W` becomes `-91.8`.

    Parameters
    ----
    value : str
        The latitude or longitude from hurdat2.txt.

    Returns
    ----
    num : float
        The converted latitude or longitude.
    """
    num = float(value[:-1])
    direction = value[-1]  # N, S, E, W

    # take N, E to be positive and S, W to be negative
    if direction in ["S", "W"]:
        num = -num
    return num

def parse_hurdat2(file_path: str, csv_name: str) -> None:
    rows = []
    current_storm_id = None

    # read each line, only keeping those with content
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            content = [i.strip() for i in line.split(",")]

            # each line is either a header line or a data line
            # if header, first characters will indicate basin
            if(len(content[0]) >= 2 and content[0][:2] in ["AL", "EP", "CP"]):
                current_storm_id = content[0]
                continue

            # data line
            try:
                if len(content) < 8:
                    continue

                date_str = content[0]
                utc_str = content[1]
                # skipping record identifier and status of system
                lat_str = content[4]
                lon_str = content[5]
                wind = int(content[6])
                pressure = int(content[7])

                # -999 pressure means no measurement
                if pressure == MISSING_PRESSURE:
                    pressure = None

                # -99 wind means no measurement
                if wind == MISSING_WIND:
                    wind = None

                # use date_str and utc_str to get datetime
                dt = datetime.strptime(date_str + utc_str, "%Y%m%d%H%M")

                # convert lat and lon
                lat = convert_lat_lon(lat_str)
                lon = convert_lat_lon(lon_str)

                rows.append({
                    "storm_id": current_storm_id,
                    "datetime": dt,
                    "lat": lat,
                    "lon": lon,
                    "wind": wind,
                    "pressure": pressure,
                    "year": dt.year
                })

            except Exception as e:
                print(f"Error: {e} | Line: {line}")
                continue
    
    df = pd.DataFrame(rows)
    # should already be sorted, but sort to be safe
    df = df.sort_values(["storm_id", "datetime"]).reset_index(drop=True)

    # lat/lon sanity check
    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()
    df.to_csv(csv_name, index=False)

if __name__ == "__main__":
    parse_hurdat2("hurdat2.txt", "parsed_hurdat2.csv")