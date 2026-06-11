import cdsapi
from pathlib import Path

ATLANTIC_NORTH = 60.0
ATLANTIC_SOUTH = -5.0
ATLANTIC_EAST = 20.0
ATLANTIC_WEST = -100.0

ATLANTIC_AREA = [
    ATLANTIC_NORTH,
    ATLANTIC_WEST,
    ATLANTIC_SOUTH,
    ATLANTIC_EAST
]

ERA5_VARIABLES = {}

def download_era5(year: int, variables: list[str], output_dir: Path) -> Path:
    """Download ERA5 data for a given year and geographic region.

    Parameters
    ----
    year : int
        Year of ERA5 data to download.
    
    variables : list[str]
        ERA5 variables to retrieve.

    output_dir : Path
        Directory where the NetCDF (or GRIB?) file will be saved.
    
    Returns
    ----
    Path
        Path to the downloaded NetCDF (or GRIB?) file.
    """
    pass