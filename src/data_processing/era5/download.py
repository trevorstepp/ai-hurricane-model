import cdsapi
from pathlib import Path

from src.constants import ERA5_DIR, ATLANTIC_AREA, ERA5_VARIABLES

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