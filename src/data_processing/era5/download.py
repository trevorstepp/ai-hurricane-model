import cdsapi
from pathlib import Path

from src.constants import ERA5_DIR, ATLANTIC_AREA, DEFAULT_ERA5_VARIABLES, ERA5_VARIABLES

def get_era5_variable(var: str) -> str:
    """
    Return the CDS API variable name corresponding to an alias.

    Parameters
    ----
    var : str
        ERA5 variable alias (for example, "sst" or "msl").
    
    Returns
    ----
    str
        The CDS API variable name corresponding to the provided alias.
    
    Raises
    ----
    ValueError
        If `var` is not a recognized ERA5 variable alias.
    """
    try:
        return ERA5_VARIABLES[var]
    except KeyError as e:
        raise ValueError(f"Unknown ERA5 variable: '{var}'") from e

def download_era5_year(year: int, variables: list[str] | None = None) -> Path:
    """
    Download ERA5 reanalysis data for a given year over the Atlantic hurricane basin.

    Parameters
    ----
    year : int
        Year of ERA5 data to download.
    
    variables : list[str], optional
        ERA5 variables to retrieve. If None, defaults to the variables specified by
        `src.constants.DEFAULT_ERA5_VARIABLES`.
    
    Returns
    ----
    Path
        Path to the downloaded NetCDF (or GRIB?) file.
    """
    variables = variables if variables is not None else DEFAULT_ERA5_VARIABLES

    # ValueError will be raised if any inputs are not in the ERA5_VARIABLES dictionary
    cds_variables = [get_era5_variable(var) for var in variables]