from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
HURDAT2_DIR = DATA_DIR / "hurdat2"
ERA5_DIR = DATA_DIR / "era5"

for directory in [MODELS_DIR, DATA_DIR, HURDAT2_DIR, ERA5_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

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

DEFAULT_ERA5_VARIABLES = ["sst", "msl", "u10", "v10"]

ERA5_VARIABLES = {}