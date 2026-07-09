from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

__authors__ = Path(__file__).parent.joinpath("AUTHORS").read_text()

try:
    __version__ = version('computervision')
except PackageNotFoundError:
    # package is not installed
    __version__ = 'unknown'

def main() -> None:
    print('Hello from the CCB computervision project!')