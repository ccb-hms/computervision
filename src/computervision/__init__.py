from pathlib import Path

# you can use os.path and open() as well
__version__ = Path(__file__).parent.joinpath("VERSION").read_text()
__authors__ = Path(__file__).parent.joinpath("AUTHORS").read_text()

def main() -> None:
    print("Hello from the CCB computervision project!")
