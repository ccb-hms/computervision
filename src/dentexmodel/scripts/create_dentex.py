"""
Script to download and create the dentex disease data set
Andreas Werdich
Center for Computational Biomedicine
"""
import glob
import os
import logging
from pathlib import Path

from dentexmodel.fileutils import FileOP
from dentexmodel.dentexdata import DentexData

# Local directories (change as needed)
dentex_dir = os.path.join(os.environ['HOME'], 'data', 'dentex')
data_dir = os.path.join(dentex_dir, 'dentex_classification')
log_dir = os.path.join(data_dir, 'log')
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(log_dir).mkdir(parents=True, exist_ok=True)

# Create a log file for the operation
dtfmt = '%Y-%m-%d %I:%M:%S %p'
logfmt = '%(asctime)s-%(levelname)s-%(module)s-%(funcName)s-%(lineno)d-"%(message)s"'
log_file = os.path.join(log_dir, 'create_dentex.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format=logfmt, datefmt=dtfmt)
logger = logging.getLogger(name=__name__)


def create_classification_dataset(output_dir):
    pass


def create_detection_dataset(output_dir):
    pass


def create_segmentation_dataset(output_dir):
    pass


def main():
    create_classification_dataset(output_dir=data_dir)


if __name__ == "__main__":
    main()
