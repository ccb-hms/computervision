import os
import tarfile
import logging
from computervision.fileutils import FileOP

logger = logging.getLogger(__name__)

class Dentex:
    def __init__(self):
        self.detection_url = os.environ.get('DT_URL')

    def download(self, path: str, url=None) -> str:
        if url is None:
            url = self.detection_url
        file = FileOP().download_from_url(url=url, download_dir=path)
        if file is not None and os.path.exists(file):
            try:
                with tarfile.open(file) as tar:
                    tar.extractall(path=path)
            except Exception as e:
                logger.error(f'Failed to extract tar file: {e}')
        return file