"""
   Test the AWS download of the Roboflow dental data
"""
import os
import glob
import tempfile
import pytest
from computervision.fileutils import FileOP



class Testdownload:

   def test_download(self):
      dataset_name = 'dataset_dental_roboflow'
      url_base = 'https://dsets.s3.us-east-1.amazonaws.com/classification_datasets'
      url = os.path.join(url_base, f'{dataset_name}.tar.gz')
      with tempfile.TemporaryDirectory() as tmpdir:
         print(f"Temporary directory created at: {tmpdir}")
         file = FileOP().download_from_url(url=url, download_dir=tmpdir)
         assert os.path.exists(file)