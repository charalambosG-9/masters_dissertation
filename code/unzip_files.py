import os
import zipfile
from myconfig import data_dir

with zipfile.ZipFile(os.path.join(data_dir, 'Data.zip'), 'r') as zip_ref:
    zip_ref.extractall(data_dir)