import numpy as np
import os
import shutil
import posixpath
import wfdb
import pandas as pd
import re

# Create data directory if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# Load targetted data for patients in both the demo data and the matched waveform directories
patient_ids_01 = ['p010013','p010042','p010045','p010061','p010083','p010124']

for pid in patient_ids_01:
    url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/p01/{pid}/"
    wfdb.dl_database(url, 'data')

patient_ids_04 = ['p040601','p041976','p042033','p042075','p042199','p042302','p043798','p043870','p044083']

for pid in patient_ids_04:
    url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/p04/{pid}/"
    wfdb.dl_database(url, 'data')