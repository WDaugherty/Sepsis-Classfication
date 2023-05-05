#Imports necessary files 
import os

# Create raw_data directory if it doesn't exist
raw_data_dir = os.path.join('project', 'data', 'raw_data')
if not os.path.exists(raw_data_dir):
    os.makedirs(raw_data_dir)

# Load targeted waveform data for patients in both the demo data and the matched waveform directories
patient_ids_01 = ['p010013','p010042','p010045','p010061','p010083','p010124']
for pid in patient_ids_01:
    url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/p01/{pid}/"
    os.system(f"wget -r -N -c -np {url} -P {raw_data_dir}")

patient_ids_04 = ['p040601','p041976','p042033','p042075','p042199','p042302','p043798','p043870','p044083']
for pid in patient_ids_04:
    url = f"https://archive.physionet.org/physiobank/database/mimic3wdb/matched/p04/{pid}/"
    os.system(f"wget -r -N -c -np {url} -P {raw_data_dir}")

# Download Mimic-III Clinical Demo data
demo_data_dir = os.path.join('project', 'data', 'demo_data')
if not os.path.exists(demo_data_dir):
    os.makedirs(demo_data_dir)
demo_data_url = 'https://physionet.org/files/mimiciii-demo/1.4/'
os.system(f"wget -r -N -c -np {demo_data_url} -P {demo_data_dir}")
