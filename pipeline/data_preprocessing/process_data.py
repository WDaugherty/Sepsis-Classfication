import os
import re
import pandas as pd
import sys

def process_header_files(header_files, folder_path):
    records = []

    for header_file in header_files:
        with open(os.path.join(folder_path, header_file), 'r') as f:
            content = f.read()

        record_name = header_file[:-4]  # Remove the '.hea' extension

        # Extract subject_id from the record_name
        subject_id = int(re.findall(r'\d+', record_name)[0])

        # Extract sample frequency and record duration
        sample_frequency_match = re.findall(r'^\S+ \d+ ([\d.]+)\/(\d+)', content, re.MULTILINE)
        sample_frequency_match_2 = re.findall(r'^\S+ \d+ (\d+)', content, re.MULTILINE)
        record_duration_match = re.findall(r'\d+', content)
        record_duration = int(record_duration_match[-1]) if record_duration_match else None

        if not sample_frequency_match and not sample_frequency_match_2:
            print(f"Cannot find sample frequency or record duration in header file {header_file}:\n{content}")
            continue

        if sample_frequency_match:
            sample_frequency = float(sample_frequency_match[0][0]) / float(sample_frequency_match[0][1])
        else:
            sample_frequency = int(sample_frequency_match_2[0])

        if not record_duration_match:
            print(f"Cannot find record duration in header file {header_file}:\n{content}")
            continue

        record_duration = int(record_duration_match[0])

        # Calculate signal length
        signal_length = int(sample_frequency * record_duration)

        records.append({
            'subject_id': subject_id,
            'record_name': record_name,
            'signal_length': signal_length
        })

    return pd.DataFrame(records)


# Initialize an empty dataframe
master_df = pd.DataFrame()

# Set the base directory path
base_dir = '/path/to/Sepsis-Classfication/project/data/raw_data/archive.physionet.org/physiobank/database/mimic3wdb/matched'

# Iterate through all the folders
for root, dirs, files in os.walk(base_dir):
    # Check if there are header files in the current folder
    header_files = [f for f in files if f.endswith('.hea')]

    if header_files:
        # Read and process the header files to create dataframes
        folder_df = process_header_files(header_files, root)

        # Append the folder dataframe to the master dataframe
        master_df = pd.concat([master_df, folder_df], ignore_index=True)

# Perform the remaining operations on the master dataframe

admisions_df = pd.read_csv('pipeline/data_preprocessing/data/demo_data/physionet.org/files/mimiciii-demo/1.4/ADMISSIONS.csv')
patients_df = pd.read_csv('pipeline/data_preprocessing/data/demo_data/physionet.org/files/mimiciii-demo/1.4/PATIENTS.csv')

# Merge the admission and patient data based on the patient ID
merged_df = pd.merge(admisions_df, patients_df, on='subject_id')

# Filter the merged dataset to include only the columns of interest
merged_df = merged_df[['subject_id', 'hadm_id', 'admittime', 'dob', 'gender']]

# Convert the date/time columns to datetime format
merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
merged_df['dob'] = pd.to_datetime(merged_df['dob'])


# Save the merged dataframe to pipeline/data_preprocessing/data/processed_data
merged_df.to_csv('pipeline/data_preprocessing/data/processed_data/merged_data.csv', index=False)